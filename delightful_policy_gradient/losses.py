"""Loss functions for policy gradient experiments.

Each loss is a plain class with __call__(logits, batch) -> (loss, metrics_dict).

From the literature:
  CE         — supervised cross-entropy oracle (upper bound)
  REINFORCE  — Williams 1992, no off-policy correction
  PG         — importance-weighted policy gradient
  DG         — Delightful Policy Gradient (Osband 2026)
  Kondo      — compute-efficient DG ("Does This Gradient Spark Joy?", Osband 2026)
  DAPO       — Decoupled clip + dynamic sampling (ByteDance, NeurIPS 2025)
  PMDMean    — Policy Mirror Descent with mean-reward partition approx (Kimi k1.5 lineage, 2025)

logits shape matches batch.actions:
  - bandit:     logits [B, A],    actions [B]
  - sequential: logits [B, T, V], actions [B, T]
"""

import math

import torch
import torch.nn.functional as F


# ── Helpers ──────────────────────────────────────────────────────────────────


def compute_baseline(kind: str, probs: torch.Tensor) -> torch.Tensor:
    """Baseline from current policy probs (no label access). Returns [B]."""
    if kind == 'zero':
        return torch.zeros(probs.shape[0], device=probs.device)
    if kind == 'constant':
        return torch.full((probs.shape[0],), 0.5, device=probs.device)
    if kind == 'expected':
        # Σ π(a)² per action slot, then average down to [B]
        per_slot = (probs ** 2).sum(-1)  # [B] or [B, T]
        while per_slot.dim() > 1:
            per_slot = per_slot.mean(-1)
        return per_slot
    assert False, f'Unknown baseline: {kind}'


def gather_log_probs(log_probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Gather log probs for taken actions. Always gathers along last dim."""
    return log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def _pg_core(logits, batch, baseline_kind):
    """Shared computation for policy gradient variants.

    Returns logp_a, advantage — both broadcastable to each other.
    logp_a: [B] or [B, T]. advantage: same shape as logp_a.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    logp_a = gather_log_probs(log_probs, batch.actions)

    baseline = compute_baseline(baseline_kind, probs)  # [B]
    advantage = batch.rewards - baseline                # [B]
    while advantage.dim() < logp_a.dim():
        advantage = advantage.unsqueeze(-1)

    return logp_a, advantage


# ── Core losses ──────────────────────────────────────────────────────────────


class CELoss:
    """Cross-entropy (supervised oracle). Uses true labels, ignores RL experience."""
    name = 'CE'

    def __call__(self, logits, batch):
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), batch.labels.reshape(-1))
        return loss, {'reward': batch.rewards.mean().item()}


class REINFORCELoss:
    """REINFORCE — no off-policy correction, uses stale actions as-is."""
    name = 'REINFORCE'

    def __init__(self, baseline: str = 'expected'):
        self.baseline = baseline

    def __call__(self, logits, batch):
        logp_a, advantage = _pg_core(logits, batch, self.baseline)
        loss = -(logp_a * advantage.detach()).mean()
        return loss, {'reward': batch.rewards.mean().item()}


class PGLoss:
    """Importance-weighted PG. Exact for one-step tasks (bandits).
    For sequential tasks, uses per-token ratios — a standard approximation,
    not exact trajectory-level off-policy correction."""
    name = 'PG'

    def __init__(self, baseline: str = 'expected', iw_cap: float = 10.0):
        self.baseline = baseline
        self.iw_cap = iw_cap

    def __call__(self, logits, batch):
        logp_a, advantage = _pg_core(logits, batch, self.baseline)

        log_iw = logp_a - batch.actor_logp_a
        iw = torch.exp(log_iw.clamp(max=math.log(self.iw_cap)))

        loss = -(logp_a * (advantage * iw).detach()).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'iw_mean': iw.mean().item(),
        }


class DGLoss:
    """Delightful policy gradient — gates by sigmoid(delight / eta).

    Osband 2026, arXiv:2603.14608.
    """
    name = 'DG'

    def __init__(self, eta: float = 1.0, baseline: str = 'expected'):
        self.eta = eta
        self.baseline = baseline

    def __call__(self, logits, batch):
        logp_a, advantage = _pg_core(logits, batch, self.baseline)

        surprisal = -logp_a
        delight = advantage * surprisal
        gate = torch.sigmoid(delight / self.eta)

        loss = -(logp_a * (gate * advantage).detach()).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'gate_mean': gate.mean().item(),
        }


class KondoLoss:
    """Compute-efficient DG — screens samples BEFORE the learner forward pass.

    "Does This Gradient Spark Joy?" (Osband 2026, arXiv:2603.20526).
    The screen() method uses actor_log_probs (already in the batch, no learner
    forward needed) to estimate delight and select the top keep_ratio fraction.
    The training loop calls screen() → batch.select() → compute_logits on the
    reduced batch, so only selected samples go through forward + backward.
    """
    name = 'Kondo'

    def __init__(self, eta: float = 1.0, keep_ratio: float = 0.5,
                 baseline: str = 'expected'):
        self.eta = eta
        self.keep_ratio = keep_ratio
        self.baseline = baseline
        self._kept_frac = 1.0  # tracked for metrics

    def screen(self, batch) -> torch.Tensor:
        """Pre-screen using actor log-probs. Returns boolean mask [B].

        Called BEFORE compute_logits — this is where the compute saving happens.
        For sequential tasks (delight is [B, T]), aggregates to one score per
        sequence via max |delight| over tokens. Mask is always [B].
        """
        actor_logp_a = batch.actor_logp_a
        baseline = batch.actor_baseline
        advantage = batch.rewards - baseline
        while advantage.dim() < actor_logp_a.dim():
            advantage = advantage.unsqueeze(-1)
        delight = advantage * (-actor_logp_a)

        # Reduce to per-sequence score: max |delight| over token positions
        per_sample = delight.abs()
        while per_sample.dim() > 1:
            per_sample = per_sample.max(dim=-1).values

        B = per_sample.shape[0]
        k = max(1, int(B * self.keep_ratio))
        threshold = per_sample.kthvalue(B - k + 1).values
        mask = per_sample >= threshold
        self._kept_frac = mask.float().mean().item()
        return mask

    def __call__(self, logits, batch):
        """Standard DG on the pre-filtered batch. No additional masking needed."""
        logp_a, advantage = _pg_core(logits, batch, self.baseline)
        surprisal = -logp_a
        gate = torch.sigmoid(advantage * surprisal / self.eta)

        loss = -(logp_a * (gate * advantage).detach()).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'gate_mean': gate.mean().item(),
            'kept_frac': self._kept_frac,
        }


class LogGrowthLoss:
    """Kelly-optimal PG via success-conditional inverse-propensity weighting.

    The CE (Kelly) gradient is ∇log π(y*|x) — computable only when y* is known.
    In bandit feedback, y* is revealed only when R=1 (correct action sampled).

    Applying 1/π(a) to successes cancels PG's implicit p-weighting across
    contexts, recovering the CE direction. Failures get weight=1 (standard PG)
    because there's no CE gradient to recover from wrong actions.

    The original LogGrowth applied 1/π(a) to ALL samples, which amplified
    rare wrong actions by up to 800× and diverged. This version is stable
    because 1/π(a) for successes is bounded by 1/π(y*), which shrinks as
    the model improves.
    """
    name = 'LogGrowth'

    def __init__(self, baseline: str = 'expected'):
        self.baseline = baseline

    def __call__(self, logits, batch):
        logp_a, advantage = _pg_core(logits, batch, self.baseline)

        # Kelly correction: 1/π(a) for successes, 1 for failures
        is_success = (batch.rewards > 0.5).float()
        while is_success.dim() < logp_a.dim():
            is_success = is_success.unsqueeze(-1)
        inv_pi = torch.exp(-logp_a.detach())
        weight = is_success * inv_pi + (1 - is_success)

        loss = -(logp_a * (advantage * weight).detach()).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'weight_mean': weight.mean().item(),
        }


# ── Field baselines ──────────────────────────────────────────────────────────


class DAPOLoss:
    """Decoupled clip and dynamic sampling policy optimization.

    ByteDance, NeurIPS 2025, arXiv:2503.14476.
    Key mechanisms:
      - Asymmetric clipping: ε_low=0.2, ε_high=0.28 (more room for exploration)
      - Token-level loss normalization (divide by total tokens, not per-sample avg)
      - Per-GROUP advantage normalization when batch.group_ids is present
        (K samples per prompt, advantages normalized within each group)
      - Dynamic filtering of all-correct/all-incorrect groups (done in sample_batch)
    """
    name = 'DAPO'

    def __init__(self, clip_low: float = 0.2, clip_high: float = 0.28):
        self.clip_low = clip_low
        self.clip_high = clip_high

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, batch.actions)

        # Per-group advantage normalization (faithful to DAPO)
        advantage = batch.rewards.clone()
        if batch.group_ids is not None:
            for gid in batch.group_ids.unique():
                mask = batch.group_ids == gid
                grp = advantage[mask]
                advantage[mask] = (grp - grp.mean()) / (grp.std() + 1e-8)
        else:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        while advantage.dim() < logp_a.dim():
            advantage = advantage.unsqueeze(-1)

        # Asymmetric clipped surrogate
        ratio = torch.exp(logp_a - batch.actor_logp_a.detach())
        clipped = torch.clamp(ratio, 1 - self.clip_low, 1 + self.clip_high)
        surrogate = torch.min(ratio * advantage.detach(),
                              clipped * advantage.detach())
        loss = -surrogate.sum() / max(surrogate.numel(), 1)

        return loss, {'reward': batch.rewards.mean().item()}


class PMDMeanLoss:
    """Policy Mirror Descent with mean-reward partition approximation.

    From the Kimi k1.5 lineage (Moonshot AI 2025, arXiv:2501.12599).
    Proven equivalent to mirror descent with adaptive KL + χ² regularization
    (arXiv:2602.05933).

    Regresses log-ratio toward mean-centered advantage:
        L = E[(τ · log(π/π_old) - (r - mean(r)))²]

    Unlike score-function PG, this is an L2 regression — no blunders from
    the score function estimator, but also no delight-style influence shaping.
    """
    name = 'PMDMean'

    def __init__(self, tau: float = 1.0):
        self.tau = tau

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, batch.actions)

        # Mean-centered advantage (the partition function approximation)
        advantage = batch.rewards - batch.rewards.mean()
        while advantage.dim() < logp_a.dim():
            advantage = advantage.unsqueeze(-1)

        # L2 regression: fit τ·log_ratio to advantage
        log_ratio = logp_a - batch.actor_logp_a
        loss = ((self.tau * log_ratio - advantage.detach()) ** 2).mean()

        return loss, {
            'reward': batch.rewards.mean().item(),
            'log_ratio_mean': log_ratio.mean().item(),
        }
