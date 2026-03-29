"""Loss functions for policy gradient experiments.

Each loss is a plain class with __call__(logits, batch) -> (loss, metrics_dict).

From the literature:
  CE         — supervised cross-entropy oracle (upper bound)
  REINFORCE  — Williams 1992, no off-policy correction
  PG         — importance-weighted policy gradient
  DG         — Delightful Policy Gradient (Osband 2026)
  Kondo      — compute-efficient DG ("Does This Gradient Spark Joy?", Osband 2026)
  DAPO       — Decoupled clip + dynamic sampling (ByteDance, NeurIPS 2025)
  MaxRL      — ML-optimal per-group mean normalization, binary only (Tajwar et al. 2026)
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

    # Baseline for advantage centering. When exact E[R|x] is available
    # (binary-reward tasks with known labels), use it. This upgrades all
    # _pg_core methods: better control variate for REINFORCE/PG, correct
    # gate calibration for DG. On tasks where E[R|x] is not cheaply exact
    # (fractional sequence reward), falls back to sum(pi^2).
    if batch.actor_expected_reward is not None:
        baseline = batch.actor_expected_reward                # [B]
    else:
        baseline = compute_baseline(baseline_kind, probs)  # [B]
    advantage = batch.rewards - baseline                # [B]
    while advantage.dim() < logp_a.dim():
        advantage = advantage.unsqueeze(-1)

    return logp_a, advantage


# ── Core losses ──────────────────────────────────────────────────────────────


class CELoss:
    """Cross-entropy supervised reference. Uses true labels, ignores RL experience.

    True oracle on MNIST (exact supervised objective). On sequence tasks,
    this is a dense upper bound: it trains all positions with per-token
    supervision, which is strictly stronger than the RL reward signal.
    """
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
        baseline = batch.actor_expected_reward if batch.actor_expected_reward is not None else batch.actor_baseline
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
    """Kelly-optimal PG via inverse-propensity weighting on exact-match success.

    Diagnostic loss, valid only for binary exact-match one-step bandits
    (MNIST, LM next-token with kl_weight=0). The derivation requires:
    R in {0,1}, success reveals the correct label, advantage is unshaped.

    Outside this regime, use DG instead, which achieves the same directional
    correction via a bounded gate without these restrictions.
    """
    name = 'LogGrowth'

    def __init__(self, baseline: str = 'expected'):
        self.baseline = baseline

    def __call__(self, logits, batch):
        assert batch.actions.dim() == 1, \
            'LogGrowth is only valid for one-step bandits (actions [B], not sequences)'
        assert ((batch.rewards == 0) | (batch.rewards == 1)).all(), \
            'LogGrowth requires binary rewards (R in {0,1}); shaped rewards (e.g. kl_weight>0) are unsupported'
        logp_a, advantage = _pg_core(logits, batch, self.baseline)

        is_success = (batch.actions == batch.labels).float()
        inv_pi = torch.exp(-logp_a.detach())
        weight = is_success * inv_pi + (1 - is_success)

        loss = -(logp_a * (advantage * weight).detach()).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'weight_mean': weight.mean().item(),
        }


class DGTokenCreditLoss:
    """DG with per-token return-to-go credit assignment.

    Instead of broadcasting one sequence reward to all tokens, each token
    gets credit based on how many remaining tokens the actor got correct.
    Token t's return-to-go = mean(correct[t:H]).

    When batch.score_mask is present, the credit signal uses masked
    reward semantics:
      - Only scored positions count as correct in the rtg numerator
      - The rtg denominator counts remaining scored positions, not all
      - Baseline is zeroed at unscored positions (no advantage there)
    Note: in autoregressive tasks, unscored prefix tokens still causally
    condition scored suffix tokens, so they may still deserve some
    gradient indirectly. This is a partial-reward credit benchmark,
    not an oracle where only scored positions matter.

    This tests whether token-level delight outperforms sequence-level delight.
    Only meaningful for sequential tasks with fractional reward where
    per-token correctness decomposes the reward. Not faithful on
    binary_reward tasks (all-or-nothing reward does not decompose
    into per-token contributions).
    """
    name = 'DGToken'

    def __init__(self, eta: float = 1.0):
        self.eta = eta

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, batch.actions)

        # Per-token return-to-go: fraction correct from position t onward.
        # Only meaningful for sequential tasks with [B, T] actions and labels.
        assert batch.actions.dim() == 2, \
            'DGToken requires sequential tasks with [B, T] actions'
        correct = (batch.actions == batch.labels).float()  # [B, T]

        if batch.score_mask is not None:
            mask_f = batch.score_mask.float()  # [B, T]
            # Numerator: only scored positions count as correct
            correct = correct * mask_f
            # Denominator: remaining scored positions from t onward
            counts = mask_f.flip(1).cumsum(1).flip(1).clamp(min=1)  # [B, T]
        else:
            H = correct.shape[1]
            counts = torch.arange(H, 0, -1, device=correct.device).float()

        rtg = correct.flip(1).cumsum(1).flip(1) / counts  # [B, T]

        baseline = (probs ** 2).sum(-1)  # [B, T]
        # Zero baseline at unscored positions so they get zero advantage
        if batch.score_mask is not None:
            baseline = baseline * mask_f
        advantage = rtg - baseline

        surprisal = -logp_a
        delight = advantage * surprisal
        gate = torch.sigmoid(delight / self.eta)

        loss = -(logp_a * (gate * advantage).detach()).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'rtg_mean': rtg.mean().item(),
            'gate_mean': gate.mean().item(),
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


class MaxRLLoss:
    """Maximum likelihood RL via per-group mean-reward normalization.

    Diagnostic comparator for binary-reward grouped settings.
    Tajwar et al. 2026, arXiv:2602.02710. Normalizing advantage by
    mean_reward (= K/N) instead of std makes the gradient an unbiased
    estimate of the ML gradient. Weight function w(p) = 1/p gives hard
    problems more gradient budget.

    Valid regime: binary rewards with grouped rollouts (group_size > 1).
    For continuous rewards, the ML connection breaks and 1/mean is not
    a principled weighting. Use DG instead outside the binary regime.
    """
    name = 'MaxRL'

    def __init__(self, iw_cap: float = 10.0):
        self.iw_cap = iw_cap

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, batch.actions)

        # Per-group mean normalization: the core MaxRL mechanism.
        # For binary rewards, 1/mean = N/K, weighting each success by 1/K
        # instead of 1/N -- an unbiased ML gradient estimator.
        advantage = batch.rewards.clone()
        assert batch.group_ids is not None, \
            'MaxRL requires grouped rollouts (group_size > 1)'
        assert ((batch.rewards == 0) | (batch.rewards == 1)).all(), \
            'MaxRL requires binary rewards (R in {0,1})'
        for gid in batch.group_ids.unique():
            mask = batch.group_ids == gid
            grp = advantage[mask]
            mean_r = grp.mean()
            advantage[mask] = (grp - mean_r) / (mean_r + 1e-8)

        while advantage.dim() < logp_a.dim():
            advantage = advantage.unsqueeze(-1)

        log_iw = logp_a - batch.actor_logp_a
        iw = torch.exp(log_iw.clamp(max=math.log(self.iw_cap)))

        loss = -(logp_a * (advantage * iw).detach()).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'iw_mean': iw.mean().item(),
        }


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
