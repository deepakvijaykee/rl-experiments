"""Loss functions for policy gradient experiments.

Each loss is a plain class with __call__(logits, batch) -> (loss, metrics_dict).

From the literature:
  CE         - supervised cross-entropy oracle (upper bound)
  REINFORCE  - Williams 1992, no off-policy correction
  PG         - per-token importance-weighted policy gradient (approximate on sequences)
  ASPO       - asymmetric IS: flips ratio for positive advantage (arXiv:2510.06062)
  TrajPG     - trajectory-level IW PG (exact off-policy on short sequences)
  DG         - Delightful Policy Gradient (Osband 2026)
  Kondo      - compute-efficient DG ("Does This Gradient Spark Joy?", Osband 2026)
  DGToken    - per-token return-to-go credit assignment (fractional reward only)
  TPO        - target policy optimization over grouped sampled candidates
  TPOFullAction - one-step full-action TPO for MNIST contextual bandit
  TPOToken   - per-prefix token-candidate TPO
  GRPOToken  - per-prefix token-candidate GRPO
  TEMPO      - prefix-tree credit via branch-gated TD (arXiv:2509.18314)
  MaxRL      - ML-optimal per-group mean normalization, binary only (Tajwar et al. 2026)
  R2VPO      - ratio-variance regularized PG, replaces clipping (arXiv:2601.03320)
  PMDMean    - Policy Mirror Descent with mean-reward partition approx (Kimi k1.5 lineage, 2025)

logits shape matches batch.actions:
  - bandit:     logits [B, A],    actions [B]
  - sequential: logits [B, T, V], actions [B, T]
"""

import math

import torch
import torch.nn.functional as F


# -- Helpers -----------------------------------------------------------------


def compute_baseline(kind: str, probs: torch.Tensor) -> torch.Tensor:
    """Baseline from current policy probs (no label access).

    Returns [B] for bandits, [B, T] for sequences. Per-token baselines
    are kept separate to avoid future-action dependence: b_t depends on
    state s_t (prefix up to t), not on a_t or later actions.
    """
    shape = probs.shape[:-1]
    if kind == 'zero':
        return torch.zeros(shape, device=probs.device)
    if kind == 'constant':
        return torch.full(shape, 0.5, device=probs.device)
    if kind == 'expected':
        return (probs ** 2).sum(-1)  # [B] for bandits, [B, T] for sequences
    raise ValueError(f'Unknown baseline: {kind}')


def gather_log_probs(log_probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Gather log probs for taken actions. Always gathers along last dim."""
    return log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def _pg_core(logits, batch, baseline_kind):
    """Shared computation for policy gradient variants.

    Returns logp_a, advantage: both broadcastable to each other.
    logp_a: [B] or [B, T]. advantage: same shape as logp_a.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    logp_a = gather_log_probs(log_probs, batch.actions)

    # Baseline for advantage centering. When exact E[R|x] is available
    # (binary-reward tasks with known labels), use it. This upgrades all
    # _pg_core methods: better control variate for REINFORCE/PG, correct
    # gate calibration for DG. On tasks where E[R|x] is not cheaply exact
    # (fractional sequence reward), falls back to per-token sum(pi^2).
    if batch.actor_expected_reward is not None:
        baseline = batch.actor_expected_reward             # [B]
    else:
        baseline = compute_baseline(baseline_kind, probs)  # [B] or [B, T]
    reward = batch.rewards                                 # [B]
    while reward.dim() < baseline.dim():
        reward = reward.unsqueeze(-1)                      # [B, 1] for [B, T] baseline
    advantage = reward - baseline
    while advantage.dim() < logp_a.dim():
        advantage = advantage.unsqueeze(-1)

    return logp_a, advantage


def _group_advantage(batch, like: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Group-relative reward centering for GRPO-style objectives."""
    assert batch.group_ids is not None, 'group-relative loss requires group_ids'
    advantage = torch.zeros_like(batch.rewards)
    for gid in batch.group_ids.unique():
        mask = batch.group_ids == gid
        group_rewards = batch.rewards[mask]
        centered = group_rewards - group_rewards.mean()
        if not normalize:
            advantage[mask] = centered
            continue
        std = group_rewards.std(unbiased=False)
        if std > 1e-8:
            advantage[mask] = centered / (std + 1e-8)
    while advantage.dim() < like.dim():
        advantage = advantage.unsqueeze(-1)
    return advantage


def _reverse_kl_from_log_ratio(log_ratio: torch.Tensor) -> torch.Tensor:
    """Reverse-KL approximation used by the TPO reference GRPO path."""
    return torch.exp(-log_ratio) + log_ratio - 1


def _ppo_surrogate(logp_a: torch.Tensor, actor_logp_a: torch.Tensor,
                   advantage: torch.Tensor, eps_low: float, eps_high: float,
                   response_sum: bool = False, beta: float = 0.0):
    """PPO/GRPO clipped surrogate with gradients through the ratio."""
    log_ratio = (logp_a - actor_logp_a).clamp(min=-20, max=20)
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, min=1 - eps_low, max=1 + eps_high)
    surrogate = torch.minimum(ratio * advantage.detach(),
                              clipped * advantage.detach())
    objective = surrogate - beta * _reverse_kl_from_log_ratio(log_ratio)
    if response_sum and surrogate.dim() > 1:
        loss = -objective.sum(dim=-1).mean()
    else:
        loss = -objective.mean()
    return loss, ratio


def _reward_uncertainty(batch, like: torch.Tensor) -> torch.Tensor:
    """Reward disagreement proxy, per sample and broadcast to token shape."""
    if batch.group_ids is None:
        uncertainty = torch.ones_like(batch.rewards) * batch.rewards.std(unbiased=False)
    else:
        uncertainty = torch.zeros_like(batch.rewards)
        for gid in batch.group_ids.unique():
            mask = batch.group_ids == gid
            uncertainty[mask] = batch.rewards[mask].std(unbiased=False)
    while uncertainty.dim() < like.dim():
        uncertainty = uncertainty.unsqueeze(-1)
    return uncertainty


def _candidate_log_scores(logp_a: torch.Tensor) -> torch.Tensor:
    """Candidate score for grouped objectives: sum token log-probs for sequences."""
    if logp_a.dim() > 1:
        return logp_a.sum(dim=-1)
    return logp_a


def _tpo_skill(scores: torch.Tensor) -> torch.Tensor:
    """TPO within-group standardized scores, matching jeankaddour/tpo."""
    centered = scores - scores.mean(dim=-1, keepdim=True)
    std = centered.std(dim=-1, unbiased=False, keepdim=True)
    return torch.where(std > 1e-6, centered / std.clamp(min=1e-6), centered)


def _grpo_advantage(scores: torch.Tensor) -> torch.Tensor:
    """GRPO z-scored rewards over the last dimension."""
    centered = scores - scores.mean(dim=-1, keepdim=True)
    std = scores.std(dim=-1, unbiased=False, keepdim=True)
    return centered / (std + 1e-8)


def _grouped_candidate_rows(batch, current_log_scores: torch.Tensor,
                            old_log_scores: torch.Tensor):
    """Yield current scores, old scores, and rewards for each sampled group."""
    assert batch.group_ids is not None, 'candidate-group loss requires group_ids'
    for gid in batch.group_ids.unique():
        mask = batch.group_ids == gid
        yield current_log_scores[mask], old_log_scores[mask], batch.rewards[mask]


# -- Core losses --------------------------------------------------------------


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
    """REINFORCE - no off-policy correction, uses stale actions as-is."""
    name = 'REINFORCE'

    def __init__(self, baseline: str = 'expected'):
        self.baseline = baseline

    def __call__(self, logits, batch):
        logp_a, advantage = _pg_core(logits, batch, self.baseline)
        loss = -(logp_a * advantage.detach()).mean()
        return loss, {'reward': batch.rewards.mean().item()}


class PGLoss:
    """Importance-weighted PG. Exact for one-step tasks (bandits).
    For sequential tasks, uses per-token ratios - a standard approximation,
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


class ASPOLoss:
    """Asymmetric importance-weighted PG - flips IS ratio for positive advantage.

    ASPO (arXiv:2510.06062). Standard PG weights every token by pi_new/pi_old,
    which suppresses rare good actions (low ratio) and amplifies common ones.
    ASPO inverts the ratio for positive-advantage tokens: pi_old/pi_new gives
    rare breakthroughs MORE gradient, not less. Negative-advantage tokens keep
    the standard ratio so common bad actions are still pushed down hard.

    One-sided PPO-clip with asymmetric bounds (eps_low, eps_high):
      A > 0: cap flipped ratio above at 1+eps_high (don't over-promote)
      A < 0: cap standard ratio below at 1-eps_low  (don't over-suppress)
    No cap in the other direction: if the policy moved away from a good
    action, the small ratio flows through uncapped.

    At delay=0 (on-policy), ratio=1 everywhere, so ASPO reduces to REINFORCE.
    """
    name = 'ASPO'

    def __init__(self, baseline: str = 'expected',
                 eps_low: float = 0.2, eps_high: float = 0.28):
        self.baseline = baseline
        self.eps_low = eps_low
        self.eps_high = eps_high

    def __call__(self, logits, batch):
        logp_a, advantage = _pg_core(logits, batch, self.baseline)

        log_ratio = logp_a - batch.actor_logp_a           # log(pi_new / pi_old)
        pos = advantage > 0

        # Ratio flip: inverted for positive advantage, standard for negative
        aspo_log_ratio = torch.where(pos, -log_ratio, log_ratio)
        aspo_ratio = torch.exp(aspo_log_ratio.clamp(min=-20, max=20))

        # One-sided PPO-clip with asymmetric bounds.
        # For A > 0: only cap above (prevent over-promotion beyond trust region)
        # For A < 0: only cap below (prevent over-suppression beyond trust region)
        weight = torch.where(
            pos,
            torch.clamp(aspo_ratio, max=1 + self.eps_high),
            torch.clamp(aspo_ratio, min=1 - self.eps_low),
        )

        loss = -(logp_a * (weight * advantage).detach()).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'ratio_mean': aspo_ratio.mean().item(),
            'weight_mean': weight.mean().item(),
            'pos_frac': pos.float().mean().item(),
        }


class TrajectoryPGLoss:
    """Trajectory-level importance-weighted PG for sequence tasks.

    Uses the full trajectory ratio rho = exp(sum_t log(pi_t/mu_t))
    instead of per-token ratios. Exact off-policy correction for
    logged trajectories (before capping). Practical on short sequences
    (seq_len <= 10) where the variance of the product is manageable.
    On one-step bandits, collapses to PGLoss.
    """
    name = 'TrajPG'

    def __init__(self, baseline: str = 'expected', iw_cap: float = 10.0):
        self.baseline = baseline
        self.iw_cap = iw_cap

    def __call__(self, logits, batch):
        logp_a, advantage = _pg_core(logits, batch, self.baseline)

        if logp_a.dim() > 1:
            # Sequence: trajectory-level ratio (product of per-token ratios)
            log_traj_iw = (logp_a - batch.actor_logp_a).sum(dim=-1)  # [B]
            traj_iw = torch.exp(log_traj_iw.clamp(max=math.log(self.iw_cap)))
            iw = traj_iw.unsqueeze(-1).expand_as(logp_a)  # [B, T]
        else:
            # Bandit: same as PGLoss
            log_iw = logp_a - batch.actor_logp_a
            iw = torch.exp(log_iw.clamp(max=math.log(self.iw_cap)))

        loss = -(logp_a * (advantage * iw).detach()).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'iw_mean': iw.mean().item(),
        }


class DGLoss:
    """Delightful policy gradient - gates by sigmoid(delight / eta).

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


class ReplayDGLoss(DGLoss):
    """DG objective used with an explicit replay buffer.

    Scoped implementation: the loss is intentionally identical to DG; replay
    semantics are owned by the training buffer, not hidden inside the objective.
    """
    name = 'ReplayDG'


class FreshDGLoss:
    """DG with an explicit exponential age penalty for replayed/stale batches."""
    name = 'FreshDG'

    def __init__(self, eta: float = 1.0, baseline: str = 'expected',
                 age_decay: float = 0.02):
        self.eta = eta
        self.baseline = baseline
        self.age_decay = age_decay

    def __call__(self, logits, batch):
        logp_a, advantage = _pg_core(logits, batch, self.baseline)

        surprisal = -logp_a
        gate = torch.sigmoid(advantage * surprisal / self.eta)
        freshness = math.exp(-self.age_decay * batch.age)

        loss = -(logp_a * (freshness * gate * advantage).detach()).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'gate_mean': gate.mean().item(),
            'freshness_weight': freshness,
            'batch_age': float(batch.age),
        }


class DGEntropyGuardLoss:
    """DG with a local entropy-collapse proxy.

    Scoped implementation: entropy collapse is estimated from the sampled
    action probability under the current learner. Positive-advantage updates
    to already-high-probability actions are downweighted; rare successes are
    mostly unchanged.
    """
    name = 'DGEntropyGuard'

    def __init__(self, eta: float = 1.0, baseline: str = 'expected',
                 guard_strength: float = 0.5):
        self.eta = eta
        self.baseline = baseline
        self.guard_strength = guard_strength

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        logp_a = gather_log_probs(log_probs, batch.actions)
        _, advantage = _pg_core(logits, batch, self.baseline)

        prob_a = logp_a.exp()
        mean_prob = probs.mean(dim=-1)
        collapse_proxy = (prob_a - mean_prob).clamp(min=0.0, max=1.0)
        guard = torch.where(
            advantage > 0,
            1.0 - self.guard_strength * collapse_proxy,
            torch.ones_like(collapse_proxy))

        gate = torch.sigmoid(advantage * (-logp_a) / self.eta)
        loss = -(logp_a * (guard * gate * advantage).detach()).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'gate_mean': gate.mean().item(),
            'entropy_guard_mean': guard.mean().item(),
        }


class UncertaintyDGLoss:
    """DG with reward-disagreement conservatism.

    Grouped batches use within-group reward std as the uncertainty signal.
    Ungrouped batches use the current batch reward std, so the method remains a
    stress-test baseline under noisy rewards rather than a full verifier model.
    """
    name = 'UncertaintyDG'

    def __init__(self, eta: float = 1.0, baseline: str = 'expected',
                 uncertainty_scale: float = 1.0):
        self.eta = eta
        self.baseline = baseline
        self.uncertainty_scale = uncertainty_scale

    def __call__(self, logits, batch):
        logp_a, advantage = _pg_core(logits, batch, self.baseline)
        uncertainty = _reward_uncertainty(batch, logp_a)
        surprisal = -logp_a
        gate = torch.sigmoid(
            (advantage * surprisal - self.uncertainty_scale * uncertainty)
            / self.eta)

        loss = -(logp_a * (gate * advantage).detach()).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'gate_mean': gate.mean().item(),
            'uncertainty_mean': uncertainty.mean().item(),
        }


class FilteredDGLoss:
    """DG that drops high-disagreement samples before applying the gate."""
    name = 'FilteredDG'

    def __init__(self, eta: float = 1.0, baseline: str = 'expected',
                 uncertainty_threshold: float = 0.5):
        self.eta = eta
        self.baseline = baseline
        self.uncertainty_threshold = uncertainty_threshold

    def __call__(self, logits, batch):
        logp_a, advantage = _pg_core(logits, batch, self.baseline)
        uncertainty = _reward_uncertainty(batch, logp_a)
        keep = (uncertainty <= self.uncertainty_threshold).float()
        gate = torch.sigmoid(advantage * (-logp_a) / self.eta)

        loss = -(logp_a * (keep * gate * advantage).detach()).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'gate_mean': gate.mean().item(),
            'kept_frac': keep.mean().item(),
            'uncertainty_mean': uncertainty.mean().item(),
        }


class RewardVarianceDGLoss:
    """DG with reward-variance shrinkage of the effective advantage."""
    name = 'RewardVarianceDG'

    def __init__(self, eta: float = 1.0, baseline: str = 'expected',
                 variance_scale: float = 1.0):
        self.eta = eta
        self.baseline = baseline
        self.variance_scale = variance_scale

    def __call__(self, logits, batch):
        logp_a, advantage = _pg_core(logits, batch, self.baseline)
        uncertainty = _reward_uncertainty(batch, logp_a)
        effective_advantage = advantage / (1.0 + self.variance_scale * uncertainty)
        gate = torch.sigmoid(effective_advantage * (-logp_a) / self.eta)

        loss = -(logp_a * (gate * effective_advantage).detach()).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'gate_mean': gate.mean().item(),
            'uncertainty_mean': uncertainty.mean().item(),
        }


class KondoLoss:
    """Compute-efficient DG - screens samples BEFORE the learner forward pass.

    "Does This Gradient Spark Joy?" (Osband 2026, arXiv:2603.20526).
    Algorithm 1: compute delight from actor log-probs, set lambda to the
    batch quantile targeting keep_ratio, then gate each sample stochastically
    via Bernoulli(sigmoid((delight - lambda) / eta)). Only gated samples
    go through the learner forward + backward pass.
    """
    name = 'Kondo'

    def __init__(self, eta: float = 1.0, keep_ratio: float = 0.5,
                 baseline: str = 'expected'):
        self.eta = eta
        self.keep_ratio = keep_ratio
        self.baseline = baseline
        self._kept_frac = 1.0
        self._gate_prob_mean = 0.5

    def screen(self, batch) -> torch.Tensor:
        """Stochastic pre-screen using actor log-probs. Returns boolean mask [B].

        Called BEFORE compute_logits: this is where the compute saving happens.
        For sequential tasks (delight is [B, T]), aggregates to one score per
        sequence via max delight over tokens. Mask is always [B].
        """
        actor_logp_a = batch.actor_logp_a
        baseline = (batch.actor_expected_reward
                    if batch.actor_expected_reward is not None
                    else batch.actor_baseline)
        reward = batch.rewards
        while reward.dim() < baseline.dim():
            reward = reward.unsqueeze(-1)
        advantage = reward - baseline
        while advantage.dim() < actor_logp_a.dim():
            advantage = advantage.unsqueeze(-1)
        delight = advantage * (-actor_logp_a)

        # Reduce to per-sequence score: max delight over token positions.
        per_sample = delight
        while per_sample.dim() > 1:
            per_sample = per_sample.max(dim=-1).values

        if self.keep_ratio == 1.0:
            mask = torch.ones_like(per_sample, dtype=torch.bool)
            self._kept_frac = 1.0
            self._gate_prob_mean = 1.0
            return mask

        # Find lambda so that mean(sigmoid((chi - lambda) / eta)) = keep_ratio.
        # Binary search: sigmoid is monotone in -lambda, so the mean is too.
        lo = per_sample.min() - 10 * self.eta
        hi = per_sample.max() + 10 * self.eta
        for _ in range(20):
            mid = (lo + hi) / 2
            if torch.sigmoid((per_sample - mid) / self.eta).mean() > self.keep_ratio:
                lo = mid
            else:
                hi = mid
        threshold = (lo + hi) / 2

        # Stochastic Bernoulli gate via sigmoid (paper Eq. for w*)
        gate_prob = torch.sigmoid((per_sample - threshold) / self.eta)
        mask = torch.bernoulli(gate_prob).bool()
        if not mask.any():
            mask[per_sample.argmax()] = True

        self._kept_frac = mask.float().mean().item()
        self._gate_prob_mean = gate_prob.mean().item()
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
            'gate_prob_mean': self._gate_prob_mean,
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
            'LogGrowth requires binary rewards; shaped rewards are unsupported'
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


class SelfDistillDGLoss:
    """DG plus oracle-reviser dense token supervision.

    Scoped toy implementation of the self-distillation axis: the task's known
    labels act as an oracle reviser, and DG's token gate decides where sparse
    reward should become dense cross-entropy.
    """
    name = 'SelfDistillDG'

    def __init__(self, eta: float = 1.0, alpha: float = 0.5):
        self.eta = eta
        self.alpha = alpha

    def __call__(self, logits, batch):
        assert batch.actions.dim() == 2, \
            'SelfDistillDG requires sequential tasks with [B, T] actions'
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        logp_a = gather_log_probs(log_probs, batch.actions)
        correct = (batch.actions == batch.labels).float()

        baseline = (probs ** 2).sum(-1)
        if batch.score_mask is not None:
            mask_f = batch.score_mask.float()
            correct = correct * mask_f
            baseline = baseline * mask_f

        advantage = correct - baseline
        gate = torch.sigmoid(advantage * (-logp_a) / self.eta)
        rl_loss = -(logp_a * (gate * advantage).detach()).mean()

        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch.labels.reshape(-1),
            reduction='none').reshape_as(logp_a)
        distill_loss = (gate.detach() * ce).mean()
        loss = rl_loss + self.alpha * distill_loss
        return loss, {
            'reward': batch.rewards.mean().item(),
            'gate_mean': gate.mean().item(),
            'distill_loss': distill_loss.item(),
        }


class SCOPELiteLoss:
    """Partial-failure recycling with first-error suffix correction.

    Uses the toy task labels as a process verifier: failed trajectories receive
    dense CE on the suffix starting at the first wrong token, while the base DG
    loss still trains from the sampled reward.
    """
    name = 'SCOPELite'

    def __init__(self, eta: float = 1.0, baseline: str = 'expected',
                 alpha: float = 0.5):
        self.eta = eta
        self.baseline = baseline
        self.alpha = alpha

    def __call__(self, logits, batch):
        assert batch.actions.dim() == 2, \
            'SCOPELite requires sequential tasks with [B, T] actions'
        log_probs = F.log_softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, batch.actions)
        _, advantage = _pg_core(logits, batch, self.baseline)
        gate = torch.sigmoid(advantage * (-logp_a) / self.eta)
        rl_loss = -(logp_a * (gate * advantage).detach()).mean()

        wrong = batch.actions != batch.labels
        has_error = wrong.any(dim=1)
        first_error = wrong.float().argmax(dim=1)
        positions = torch.arange(batch.actions.size(1), device=batch.actions.device)
        suffix_mask = positions.unsqueeze(0) >= first_error.unsqueeze(1)
        suffix_mask = suffix_mask & has_error.unsqueeze(1)
        if batch.score_mask is not None:
            suffix_mask = suffix_mask & batch.score_mask

        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch.labels.reshape(-1),
            reduction='none').reshape_as(logp_a)
        denom = suffix_mask.float().sum().clamp(min=1.0)
        correction_loss = (ce * suffix_mask.float()).sum() / denom
        loss = rl_loss + self.alpha * correction_loss
        return loss, {
            'reward': batch.rewards.mean().item(),
            'gate_mean': gate.mean().item(),
            'first_error_frac': has_error.float().mean().item(),
            'correction_loss': correction_loss.item(),
        }


class GRPOLoss:
    """Canonical group-relative PPO baseline for RLVR-style rollouts.

    Scoped faithful path: critic-free group-normalized advantages, PPO clipped
    ratio surrogate, and the reverse-KL term used by the TPO reference GRPO
    implementation. It does not implement large-system rollout infrastructure
    beyond this repo's grouped sampler.
    """
    name = 'GRPO'

    def __init__(self, eps: float = 0.2, beta: float = 0.04):
        self.eps = eps
        self.beta = beta

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, batch.actions)
        advantage = _group_advantage(batch, logp_a)
        loss, ratio = _ppo_surrogate(
            logp_a, batch.actor_logp_a, advantage, self.eps, self.eps,
            beta=self.beta)
        return loss, {
            'reward': batch.rewards.mean().item(),
            'ratio_mean': ratio.mean().item(),
            'adv_abs_mean': advantage.abs().mean().item(),
            'reverse_kl_mean': _reverse_kl_from_log_ratio(
                (logp_a - batch.actor_logp_a).clamp(min=-20, max=20)).mean().item(),
        }


class DrGRPOLoss:
    """Dr.GRPO-lite: group-centered PPO without reward-std normalization.

    Dr.GRPO removes two GRPO weighting biases: response-length normalization
    and reward-std normalization. The toy sequence tasks have fixed response
    length, so the observable distinction here is unnormalized centered reward
    plus response-summed token loss. The optional reverse-KL term stays aligned
    with the rollout GRPO baseline.
    """
    name = 'DrGRPO'

    def __init__(self, eps: float = 0.2, beta: float = 0.04):
        self.eps = eps
        self.beta = beta

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, batch.actions)
        advantage = _group_advantage(batch, logp_a, normalize=False)
        loss, ratio = _ppo_surrogate(
            logp_a, batch.actor_logp_a, advantage, self.eps, self.eps,
            response_sum=True, beta=self.beta)
        return loss, {
            'reward': batch.rewards.mean().item(),
            'ratio_mean': ratio.mean().item(),
            'adv_abs_mean': advantage.abs().mean().item(),
            'reverse_kl_mean': _reverse_kl_from_log_ratio(
                (logp_a - batch.actor_logp_a).clamp(min=-20, max=20)).mean().item(),
        }


class DAPOLiteLoss:
    """DAPO-lite: decoupled clipping plus token-level grouped PG.

    Scoped implementation of the design choices that fit this repo:
    group-relative advantages, asymmetric clip bounds, dynamic mixed-reward
    group filtering from the sampler, and per-token ratio clipping. It omits
    overlong reward shaping because current toy tasks have fixed response
    lengths.
    """
    name = 'DAPOLite'

    def __init__(self, eps_low: float = 0.2, eps_high: float = 0.28):
        self.eps_low = eps_low
        self.eps_high = eps_high

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, batch.actions)
        advantage = _group_advantage(batch, logp_a)
        loss, ratio = _ppo_surrogate(
            logp_a, batch.actor_logp_a, advantage,
            self.eps_low, self.eps_high)
        return loss, {
            'reward': batch.rewards.mean().item(),
            'ratio_mean': ratio.mean().item(),
            'adv_abs_mean': advantage.abs().mean().item(),
        }


class TPOLoss:
    """Target Policy Optimization over grouped sampled candidates.

    Scoped faithful path from jeankaddour/tpo: for each context group, build
    q_i proportional to softmax(log pi_old(candidate_i)) * exp(skill_i / eta),
    then fit the current policy over the same sampled candidates by cross
    entropy. For sequences, candidate log-prob is the response-summed log-prob.
    """
    name = 'TPO'

    def __init__(self, eta: float = 1.0, anchor_old_policy: bool = True):
        self.eta = eta
        self.anchor_old_policy = anchor_old_policy

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, batch.actions)
        current_scores = _candidate_log_scores(logp_a)
        old_scores = _candidate_log_scores(batch.actor_logp_a).detach()

        losses = []
        target_entropy = []
        target_top_prob = []
        for current_group, old_group, reward_group in _grouped_candidate_rows(
                batch, current_scores, old_scores):
            skill = _tpo_skill(reward_group.unsqueeze(0)).squeeze(0)
            if self.anchor_old_policy:
                target_logits = F.log_softmax(old_group, dim=-1) + skill / self.eta
            else:
                target_logits = skill / self.eta
            q = F.softmax(target_logits, dim=-1).detach()
            log_p = F.log_softmax(current_group, dim=-1)
            losses.append(-(q * log_p).sum())
            target_entropy.append(-(q * q.clamp_min(1e-12).log()).sum())
            target_top_prob.append(q.max())

        loss = torch.stack(losses).mean()
        entropy = torch.stack(target_entropy).mean()
        top_prob = torch.stack(target_top_prob).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'tpo_target_entropy': entropy.item(),
            'tpo_target_top_prob': top_prob.item(),
        }


class TPONoAnchorLoss(TPOLoss):
    """TPO ablation with q proportional to exp(skill / eta), no old-policy anchor."""
    name = 'TPONoAnchor'

    def __init__(self, eta: float = 1.0):
        super().__init__(eta=eta, anchor_old_policy=False)


class GroupPGLoss:
    """Scalar-weighted grouped policy-gradient ablation using TPO skill."""
    name = 'GroupPG'

    def __init__(self, eta: float = 1.0):
        self.eta = eta

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, batch.actions)
        current_scores = _candidate_log_scores(logp_a)

        losses = []
        skill_abs = []
        assert batch.group_ids is not None, 'GroupPG requires grouped rollouts'
        for gid in batch.group_ids.unique():
            mask = batch.group_ids == gid
            current_group = current_scores[mask]
            reward_group = batch.rewards[mask]
            skill = (_tpo_skill(reward_group.unsqueeze(0)).squeeze(0)
                     / self.eta).detach()
            losses.append(-(skill * current_group).sum())
            skill_abs.append(skill.abs().mean())

        loss = torch.stack(losses).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'tpo_skill_abs_mean': torch.stack(skill_abs).mean().item(),
        }


class TPOFullActionLoss:
    """Single-sample full-action TPO for the MNIST contextual bandit.

    Faithful local path from jeankaddour/tpo's classification TPO:
    sample one class, place its advantage in a full action-score vector,
    build the anchored TPO target over all classes, then fit by cross-entropy.
    The trainer validates this for on-policy, one-step, single-epoch runs so
    the current detached logits are the old-policy anchor.
    """
    name = 'TPOFullAction'

    def __init__(self, eta: float = 1.0):
        self.eta = eta

    def __call__(self, logits, batch):
        if batch.actions.dim() != 1:
            raise ValueError('TPOFullAction requires one-step bandit actions')

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        baseline = (probs ** 2).sum(dim=-1)
        advantage = (batch.rewards - baseline).detach()

        scores = torch.zeros_like(log_probs)
        scores.scatter_(1, batch.actions.unsqueeze(1), advantage.unsqueeze(1))
        skill = _tpo_skill(scores)

        target_logits = log_probs.detach() + skill / self.eta
        q = F.softmax(target_logits, dim=-1).detach()
        loss = -(q * log_probs).sum(dim=-1).mean()

        entropy = -(q * q.clamp_min(1e-12).log()).sum(dim=-1).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'tpo_target_entropy': entropy.item(),
            'tpo_sample_adv_mean': advantage.mean().item(),
        }


class TPOTokenLoss:
    """Per-prefix token-candidate TPO.

    Scoped faithful path from jeankaddour/tpo's token-candidate experiment:
    each prefix has K sampled next-token candidates, dense verifier rewards,
    frozen actor log-probs, and a TPO target on that local candidate simplex.
    """
    name = 'TPOToken'

    def __init__(self, eta: float = 1.0):
        self.eta = eta

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        current = log_probs.gather(-1, batch.candidate_actions)
        skill = _tpo_skill(batch.candidate_rewards)
        target_logits = (
            F.log_softmax(batch.old_candidate_logp.detach(), dim=-1)
            + skill / self.eta
        )
        q = F.softmax(target_logits, dim=-1).detach()
        log_p = F.log_softmax(current, dim=-1)
        loss = -(q * log_p).sum(dim=-1).mean()

        entropy = -(q * q.clamp_min(1e-12).log()).sum(dim=-1).mean()
        return loss, {
            'reward': batch.behavior_sequence_rewards.mean().item(),
            'candidate_reward': batch.candidate_rewards.mean().item(),
            'behavior_token_reward': batch.behavior_rewards.mean().item(),
            'tpo_target_entropy': entropy.item(),
        }


class GRPOTokenLoss:
    """Per-prefix token-candidate GRPO with the reverse-KL term from TPO."""
    name = 'GRPOToken'

    def __init__(self, eps: float = 0.2, beta: float = 0.04):
        self.eps = eps
        self.beta = beta

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        current = log_probs.gather(-1, batch.candidate_actions)
        old = batch.old_candidate_logp.detach()
        advantages = _grpo_advantage(batch.candidate_rewards).detach()

        log_ratio = (current - old).clamp(min=-20, max=20)
        ratio = log_ratio.exp()
        clipped = ratio.clamp(min=1 - self.eps, max=1 + self.eps)
        surrogate = torch.minimum(ratio * advantages, clipped * advantages)
        reverse_kl = _reverse_kl_from_log_ratio(log_ratio)
        loss = -((surrogate - self.beta * reverse_kl).sum(dim=-1)).mean()

        return loss, {
            'reward': batch.behavior_sequence_rewards.mean().item(),
            'candidate_reward': batch.candidate_rewards.mean().item(),
            'behavior_token_reward': batch.behavior_rewards.mean().item(),
            'ratio_mean': ratio.mean().item(),
            'adv_abs_mean': advantages.abs().mean().item(),
            'reverse_kl_mean': reverse_kl.mean().item(),
        }


class TEMPOLoss:
    """Prefix-tree credit assignment via branch-gated TD.

    TEMPO (arXiv:2509.18314). Requires grouped rollouts (group_size > 1).
    Builds an implicit prefix tree from responses sharing the same context,
    computes nonparametric prefix values V(s_t) = mean reward of descendants,
    and adds a TD correction at branching tokens where rollouts diverge.

    Advantage = GRPO baseline + branch-gated TD:
        A_{i,t} = (r_i - mean(r) + V(s_{t+1}) - V(s_t)) / std(r)

    At non-branching tokens (all rollouts agree), TD = 0 and this
    reduces to GRPO. At branching tokens, TD provides token-level
    credit without a learned critic.

    For bandits (no sequence dimension), reduces to GRPO.
    """
    name = 'TEMPO'

    def __init__(self, iw_cap: float = 10.0):
        self.iw_cap = iw_cap

    @staticmethod
    def _prefix_values(actions, rewards):
        """Nonparametric prefix values from grouped rollouts.

        Builds an implicit prefix tree by progressive token matching.
        V(s_t) for sample i = mean reward of all samples in the group
        that agree with sample i on output tokens 0..t-1.

        actions: [K, H] token sequences for one group
        rewards: [K] outcome rewards
        Returns: [K, H+1] prefix values (s_0 through s_H)
        """
        K, H = actions.shape
        vals = torch.zeros(K, H + 1, device=actions.device)
        vals[:, 0] = rewards.mean()

        # match[i,j] tracks whether samples i and j agree on all tokens so far
        match = torch.ones(K, K, dtype=torch.bool, device=actions.device)
        for t in range(H):
            same_at_t = actions[:, t].unsqueeze(1) == actions[:, t].unsqueeze(0)
            match = match & same_at_t
            match_f = match.float()
            counts = match_f.sum(dim=1).clamp(min=1)
            vals[:, t + 1] = (match_f @ rewards) / counts

        return vals

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, batch.actions)

        assert batch.group_ids is not None, \
            'TEMPO requires grouped rollouts (group_size > 1)'

        rewards = batch.rewards
        actions = batch.actions
        is_seq = actions.dim() > 1

        # Compute per-token TEMPO advantage for each group
        advantage = torch.zeros_like(logp_a)
        td_nonzero = 0
        td_total = 0

        for gid in batch.group_ids.unique():
            mask = batch.group_ids == gid
            g_rewards = rewards[mask]
            K = g_rewards.shape[0]
            if K < 2:
                continue

            g_std = g_rewards.std(unbiased=False)
            if g_std < 1e-8:
                continue  # all same reward; no learning signal
            g_mean = g_rewards.mean()

            if is_seq:
                g_actions = actions[mask]  # [K, H]
                pv = self._prefix_values(g_actions, g_rewards)  # [K, H+1]
                td = pv[:, 1:] - pv[:, :-1]  # [K, H]

                grpo_base = (g_rewards - g_mean).unsqueeze(1) / g_std  # [K, 1]
                advantage[mask] = grpo_base + td / g_std

                td_nonzero += (td.abs() > 1e-8).sum().item()
                td_total += td.numel()
            else:
                # Bandit: no sequence, just GRPO normalization
                advantage[mask] = (g_rewards - g_mean) / g_std

        # IS correction
        log_ratio = logp_a - batch.actor_logp_a
        ratio = torch.exp(log_ratio.clamp(max=math.log(self.iw_cap)))

        loss = -(logp_a * (ratio * advantage).detach()).mean()
        return loss, {
            'reward': rewards.mean().item(),
            'ratio_mean': ratio.mean().item(),
            'td_nonzero_frac': td_nonzero / max(td_total, 1),
        }


# -- Field baselines ----------------------------------------------------------


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


class R2VPOLoss:
    """Ratio-variance regularized PG - replaces hard clipping with soft penalty.

    R2VPO (arXiv:2601.03320). Instead of capping/clipping the IS ratio,
    penalizes its variance: L = ratio * A - lam * (ratio - 1)^2.
    The penalty 2*lam*(ratio-1) acts as a dynamic regularizer that
    smoothly scales the effective advantage based on ratio deviation.

    Unlike hard clipping (which zeros gradient at the boundary), R2VPO
    preserves gradient sign but scales magnitude - rare high-advantage,
    high-divergence samples ("eureka moments") still contribute.

    Comparison to DG/ASPO: those *amplify* rare breakthroughs.
    R2VPO merely *preserves* them by not clipping. Tests whether
    amplification is necessary or preservation is sufficient.

    No ratio capping: the variance penalty self-corrects for large ratios.
    """
    name = 'R2VPO'

    def __init__(self, baseline: str = 'expected', lam: float = 0.04):
        self.baseline = baseline
        self.lam = lam

    def __call__(self, logits, batch):
        logp_a, advantage = _pg_core(logits, batch, self.baseline)

        log_ratio = logp_a - batch.actor_logp_a
        ratio = torch.exp(log_ratio.clamp(min=-20, max=20))  # numerical guard only

        # R2VPO: effective advantage = A - 2*lam*(ratio-1)
        # The penalty pulls ratio toward 1, softly constraining the trust region.
        variance_penalty = 2 * self.lam * (ratio - 1)
        effective = ratio * (advantage - variance_penalty)

        loss = -(logp_a * effective.detach()).mean()
        return loss, {
            'reward': batch.rewards.mean().item(),
            'ratio_mean': ratio.mean().item(),
            'var_penalty_mean': variance_penalty.mean().item(),
        }


class PMDMeanLoss:
    """Policy Mirror Descent with mean-reward partition approximation.

    From the Kimi k1.5 lineage (Moonshot AI 2025, arXiv:2501.12599).
    Proven equivalent to mirror descent with adaptive KL + chi-squared
    regularization (arXiv:2602.05933).

    Regresses trajectory-level log-ratio toward context-centered reward:
        L = E[(tau * log(pi/pi_old) - (r - E[r|x]))^2]

    Uses exact E[R|x] when available (actor_expected_reward), falls back
    to batch mean otherwise. On sequences, the log-ratio is summed across
    tokens (trajectory-level), not regressed per-token.
    """
    name = 'PMDMean'

    def __init__(self, tau: float = 1.0):
        self.tau = tau

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, batch.actions)

        # Context-conditional baseline: exact E[R|x] when available
        if batch.actor_expected_reward is not None:
            advantage = batch.rewards - batch.actor_expected_reward
        else:
            advantage = batch.rewards - batch.rewards.mean()

        # Trajectory-level log-ratio for sequences
        log_ratio = logp_a - batch.actor_logp_a
        if log_ratio.dim() > 1:
            log_ratio = log_ratio.sum(dim=-1)  # [B]

        loss = ((self.tau * log_ratio - advantage.detach()) ** 2).mean()

        return loss, {
            'reward': batch.rewards.mean().item(),
            'log_ratio_mean': log_ratio.mean().item(),
        }
