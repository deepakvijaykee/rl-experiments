"""Training loop, experience queue, gradient diagnostics, and CLI entry point."""

import argparse
import dataclasses
import math
import time
from collections import deque
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F

from . import losses as L
from .tasks import (
    Batch, MNISTBandit, TokenReversal, MaskedReversal,
    RewardChainReversal, ChainArithmetic, FormatAnswerArithmetic, LMBandit,
)


GROUPED_METHODS = {
    'GRPO', 'DrGRPO', 'DAPO', 'DAPOLite', 'TPO', 'TPONoAnchor',
    'GroupPG', 'TEMPO', 'MaxRL',
}

SEQUENTIAL_METHODS = {
    'DGToken', 'SelfDistillDG', 'SCOPELite',
}

TOKEN_CANDIDATE_METHODS = {'TPOToken', 'GRPOToken'}

REPLAY_PRIORITIES = {
    'uniform', 'reward', 'advantage', 'surprisal', 'delight', 'fresh_delight',
}

BASELINES = {'zero', 'constant', 'expected'}

REWARD_NOISE_MODES = {
    'none', 'label_flip', 'random_reward', 'false_positive_action0',
    'false_positive_rare_token', 'false_positive_sep', 'spurious_feature',
}

BANDIT_REWARD_NOISE_MODES = {
    'none', 'label_flip', 'random_reward', 'false_positive_action0',
    'spurious_feature',
}

SEQUENCE_REWARD_NOISE_MODES = {
    'none', 'label_flip', 'random_reward', 'false_positive_action0',
    'false_positive_rare_token', 'false_positive_sep', 'spurious_feature',
}

BANDIT_TASKS = {'mnist', 'lm_bandit'}
REWARD_CHAIN_TASKS = {'chain_reversal', 'chain_arithmetic', 'format_answer'}
TPO_METHODS = {'TPO', 'TPONoAnchor', 'GroupPG', 'TPOFullAction', 'TPOToken'}
ETA_METHODS = {
    'DG', 'ReplayDG', 'FreshDG', 'DGEntropyGuard', 'UncertaintyDG',
    'FilteredDG', 'RewardVarianceDG', 'Kondo', 'DGToken',
    'SelfDistillDG', 'SCOPELite',
}


# -- Experience Queue ---------------------------------------------------------


class ExperienceQueue:
    """Ring buffer of pre-sampled Batches for delayed training.

    Replaces StalenessBuffer (which stored model state_dicts). Scales with
    batch size, not model size: O(delay * batch) vs O(delay * parameter_count).

    Staleness semantics: at step t, the training batch has actor_log_probs
    from D steps ago. The learner's current forward pass provides fresh logits.
    """

    def __init__(self, delay: int):
        self.delay = delay
        self.buffer: deque[Batch] = deque(maxlen=delay + 1)

    def push(self, batch: Batch):
        self.buffer.append(batch.to('cpu'))

    def ready(self) -> bool:
        return len(self.buffer) > self.delay

    def get_stale(self, device) -> Batch:
        """Return the oldest batch in the queue (= D steps ago), on device."""
        return self.buffer[0].with_age(self.delay).to(device)


class ExperienceReplayBuffer:
    """Batch-level replay with explicit freshness-aware priority."""

    def __init__(self, capacity: int, priority: str, age_decay: float):
        if capacity <= 0:
            raise ValueError('replay capacity must be > 0')
        if priority not in REPLAY_PRIORITIES:
            raise ValueError(f'Unknown replay_priority: {priority}')
        self.capacity = capacity
        self.priority = priority
        self.age_decay = age_decay
        self.buffer: list[Batch] = []

    def push(self, batch: Batch):
        self.buffer = [b.with_age(b.age + 1) for b in self.buffer]
        self.buffer.append(batch.with_age(0).to('cpu'))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def ready(self, min_age: int) -> bool:
        return any(b.age >= min_age for b in self.buffer)

    def _priority(self, batch: Batch) -> float:
        if self.priority == 'uniform':
            return 1.0

        actor_logp = batch.actor_logp_a
        reward = batch.rewards
        baseline = (batch.actor_expected_reward
                    if batch.actor_expected_reward is not None
                    else batch.actor_baseline)
        while reward.dim() < baseline.dim():
            reward = reward.unsqueeze(-1)
        advantage = reward - baseline
        while advantage.dim() < actor_logp.dim():
            advantage = advantage.unsqueeze(-1)

        if self.priority == 'reward':
            score = batch.rewards.abs().mean()
        elif self.priority == 'advantage':
            score = advantage.abs().mean()
        elif self.priority == 'surprisal':
            score = (-actor_logp).mean()
        elif self.priority == 'delight':
            score = (advantage * (-actor_logp)).clamp(min=0).mean()
        elif self.priority == 'fresh_delight':
            freshness = math.exp(-self.age_decay * batch.age)
            score = freshness * (advantage * (-actor_logp)).clamp(min=0).mean()
        else:
            raise ValueError(f'Unknown replay_priority: {self.priority}')

        return max(float(score.item()), 0.0)

    def sample(self, device, min_age: int):
        candidates = [b for b in self.buffer if b.age >= min_age]
        assert candidates, 'replay buffer sampled before ready'
        priorities = torch.tensor([self._priority(b) for b in candidates])
        if priorities.sum() <= 0:
            idx = torch.randint(len(candidates), (1,)).item()
        else:
            idx = torch.multinomial(priorities, 1).item()
        batch = candidates[idx].to(device)
        return batch, {
            'replay_age': float(batch.age),
            'replay_size': float(len(self.buffer)),
            'replay_priority': float(priorities[idx].item()),
        }


# -- Gradient Diagnostics -----------------------------------------------------


def compute_gradient_cosines(
        model, task, batch, loss_fn, method_logits_fn, device) -> dict[str, float]:
    """Cosine similarity of method gradient to CE oracle gradient.

    method_logits_fn: the logits function the method uses during training
    (compute_logits for RL methods, compute_logits_oracle for CE).
    """
    def flat_grad(logits_fn, compute_loss):
        model.zero_grad()
        logits = logits_fn(model, batch)
        loss = compute_loss(logits, batch)
        loss.backward()
        return torch.cat([p.grad.flatten() for p in model.parameters()])

    g_method = flat_grad(method_logits_fn, lambda l, b: loss_fn(l, b)[0])
    g_ce = flat_grad(task.compute_logits_oracle, lambda l, b: F.cross_entropy(
        l.reshape(-1, l.size(-1)), b.labels.reshape(-1)))

    cos = F.cosine_similarity
    result = {
        'cos_method_ce': cos(g_method.unsqueeze(0), g_ce.unsqueeze(0)).item(),
        'grad_norm': g_method.norm().item(),
    }
    model.zero_grad()
    return result


def compute_entropy_diagnostics(logits_before, logits_after, batch) -> dict[str, float]:
    """Token/bandit entropy-change diagnostics on the just-updated batch."""
    lp_before = F.log_softmax(logits_before.float(), dim=-1)
    lp_after = F.log_softmax(logits_after.float(), dim=-1)
    p_before = lp_before.exp()
    p_after = lp_after.exp()
    ent_before = -(p_before * lp_before).sum(dim=-1)
    ent_after = -(p_after * lp_after).sum(dim=-1)
    delta_ent = ent_after - ent_before

    logit_a_before = logits_before.gather(
        -1, batch.actions.unsqueeze(-1)).squeeze(-1).float()
    logit_a_after = logits_after.gather(
        -1, batch.actions.unsqueeze(-1)).squeeze(-1).float()
    prob_a = p_before.gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
    delta_logit = logit_a_after - logit_a_before
    cov = ((prob_a - prob_a.mean()) *
           (delta_logit - delta_logit.mean())).mean()

    reward = batch.rewards
    baseline = (batch.actor_expected_reward
                if batch.actor_expected_reward is not None
                else batch.actor_baseline)
    while reward.dim() < baseline.dim():
        reward = reward.unsqueeze(-1)
    advantage = reward - baseline
    while advantage.dim() < delta_ent.dim():
        advantage = advantage.unsqueeze(-1)
    pos = advantage > 0
    neg = advantage < 0

    entropy_drop = -delta_ent
    surprisal = -lp_before.gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
    delight = advantage * surprisal

    def add_bucket_metrics(result, name: str, score: torch.Tensor):
        flat_score = score.float().reshape(-1)
        flat_drop = entropy_drop.float().reshape(-1)
        cuts = torch.quantile(
            flat_score,
            torch.tensor([1.0 / 3.0, 2.0 / 3.0], device=flat_score.device))
        masks = {
            'low': flat_score <= cuts[0],
            'mid': (flat_score > cuts[0]) & (flat_score <= cuts[1]),
            'high': flat_score > cuts[1],
        }
        for bucket, mask in masks.items():
            key = f'entropy_drop_{name}_{bucket}'
            result[key] = flat_drop[mask].mean().item() if mask.any() else float('nan')

    result = {
        'batch_entropy_before': ent_before.mean().item(),
        'batch_entropy_after': ent_after.mean().item(),
        'batch_delta_entropy': delta_ent.mean().item(),
        'cov_prob_delta_logit': cov.item(),
    }
    if pos.any():
        result['entropy_drop_pos_adv'] = (-delta_ent[pos]).mean().item()
    if neg.any():
        result['entropy_drop_neg_adv'] = (-delta_ent[neg]).mean().item()
    add_bucket_metrics(result, 'surprisal', surprisal)
    add_bucket_metrics(result, 'delight', delight)
    return result


# -- Training Loop ------------------------------------------------------------


def _use_autocast(config, device) -> bool:
    return config.task == 'lm_bandit' and device.type == 'cuda'


def _training_logits(task, model, batch, method: str, token_candidate: bool):
    """Logits path used by the learner update and update-local diagnostics."""
    if method == 'CE':
        return task.compute_logits_oracle(model, batch)
    if token_candidate:
        return task.compute_token_candidate_logits(model, batch)
    return task.compute_logits(model, batch)


def train_one_seed(task, loss_fn, model, config, seed, device) -> list[dict]:
    """Run one training seed. Returns list of metric dicts at eval points."""
    torch.manual_seed(seed)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    autocast = _use_autocast(config, device)

    # Compute difficulty terciles from the initial pre-trained model, before training
    task.compute_difficulty(model, device)

    token_candidate = config.method in TOKEN_CANDIDATE_METHODS

    # Determine group_size for grouped methods
    group_size = config.group_size if config.method in GROUPED_METHODS else 1
    assert config.batch_size % group_size == 0, \
        f'batch_size ({config.batch_size}) must be divisible by group_size ({group_size})'

    # Unified loop: first `delay` rollout steps are warmup. Remaining steps
    # train on genuinely stale data. Each rollout batch can receive multiple
    # inner optimizer epochs, which is what makes PPO-style clipping active
    # for GRPO/DAPO even when delay=0.
    queue = ExperienceQueue(config.delay)
    replay = None
    if config.replay_capacity > 0:
        replay = ExperienceReplayBuffer(
            config.replay_capacity, config.replay_priority,
            config.replay_age_decay)

    results = []
    consecutive_fallbacks = 0
    last_completed_step = -1
    for step in range(config.num_steps):
        # Evaluate BEFORE training so step reflects the model's current state.
        # Skip warmup (step < delay). Step in CSV is the absolute update index,
        # so delay-sweep curves are directly comparable at the same x-axis value.
        past_warmup = step >= config.delay
        eval_due = past_warmup and (step - config.delay) % config.eval_every == 0
        if eval_due:
            model.eval()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
                eval_metrics = task.evaluate(model, device)

        replay_metrics = {}
        if token_candidate:
            batch = task.sample_token_candidates(
                model, config.batch_size, device, config.group_size)
        else:
            # Sample fresh batch with current model (acts as actor)
            fresh_batch = task.sample_batch(model, config.batch_size, device,
                                            group_size=group_size)

            if replay is not None:
                replay.push(fresh_batch)
                if replay.ready(config.delay):
                    batch, replay_metrics = replay.sample(device, config.delay)
                else:
                    batch = fresh_batch.with_age(0)
            elif config.delay == 0:
                batch = fresh_batch.with_age(0)
            else:
                queue.push(fresh_batch)
                # Warmup: train on fresh data while filling the queue.
                # After warmup: pop genuinely stale batch from queue.
                if step < config.delay:
                    batch = fresh_batch.with_age(0)
                else:
                    batch = queue.get_stale(device)

        # Stop if grouped method is out of regime (sustained zero-signal)
        if not token_candidate and batch.used_group_fallback:
            consecutive_fallbacks += 1
            if consecutive_fallbacks >= 10:
                print(f'  STOPPING: {consecutive_fallbacks} consecutive batches with no '
                      f'mixed-reward groups. Method is out of regime.')
                break
        else:
            consecutive_fallbacks = 0

        # Kondo screens samples before the forward pass
        if config.method == 'Kondo':
            batch = batch.select(loss_fn.screen(batch))

        # Learner forward pass in eval mode: keeps dropout consistent with
        # actor sampling (eval mode) so importance weights are exact at delay=0.
        # Gradients still flow; eval only disables stochastic layers.
        logits_before = None
        for inner_idx in range(config.inner_epochs):
            model.eval()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
                logits = _training_logits(
                    task, model, batch, config.method, token_candidate)
                loss, metrics = loss_fn(logits, batch)
                if config.entropy_diagnostics and eval_due and inner_idx == 0:
                    logits_before = logits.detach()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        last_completed_step = step

        if eval_due:
            row = {
                'step': step, 'loss': loss.item(), **metrics,
                **replay_metrics, **eval_metrics,
            }
            if logits_before is not None:
                with torch.no_grad(), torch.amp.autocast(
                        'cuda', dtype=torch.bfloat16, enabled=autocast):
                    logits_after = _training_logits(
                        task, model, batch, config.method, token_candidate)
                row.update(compute_entropy_diagnostics(
                    logits_before, logits_after.detach(), batch))
            if not token_candidate and batch.informative_group_rate is not None:
                row['mixed_group_rate'] = batch.informative_group_rate
                row['retained_group_rate'] = batch.retained_group_rate
                row['group_fallback'] = float(batch.used_group_fallback)

            if config.diagnostics and (step - config.delay) % (config.eval_every * 5) == 0:
                logits_fn = (
                    task.compute_logits_oracle
                    if config.method == 'CE' else task.compute_logits)
                row.update(compute_gradient_cosines(
                    model, task, batch, loss_fn, logits_fn, device))

            results.append(row)
            if config.verbose and (step - config.delay) % (config.eval_every * 10) == 0:
                print(f'  step {step:5d}  test_error={eval_metrics["test_error"]:.4f}'
                      f'  loss={loss.item():.4f}')

    # Final evaluation of the fully trained model (post last update)
    model.eval()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
        final_metrics = task.evaluate(model, device)
    results.append({'step': last_completed_step + 1, **final_metrics})

    return results


# -- Config and CLI -----------------------------------------------------------


TASKS = {
    'mnist': lambda c: MNISTBandit(
        reward_noise=c.reward_noise, reward_noise_mode=c.reward_noise_mode),
    'token_reversal': lambda c: TokenReversal(
        vocab_size=c.vocab_size, seq_len=c.seq_len,
        binary_reward=c.binary_reward, reward_noise=c.reward_noise,
        reward_noise_mode=c.reward_noise_mode),
    'masked_reversal': lambda c: MaskedReversal(
        vocab_size=c.vocab_size, seq_len=c.seq_len, score_len=c.score_len,
        binary_reward=c.binary_reward, reward_noise=c.reward_noise,
        reward_noise_mode=c.reward_noise_mode),
    'chain_reversal': lambda c: RewardChainReversal(
        vocab_size=c.vocab_size, seq_len=c.seq_len,
        binary_reward=c.binary_reward, reward_noise=c.reward_noise,
        reward_noise_mode=c.reward_noise_mode),
    'chain_arithmetic': lambda c: ChainArithmetic(
        vocab_size=c.vocab_size, seq_len=c.seq_len,
        binary_reward=c.binary_reward, reward_noise=c.reward_noise,
        reward_noise_mode=c.reward_noise_mode),
    'format_answer': lambda c: FormatAnswerArithmetic(
        vocab_size=c.vocab_size, seq_len=c.seq_len,
        binary_reward=c.binary_reward, reward_noise=c.reward_noise,
        reward_noise_mode=c.reward_noise_mode),
    'lm_bandit': lambda c: LMBandit(
        model_name=c.model_name, context_len=c.context_len,
        kl_weight=c.kl_weight, reward_noise=c.reward_noise,
        reward_noise_mode=c.reward_noise_mode),
}

LOSSES = {
    'CE': lambda c: L.CELoss(),
    'REINFORCE': lambda c: L.REINFORCELoss(baseline=c.baseline),
    'PG': lambda c: L.PGLoss(baseline=c.baseline, iw_cap=c.iw_cap),
    'ASPO': lambda c: L.ASPOLoss(baseline=c.baseline, eps_low=c.clip_low, eps_high=c.clip_high),
    'TrajPG': lambda c: L.TrajectoryPGLoss(baseline=c.baseline, iw_cap=c.iw_cap),
    'DG': lambda c: L.DGLoss(eta=c.eta, baseline=c.baseline),
    'ReplayDG': lambda c: L.ReplayDGLoss(eta=c.eta, baseline=c.baseline),
    'FreshDG': lambda c: L.FreshDGLoss(
        eta=c.eta, baseline=c.baseline, age_decay=c.replay_age_decay),
    'DGEntropyGuard': lambda c: L.DGEntropyGuardLoss(
        eta=c.eta, baseline=c.baseline),
    'UncertaintyDG': lambda c: L.UncertaintyDGLoss(
        eta=c.eta, baseline=c.baseline, uncertainty_scale=c.uncertainty_scale),
    'FilteredDG': lambda c: L.FilteredDGLoss(
        eta=c.eta, baseline=c.baseline,
        uncertainty_threshold=c.uncertainty_threshold),
    'RewardVarianceDG': lambda c: L.RewardVarianceDGLoss(
        eta=c.eta, baseline=c.baseline, variance_scale=c.uncertainty_scale),
    'Kondo': lambda c: L.KondoLoss(eta=c.eta, keep_ratio=c.kondo_keep, baseline=c.baseline),
    'LogGrowth': lambda c: L.LogGrowthLoss(baseline=c.baseline),
    'DGToken': lambda c: L.DGTokenCreditLoss(eta=c.eta),
    'SelfDistillDG': lambda c: L.SelfDistillDGLoss(
        eta=c.eta, alpha=c.distill_alpha),
    'SCOPELite': lambda c: L.SCOPELiteLoss(
        eta=c.eta, baseline=c.baseline, alpha=c.distill_alpha),
    'GRPO': lambda c: L.GRPOLoss(eps=c.clip_low, beta=c.grpo_beta),
    'DrGRPO': lambda c: L.DrGRPOLoss(eps=c.clip_low, beta=c.grpo_beta),
    'DAPO': lambda c: L.DAPOLiteLoss(eps_low=c.clip_low, eps_high=c.clip_high),
    'DAPOLite': lambda c: L.DAPOLiteLoss(eps_low=c.clip_low, eps_high=c.clip_high),
    'TPO': lambda c: L.TPOLoss(eta=c.tpo_eta),
    'TPONoAnchor': lambda c: L.TPONoAnchorLoss(eta=c.tpo_eta),
    'GroupPG': lambda c: L.GroupPGLoss(eta=c.tpo_eta),
    'TPOFullAction': lambda c: L.TPOFullActionLoss(eta=c.tpo_eta),
    'TPOToken': lambda c: L.TPOTokenLoss(eta=c.tpo_eta),
    'GRPOToken': lambda c: L.GRPOTokenLoss(eps=c.clip_low, beta=c.grpo_beta),
    'TEMPO': lambda c: L.TEMPOLoss(iw_cap=c.iw_cap),
    'MaxRL': lambda c: L.MaxRLLoss(iw_cap=c.iw_cap),
    'R2VPO': lambda c: L.R2VPOLoss(baseline=c.baseline, lam=c.eta),
    'PMDMean': lambda c: L.PMDMeanLoss(tau=c.eta),
}

MODEL_BUILDERS = {
    'mnist': lambda c, task: task.make_model(hidden=c.hidden),
    'token_reversal': lambda c, task: task.make_model(
        d_model=c.d_model, nhead=c.nhead, num_layers=c.num_layers),
    'masked_reversal': lambda c, task: task.make_model(
        d_model=c.d_model, nhead=c.nhead, num_layers=c.num_layers),
    'chain_reversal': lambda c, task: task.make_model(
        d_model=c.d_model, nhead=c.nhead, num_layers=c.num_layers),
    'chain_arithmetic': lambda c, task: task.make_model(
        d_model=c.d_model, nhead=c.nhead, num_layers=c.num_layers),
    'format_answer': lambda c, task: task.make_model(
        d_model=c.d_model, nhead=c.nhead, num_layers=c.num_layers),
    'lm_bandit': lambda c, task: task.make_model(),
}


@dataclass
class Config:
    task: str = 'mnist'
    method: str = 'DG'
    delay: int = 0
    num_steps: int = 1_000
    batch_size: int = 100
    lr: float = 1e-3
    inner_epochs: int = 1
    eval_every: int = 20
    num_seeds: int = 5
    seed: int = 0
    baseline: str = 'expected'
    diagnostics: bool = False
    entropy_diagnostics: bool = False
    verbose: bool = True
    output: str = 'results.csv'
    sweep: bool = False
    # MLP
    hidden: int = 50
    # DG / Kondo / PMDMean
    eta: float = 1.0
    # TPO
    tpo_eta: float = 1.0
    grpo_beta: float = 0.04
    # PG
    iw_cap: float = 10.0
    # ASPO
    clip_low: float = 0.2
    clip_high: float = 0.28
    # Kondo
    kondo_keep: float = 0.5
    # Freshness-aware replay
    replay_capacity: int = 0
    replay_priority: str = 'fresh_delight'
    replay_age_decay: float = 0.02
    # Reward-noise / uncertainty
    reward_noise: float = 0.0
    reward_noise_mode: str = 'none'
    uncertainty_scale: float = 1.0
    uncertainty_threshold: float = 0.5
    # Dense correction
    distill_alpha: float = 0.5
    # MaxRL
    group_size: int = 4
    # Token reversal / masked reversal
    vocab_size: int = 2
    seq_len: int = 10
    score_len: int = 5
    binary_reward: bool = False
    d_model: int = 64
    nhead: int = 2
    num_layers: int = 2
    # LM bandit
    model_name: str = 'distilgpt2'
    context_len: int = 128
    kl_weight: float = 0.0


def _validate_basic_config(config: Config):
    if config.task not in TASKS:
        raise ValueError(f'Unknown task: {config.task}')
    if config.method not in LOSSES:
        raise ValueError(f'Unknown method: {config.method}')
    if config.delay < 0:
        raise ValueError('delay must be >= 0')
    if config.num_steps < 1:
        raise ValueError('num_steps must be >= 1')
    if config.batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    if config.eval_every < 1:
        raise ValueError('eval_every must be >= 1')
    if config.num_seeds < 1:
        raise ValueError('num_seeds must be >= 1')
    if config.lr <= 0:
        raise ValueError('lr must be > 0')
    if config.baseline not in BASELINES:
        raise ValueError(f'Unknown baseline: {config.baseline}')
    if config.inner_epochs < 1:
        raise ValueError('inner_epochs must be >= 1')
    if config.group_size < 1:
        raise ValueError('group_size must be >= 1')
    if config.replay_capacity < 0:
        raise ValueError('replay_capacity must be >= 0')
    if config.replay_capacity > 0 and config.delay > 0:
        if config.replay_capacity <= config.delay:
            raise ValueError('replay_capacity must be > delay to expose stale replay')
    if config.replay_age_decay < 0:
        raise ValueError('replay_age_decay must be >= 0')
    if config.replay_priority not in REPLAY_PRIORITIES:
        raise ValueError(f'Unknown replay_priority: {config.replay_priority}')
    if config.seq_len < 1:
        raise ValueError('seq_len must be >= 1')
    if config.task in ('chain_arithmetic', 'format_answer') and config.seq_len < 2:
        raise ValueError(f'{config.task} requires seq_len >= 2')
    if (config.task == 'masked_reversal'
            and (config.score_len < 1 or config.score_len > config.seq_len)):
        raise ValueError('score_len must be in [1, seq_len]')
    if config.vocab_size < 1:
        raise ValueError('vocab_size must be >= 1')
    if config.context_len < 1:
        raise ValueError('context_len must be >= 1')


def _validate_reward_noise_config(config: Config):
    if not (0.0 <= config.reward_noise <= 1.0):
        raise ValueError('reward_noise must be in [0, 1]')
    if config.reward_noise > 0 and config.reward_noise_mode == 'none':
        raise ValueError('reward_noise_mode must not be none when reward_noise > 0')
    if config.reward_noise == 0 and config.reward_noise_mode != 'none':
        raise ValueError('reward_noise_mode must be none when reward_noise == 0')
    if config.reward_noise_mode not in REWARD_NOISE_MODES:
        raise ValueError(f'Unknown reward_noise_mode: {config.reward_noise_mode}')

    allowed_modes = (
        BANDIT_REWARD_NOISE_MODES
        if config.task in BANDIT_TASKS
        else SEQUENCE_REWARD_NOISE_MODES
    )
    if config.reward_noise_mode not in allowed_modes:
        raise ValueError(
            f'{config.reward_noise_mode} is not supported for task {config.task}')


def _validate_method_regime(config: Config):
    if config.method in GROUPED_METHODS:
        if config.group_size <= 1:
            raise ValueError(f'{config.method} requires group_size > 1')
        if config.batch_size % config.group_size != 0:
            raise ValueError('batch_size must be divisible by group_size')

    if config.method in SEQUENTIAL_METHODS and config.task in BANDIT_TASKS:
        raise ValueError(f'{config.method} requires a sequential task')

    if config.method in TOKEN_CANDIDATE_METHODS:
        if config.task not in ('token_reversal', 'masked_reversal'):
            raise ValueError(f'{config.method} requires token_reversal or masked_reversal')
        if config.binary_reward:
            raise ValueError(f'{config.method} requires dense token rewards')
        if config.group_size <= 1:
            raise ValueError(f'{config.method} requires group_size > 1')
        if config.delay != 0:
            raise ValueError(f'{config.method} requires delay=0')
        if config.replay_capacity != 0:
            raise ValueError(f'{config.method} requires replay_capacity=0')
        if config.reward_noise > 0:
            raise ValueError(f'{config.method} requires clean dense token rewards')
        if config.diagnostics or config.entropy_diagnostics:
            raise ValueError(f'{config.method} does not support diagnostics yet')

    if config.method == 'DGToken':
        if config.task not in ('token_reversal', 'masked_reversal'):
            raise ValueError('DGToken requires token_reversal or masked_reversal')
        if config.binary_reward:
            raise ValueError('DGToken requires a fractional sequential reward')
        if config.reward_noise > 0:
            raise ValueError('DGToken requires decomposable clean rewards')

    if config.method == 'ReplayDG' and config.replay_capacity == 0:
        raise ValueError('ReplayDG requires replay_capacity > 0')
    if config.method == 'FreshDG' and config.replay_capacity == 0 and config.delay == 0:
        raise ValueError('FreshDG requires replay_capacity > 0 or delay > 0')

    if config.method == 'LogGrowth':
        if config.task not in BANDIT_TASKS:
            raise ValueError('LogGrowth requires a one-step bandit task')
        if config.kl_weight != 0:
            raise ValueError('LogGrowth requires kl_weight=0')
        if config.reward_noise > 0:
            raise ValueError('LogGrowth requires clean correctness rewards')

    if config.method == 'MaxRL':
        if config.task in REWARD_CHAIN_TASKS:
            raise ValueError('MaxRL requires binary rewards, not reward-chain rewards')
        if config.task in ('token_reversal', 'masked_reversal') and not config.binary_reward:
            raise ValueError('MaxRL requires binary_reward=true on sequence tasks')
        if config.task == 'lm_bandit' and config.kl_weight != 0:
            raise ValueError('MaxRL requires binary rewards; set kl_weight=0')
        if config.reward_noise > 0:
            raise ValueError('MaxRL requires clean binary rewards')

    if config.method in ('DAPO', 'DAPOLite') and config.kl_weight != 0:
        raise ValueError('DAPO-lite removes the KL penalty; set kl_weight=0')

    if config.method == 'TPOFullAction':
        if config.task != 'mnist':
            raise ValueError('TPOFullAction is scoped to the MNIST bandit')
        if config.delay != 0:
            raise ValueError('TPOFullAction requires delay=0')
        if config.replay_capacity != 0:
            raise ValueError('TPOFullAction requires replay_capacity=0')
        if config.inner_epochs != 1:
            raise ValueError('TPOFullAction requires inner_epochs=1')
        if config.reward_noise > 0:
            raise ValueError('TPOFullAction requires clean bandit feedback')

    if config.task in REWARD_CHAIN_TASKS and config.binary_reward:
        raise ValueError(f'{config.task} uses fractional checkpoint rewards')


def _validate_method_hyperparams(config: Config):
    if config.kondo_keep <= 0 or config.kondo_keep > 1:
        raise ValueError('kondo_keep must be in (0, 1]')
    if config.clip_low < 0 or config.clip_high < 0:
        raise ValueError('clip bounds must be non-negative')
    if config.method in ETA_METHODS and config.eta <= 0:
        raise ValueError(f'{config.method} requires eta > 0')
    if config.method in TPO_METHODS and config.tpo_eta <= 0:
        raise ValueError(f'{config.method} requires tpo_eta > 0')
    if config.method == 'R2VPO' and config.eta < 0:
        raise ValueError('R2VPO requires eta >= 0')
    if config.grpo_beta < 0:
        raise ValueError('grpo_beta must be >= 0')
    if config.method == 'PMDMean' and config.eta <= 0:
        raise ValueError('PMDMean requires eta > 0')
    if config.uncertainty_scale < 0:
        raise ValueError('uncertainty_scale must be >= 0')
    if config.uncertainty_threshold < 0:
        raise ValueError('uncertainty_threshold must be >= 0')
    if config.distill_alpha < 0:
        raise ValueError('distill_alpha must be >= 0')


def validate_config(config: Config):
    _validate_basic_config(config)
    _validate_reward_noise_config(config)
    _validate_method_regime(config)
    _validate_method_hyperparams(config)


def run_config(config: Config) -> pd.DataFrame:
    validate_config(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task = TASKS[config.task](config)

    all_rows = []
    for i in range(config.num_seeds):
        seed = config.seed + i
        if config.verbose:
            print(f'{config.method} delay={config.delay} seed={seed}')
        torch.manual_seed(seed)
        loss_fn = LOSSES[config.method](config)
        model = MODEL_BUILDERS[config.task](config, task)
        rows = train_one_seed(task, loss_fn, model, config, seed, device)
        for r in rows:
            r.update({'seed': seed, 'method': config.method, 'delay': config.delay})
        all_rows.extend(rows)
        # Write after each seed so completed work survives crashes
        pd.DataFrame(all_rows).to_csv(config.output, index=False)

    return pd.DataFrame(all_rows)


def run_sweep(config: Config) -> pd.DataFrame:
    dfs = []
    for method in ['REINFORCE', 'PG', 'ASPO', 'TrajPG', 'DG', 'Kondo']:
        for delay in [0, 1, 3, 10, 30, 100]:
            cfg = dataclasses.replace(config, method=method, delay=delay)
            dfs.append(run_config(cfg))
    return pd.concat(dfs, ignore_index=True)


# -- CLI ----------------------------------------------------------------------

TYPE_MAP = {int: int, float: float, str: str, bool: bool}


def parse_bool(text: str) -> bool:
    lowered = text.lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    raise argparse.ArgumentTypeError('expected true or false')


def main():
    parser = argparse.ArgumentParser(description='Delightful Policy Gradient')
    for f in dataclasses.fields(Config):
        ty = TYPE_MAP[f.type] if f.type in TYPE_MAP else str
        if f.type is bool:
            parser.add_argument(f'--{f.name}', type=parse_bool,
                                default=f.default, metavar='BOOL')
        else:
            parser.add_argument(f'--{f.name}', type=ty, default=f.default)
    args = parser.parse_args()
    config = Config(**{
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(Config)
    })

    t0 = time.time()
    df = run_sweep(config) if config.sweep else run_config(config)
    df.to_csv(config.output, index=False)
    print(f'Saved {len(df)} rows to {config.output} ({time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
