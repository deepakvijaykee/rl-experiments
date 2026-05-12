"""Tasks: bandits, autoregressive reversal, and reward-chain toys.

Each task provides:
  - sample_batch(model, batch_size, device, group_size=1) -> Batch
      Uses model directly as actor in eval() mode. No stale params -
      staleness is handled by the experience queue in the training loop.
  - compute_logits(model, batch) -> Tensor
      Learner forward pass. Logits aligned with batch.actions.
  - compute_logits_oracle(model, batch) -> Tensor
      For CE: conditions on ground truth, not actor-generated sequence.
      Same as compute_logits for bandit tasks. Different for sequential tasks.
  - evaluate(model, device) -> dict
  - make_model() -> nn.Module
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets


@dataclass
class Batch:
    """A batch of bandit/RL experience.

    Stores sufficient statistics instead of full actor distributions:
      actor_logp_a: [B] or [B, T] - log-prob of taken action under actor policy
      actor_baseline: [B] or [B, T] - sum_a pi(a)^2 per-slot baseline
    This scales with batch size, not vocab size.
    """
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    labels: torch.Tensor
    actor_logp_a: torch.Tensor   # [B] or [B, T]
    actor_baseline: torch.Tensor  # [B] or [B, T]
    group_ids: torch.Tensor | None = None
    score_mask: torch.Tensor | None = None  # [B, T] bool: which positions are scored
    actor_expected_reward: torch.Tensor | None = None  # [B]: exact E[R|x] under actor
    # Sampler-side group diagnostics (plain scalars, not tensors)
    informative_group_rate: float | None = None  # pre-filter mixed-group rate
    retained_group_rate: float | None = None     # post-filter fraction kept
    used_group_fallback: bool = False            # True if empty-batch safeguard fired
    age: int = 0                                 # optimizer steps since sampling

    def to(self, device) -> Batch:
        return replace(
            self,
            obs=self.obs.to(device), actions=self.actions.to(device),
            rewards=self.rewards.to(device), labels=self.labels.to(device),
            actor_logp_a=self.actor_logp_a.to(device),
            actor_baseline=self.actor_baseline.to(device),
            group_ids=(
                self.group_ids.to(device)
                if self.group_ids is not None else None),
            score_mask=(
                self.score_mask.to(device)
                if self.score_mask is not None else None),
            actor_expected_reward=(
                self.actor_expected_reward.to(device)
                if self.actor_expected_reward is not None else None))

    def select(self, mask: torch.Tensor) -> Batch:
        return replace(
            self,
            obs=self.obs[mask], actions=self.actions[mask],
            rewards=self.rewards[mask], labels=self.labels[mask],
            actor_logp_a=self.actor_logp_a[mask],
            actor_baseline=self.actor_baseline[mask],
            group_ids=self.group_ids[mask] if self.group_ids is not None else None,
            score_mask=self.score_mask[mask] if self.score_mask is not None else None,
            actor_expected_reward=(
                self.actor_expected_reward[mask]
                if self.actor_expected_reward is not None else None))

    def with_age(self, age: int) -> Batch:
        return replace(self, age=age)


@dataclass
class TokenCandidateBatch:
    """Per-prefix candidate simplex for token-candidate TPO/GRPO.

    Each row in `states` is one autoregressive prefix padded to a common
    length. `step_ids` selects the prefix position whose logits predict the
    next token. Candidate fields are shaped [B*T, K].
    """
    states: torch.Tensor
    step_ids: torch.Tensor
    candidate_actions: torch.Tensor
    old_candidate_logp: torch.Tensor
    candidate_rewards: torch.Tensor
    behavior_actions: torch.Tensor
    behavior_rewards: torch.Tensor
    behavior_sequence_rewards: torch.Tensor
    labels: torch.Tensor

    def to(self, device) -> TokenCandidateBatch:
        return replace(
            self,
            states=self.states.to(device),
            step_ids=self.step_ids.to(device),
            candidate_actions=self.candidate_actions.to(device),
            old_candidate_logp=self.old_candidate_logp.to(device),
            candidate_rewards=self.candidate_rewards.to(device),
            behavior_actions=self.behavior_actions.to(device),
            behavior_rewards=self.behavior_rewards.to(device),
            behavior_sequence_rewards=self.behavior_sequence_rewards.to(device),
            labels=self.labels.to(device))


def corrupt_rewards(rewards: torch.Tensor, actions: torch.Tensor, labels: torch.Tensor,
                    noise: float, mode: str, rare_token: int | None = None,
                    obs: torch.Tensor | None = None) -> torch.Tensor:
    """Apply controlled verifier/proxy corruption to sampled rewards."""
    if noise <= 0 or mode == 'none':
        return rewards

    if not (0.0 <= noise <= 1.0):
        raise ValueError('reward_noise must be in [0, 1]')
    noisy = rewards.clone()
    sample_shape = rewards.shape
    draw = torch.rand(sample_shape, device=rewards.device) < noise

    if mode == 'label_flip':
        return torch.where(draw, 1.0 - noisy, noisy)

    if mode == 'random_reward':
        return torch.where(draw, torch.rand_like(noisy), noisy)

    if actions.dim() == 1:
        action0 = actions == 0
        incorrect = actions != labels
        if mode in ('false_positive_action0', 'spurious_feature'):
            return torch.where(draw & action0 & incorrect, torch.ones_like(noisy), noisy)
        raise ValueError(f'Unknown reward_noise_mode for bandit task: {mode}')

    if actions.dim() == 2 and labels.dim() == 1:
        action0 = actions == 0
        incorrect = actions != labels.unsqueeze(1)
        if mode in ('false_positive_action0', 'spurious_feature'):
            return torch.where(draw & action0 & incorrect, torch.ones_like(noisy), noisy)
        raise ValueError(f'Unknown reward_noise_mode for grouped bandit task: {mode}')

    incorrect_seq = ~(actions == labels).all(dim=-1)
    if mode == 'false_positive_action0':
        return torch.where(
            draw & actions.eq(0).any(dim=-1) & incorrect_seq,
            torch.ones_like(noisy), noisy)

    any_rare = actions.eq(rare_token if rare_token is not None else 0).any(dim=-1)
    if mode in ('false_positive_rare_token', 'false_positive_sep'):
        return torch.where(draw & any_rare & incorrect_seq, torch.ones_like(noisy), noisy)

    if mode == 'spurious_feature':
        if obs is None:
            raise ValueError('spurious_feature sequence noise requires obs')
        spurious = actions[:, 0] == obs[:, 0]
        return torch.where(draw & spurious & incorrect_seq, torch.ones_like(noisy), noisy)

    raise ValueError(f'Unknown reward_noise_mode: {mode}')


# -- MNIST Contextual Bandit --------------------------------------------------


class MNISTBandit:
    """MNIST as a one-step contextual bandit."""
    num_actions = 10

    def __init__(self, data_dir: str = './data', reward_noise: float = 0.0,
                 reward_noise_mode: str = 'none'):
        train = datasets.MNIST(data_dir, train=True, download=True)
        test = datasets.MNIST(data_dir, train=False, download=True)
        self.train_images = train.data.float().reshape(-1, 784) / 255.0
        self.train_labels = train.targets
        self.test_images = test.data.float().reshape(-1, 784) / 255.0
        self.test_labels = test.targets
        self.reward_noise = reward_noise
        self.reward_noise_mode = reward_noise_mode

    def make_model(self, hidden: int) -> nn.Module:
        from .models import MLP
        return MLP(obs_dim=784, hidden=hidden, num_actions=10)

    def sample_batch(self, model: nn.Module, batch_size: int,
                     device: torch.device, group_size: int = 1) -> Batch:
        if group_size > 1:
            return self._sample_grouped(model, batch_size, group_size, device)

        idx = torch.randint(len(self.train_images), (batch_size,))
        images = self.train_images[idx].to(device)
        labels = self.train_labels[idx].to(device)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            actor_logits = model(images)
            actor_lp = F.log_softmax(actor_logits, dim=-1)
            actor_probs = F.softmax(actor_logits, dim=-1)
            actions = torch.distributions.Categorical(logits=actor_logits).sample()
            logp_a = actor_lp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            baseline = (actor_probs ** 2).sum(-1)
            p_success = actor_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        model.train(was_training)

        clean_rewards = (actions == labels).float()
        rewards = corrupt_rewards(
            clean_rewards, actions, labels, self.reward_noise,
            self.reward_noise_mode)
        expected_reward = None if self.reward_noise > 0 else p_success

        return Batch(obs=images, actions=actions,
                     rewards=rewards,
                     labels=labels, actor_logp_a=logp_a, actor_baseline=baseline,
                     actor_expected_reward=expected_reward)

    def _sample_grouped(self, model, batch_size, group_size, device) -> Batch:
        """K rollouts per context for grouped methods. Returns flattened batch with group_ids."""
        num_contexts = batch_size // group_size
        idx = torch.randint(len(self.train_images), (num_contexts,))
        images = self.train_images[idx].to(device)
        labels = self.train_labels[idx].to(device)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            actor_logits = model(images)
            actor_lp = F.log_softmax(actor_logits, dim=-1)
            actor_bl = (F.softmax(actor_logits, dim=-1) ** 2).sum(-1)  # [N]
            actions = torch.stack([
                torch.distributions.Categorical(logits=actor_logits).sample()
                for _ in range(group_size)], dim=1)  # [N, K]
        model.train(was_training)

        clean_rewards = (actions == labels.unsqueeze(1)).float()  # [N, K]
        rewards = corrupt_rewards(
            clean_rewards, actions, labels, self.reward_noise,
            self.reward_noise_mode)

        raw_keep = (rewards.sum(1) > 0) & (rewards.sum(1) < group_size)
        informative_rate = raw_keep.float().mean().item()
        fallback = raw_keep.sum() == 0
        if fallback:
            raw_keep[:] = True

        n_valid = raw_keep.sum().item()
        K = group_size

        images_f = images[raw_keep].unsqueeze(1).expand(-1, K, -1).reshape(-1, 784)
        labels_f = labels[raw_keep].unsqueeze(1).expand(-1, K).reshape(-1)
        actions_f = actions[raw_keep].reshape(-1)
        rewards_f = rewards[raw_keep].reshape(-1)
        # Per-action log-probs: actor_lp [N, V], actions [N, K] -> [N, K]
        logp_a_f = actor_lp[raw_keep].gather(-1, actions[raw_keep]).reshape(-1)
        baseline_f = actor_bl[raw_keep].unsqueeze(1).expand(-1, K).reshape(-1)
        group_ids = torch.arange(n_valid, device=device).unsqueeze(1).expand(-1, K).reshape(-1)

        return Batch(obs=images_f, actions=actions_f, rewards=rewards_f,
                     labels=labels_f, actor_logp_a=logp_a_f,
                     actor_baseline=baseline_f, group_ids=group_ids,
                     informative_group_rate=informative_rate,
                     retained_group_rate=n_valid / num_contexts,
                     used_group_fallback=fallback.item())

    def compute_logits(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        return model(batch.obs)

    def compute_logits_oracle(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        return model(batch.obs)

    def compute_difficulty(self, model: nn.Module, device: torch.device):
        return None

    @torch.no_grad()
    def evaluate(self, model: nn.Module, device: torch.device) -> dict[str, float]:
        was_training = model.training
        model.eval()
        logits = model(self.test_images.to(device))
        acc = (logits.argmax(-1) == self.test_labels.to(device)).float().mean().item()
        lp = F.log_softmax(logits, dim=-1)
        entropy = -(lp.exp() * lp).sum(dim=-1).mean().item()
        model.train(was_training)
        return {'test_error': 1.0 - acc, 'entropy': entropy}


# -- Token Reversal -----------------------------------------------------------


class TokenReversal:
    """Token reversal task from DG paper Section 5."""

    def __init__(self, vocab_size: int, seq_len: int, binary_reward: bool = False,
                 reward_noise: float = 0.0, reward_noise_mode: str = 'none'):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.binary_reward = binary_reward
        self.reward_noise = reward_noise
        self.reward_noise_mode = reward_noise_mode
        self.effective_vocab = vocab_size + 1
        self.sep_token = vocab_size
        self.num_actions = self.effective_vocab

    def make_model(self, d_model: int, nhead: int, num_layers: int) -> nn.Module:
        from .models import CausalTransformer
        return CausalTransformer(
            vocab_size=self.effective_vocab, d_model=d_model,
            nhead=nhead, num_layers=num_layers,
            max_seq_len=self.seq_len * 2 + 1)

    def sample_inputs(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(self.vocab_size, (batch_size, self.seq_len), device=device)

    def target_tokens(self, input_tokens: torch.Tensor) -> torch.Tensor:
        return input_tokens.flip(1)

    def reward_from_actions(self, actions: torch.Tensor,
                            targets: torch.Tensor) -> torch.Tensor:
        correct = (actions == targets).float()
        if self.binary_reward:
            return correct.all(dim=1).float()
        return correct.mean(dim=1)

    def token_candidate_rewards(self, candidate_actions: torch.Tensor,
                                targets: torch.Tensor, step: int,
                                alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        correct = (candidate_actions == targets[:, step:step + 1]).float()
        if self.binary_reward:
            rewards = alive[:, None] * correct
            return rewards, rewards[:, 0]
        return correct, alive

    def _rollout(self, model, input_tokens, device):
        """Single autoregressive rollout. Returns sufficient stats, not full distributions."""
        H = self.seq_len
        B = input_tokens.shape[0]
        target_tokens = self.target_tokens(input_tokens)
        sep = torch.full((B, 1), self.sep_token, device=device, dtype=torch.long)
        prefix = torch.cat([input_tokens, sep], dim=1)

        generated, per_token_logp, per_token_baseline = [], [], []
        with torch.no_grad():
            for _ in range(H):
                logits = model(prefix)
                next_logits = logits[:, -1]
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.distributions.Categorical(probs=probs).sample()
                generated.append(next_token)
                lp = F.log_softmax(next_logits, dim=-1)
                per_token_logp.append(lp.gather(-1, next_token.unsqueeze(-1)).squeeze(-1))
                per_token_baseline.append((probs ** 2).sum(-1))
                prefix = torch.cat([prefix, next_token.unsqueeze(1)], dim=1)

        actions = torch.stack(generated, dim=1)           # [B, H]
        logp_a = torch.stack(per_token_logp, dim=1)       # [B, H]
        baseline = torch.stack(per_token_baseline, dim=1)  # [B, T]
        rewards = self.reward_from_actions(actions, target_tokens)
        obs = torch.cat([input_tokens, sep, actions], dim=1)
        return actions, logp_a, baseline, rewards, obs, target_tokens

    def sample_batch(self, model: nn.Module, batch_size: int,
                     device: torch.device, group_size: int = 1) -> Batch:
        H = self.seq_len
        was_training = model.training
        model.eval()

        if group_size == 1:
            input_tokens = self.sample_inputs(batch_size, device)
            actions, logp_a, baseline, rewards, obs, labels = self._rollout(
                model, input_tokens, device)

            # Exact P(success) for binary tasks: one teacher-forced oracle pass
            p_success = None
            if self.binary_reward:
                oracle_prefix = torch.cat([input_tokens,
                                           obs[:, H:H+1],  # sep token
                                           labels[:, :-1]], dim=1)
                with torch.no_grad():
                    oracle_lp = F.log_softmax(model(oracle_prefix)[:, H:, :], dim=-1)
                    p_success = oracle_lp.gather(
                        -1, labels.unsqueeze(-1)).squeeze(-1).sum(dim=1).exp()

            rewards = corrupt_rewards(
                rewards, actions, labels, self.reward_noise,
                self.reward_noise_mode, rare_token=self.sep_token, obs=obs)
            if self.reward_noise > 0:
                p_success = None

            model.train(was_training)
            return Batch(obs=obs, actions=actions, rewards=rewards,
                         labels=labels, actor_logp_a=logp_a, actor_baseline=baseline,
                         actor_expected_reward=p_success)

        # K rollouts per input for grouped methods (MaxRL, etc.)
        num_contexts = batch_size // group_size
        input_tokens = self.sample_inputs(num_contexts, device)

        all_a, all_logp, all_bl, all_r, all_obs, all_lab = [], [], [], [], [], []
        for _ in range(group_size):
            a, lp, bl, r, o, l = self._rollout(model, input_tokens, device)
            all_a.append(a)
            all_logp.append(lp)
            all_bl.append(bl)
            all_r.append(r)
            all_obs.append(o)
            all_lab.append(l)
        model.train(was_training)

        actions = torch.stack(all_a, dim=1)      # [N, K, H]
        logp_a = torch.stack(all_logp, dim=1)    # [N, K, H]
        baselines = torch.stack(all_bl, dim=1)   # [N, K, H]
        rewards = torch.stack(all_r, dim=1)       # [N, K]
        obs = torch.stack(all_obs, dim=1)         # [N, K, 2H+1]
        labels = torch.stack(all_lab, dim=1)      # [N, K, H]

        rewards = corrupt_rewards(
            rewards.reshape(-1), actions.reshape(-1, H), labels.reshape(-1, H),
            self.reward_noise, self.reward_noise_mode,
            rare_token=self.sep_token, obs=obs.reshape(-1, obs.size(-1))
        ).reshape_as(rewards)

        # Filter zero-variance groups (continuous rewards need std, not sum)
        raw_keep = rewards.std(1) > 1e-6
        informative_rate = raw_keep.float().mean().item()
        fallback = raw_keep.sum() == 0
        if fallback:
            raw_keep[:] = True

        n_valid = raw_keep.sum().item()
        K = group_size
        group_ids = torch.arange(n_valid, device=device).unsqueeze(1).expand(-1, K).reshape(-1)

        return Batch(
            obs=obs[raw_keep].reshape(-1, obs.size(-1)),
            actions=actions[raw_keep].reshape(-1, H),
            rewards=rewards[raw_keep].reshape(-1),
            labels=labels[raw_keep].reshape(-1, H),
            actor_logp_a=logp_a[raw_keep].reshape(-1, H),
            actor_baseline=baselines[raw_keep].reshape(-1, H),
            group_ids=group_ids,
            informative_group_rate=informative_rate,
            retained_group_rate=n_valid / num_contexts,
            used_group_fallback=fallback.item())

    def sample_token_candidates(self, model: nn.Module, batch_size: int,
                                device: torch.device,
                                group_size: int) -> TokenCandidateBatch:
        """Sample K next-token candidates at every generated prefix.

        The first candidate is the behavior action that extends the rollout.
        Candidate rewards are dense per-token verifier rewards, matching the
        token-candidate TPO/GRPO contract in jeankaddour/tpo.
        """
        if group_size <= 1:
            raise ValueError('token-candidate methods require group_size > 1')

        H = self.seq_len
        input_tokens = self.sample_inputs(batch_size, device)
        targets = self.target_tokens(input_tokens)
        sep = torch.full((batch_size, 1), self.sep_token,
                         device=device, dtype=torch.long)
        prefix = torch.cat([input_tokens, sep], dim=1)
        alive = torch.ones(batch_size, device=device)
        state_len = 2 * H

        states, step_ids = [], []
        candidate_actions, candidate_logp, candidate_rewards = [], [], []
        behavior_actions, behavior_rewards = [], []

        was_training = model.training
        model.eval()
        with torch.no_grad():
            for step in range(H):
                state = torch.zeros(
                    batch_size, state_len, dtype=torch.long, device=device)
                state[:, :prefix.size(1)] = prefix
                logits = model(prefix)[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                candidates = torch.distributions.Categorical(
                    logits=logits).sample((group_size,)).transpose(0, 1)
                rewards, next_alive = self.token_candidate_rewards(
                    candidates, targets, step, alive)
                behavior = candidates[:, 0]
                behavior_reward = rewards[:, 0]

                states.append(state)
                step_ids.append(torch.full(
                    (batch_size,), H + step, dtype=torch.long, device=device))
                candidate_actions.append(candidates)
                candidate_logp.append(log_probs.gather(-1, candidates))
                candidate_rewards.append(rewards)
                behavior_actions.append(behavior)
                behavior_rewards.append(behavior_reward)

                alive = next_alive
                prefix = torch.cat([prefix, behavior.unsqueeze(1)], dim=1)
        model.train(was_training)

        behavior_actions_t = torch.stack(behavior_actions, dim=1)
        behavior_rewards_t = torch.stack(behavior_rewards, dim=1)
        sequence_rewards = self.reward_from_actions(behavior_actions_t, targets)

        return TokenCandidateBatch(
            states=torch.stack(states, dim=1).reshape(batch_size * H, state_len),
            step_ids=torch.stack(step_ids, dim=1).reshape(batch_size * H),
            candidate_actions=torch.stack(candidate_actions, dim=1).reshape(
                batch_size * H, group_size),
            old_candidate_logp=torch.stack(candidate_logp, dim=1).reshape(
                batch_size * H, group_size),
            candidate_rewards=torch.stack(candidate_rewards, dim=1).reshape(
                batch_size * H, group_size),
            behavior_actions=behavior_actions_t,
            behavior_rewards=behavior_rewards_t,
            behavior_sequence_rewards=sequence_rewards,
            labels=targets)

    def compute_logits(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        """Teacher-forced on ACTOR-GENERATED prefix -> logits at output positions."""
        logits_full = model(batch.obs[:, :-1])
        return logits_full[:, self.seq_len:, :]

    def compute_token_candidate_logits(
            self, model: nn.Module, batch: TokenCandidateBatch) -> torch.Tensor:
        logits = model(batch.states)
        rows = torch.arange(batch.states.size(0), device=batch.states.device)
        return logits[rows, batch.step_ids, :]

    def compute_difficulty(self, model: nn.Module, device: torch.device):
        return None

    def compute_logits_oracle(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        """Teacher-forced on GROUND TRUTH prefix -> true supervised oracle.

        CE with compute_logits conditions on the actor's (possibly wrong) prefix.
        This method conditions on the correct reversed sequence, making the CE
        comparison a proper upper bound.
        """
        H = self.seq_len
        input_tokens = batch.obs[:, :H]
        sep = batch.obs[:, H:H + 1]
        # Build oracle prefix: [input, sep, target_1, ..., target_{H-1}]
        oracle_prefix = torch.cat([input_tokens, sep, batch.labels[:, :-1]], dim=1)
        logits_full = model(oracle_prefix)
        return logits_full[:, H:, :]  # [B, H, V]

    @torch.no_grad()
    def _greedy_generate(self, model, input_tokens):
        H = self.seq_len
        B = input_tokens.size(0)
        device = input_tokens.device
        target_tokens = self.target_tokens(input_tokens)
        sep = torch.full((B, 1), self.sep_token, device=device, dtype=torch.long)
        prefix = torch.cat([input_tokens, sep], dim=1)
        total_entropy = 0.0

        for _ in range(H):
            step_logits = model(prefix)[:, -1]
            lp = F.log_softmax(step_logits, dim=-1)
            total_entropy += -(lp.exp() * lp).sum(dim=-1).sum().item()
            next_token = step_logits.argmax(dim=-1)
            prefix = torch.cat([prefix, next_token.unsqueeze(1)], dim=1)

        return prefix[:, H + 1:], target_tokens, total_entropy, B * H

    @torch.no_grad()
    def evaluate(self, model: nn.Module, device: torch.device,
                 num_batches: int = 10, batch_size: int = 100) -> dict[str, float]:
        H, M = self.seq_len, self.vocab_size
        total_correct, total_tokens, total_exact, total_seqs = 0, 0, 0, 0
        total_entropy, total_ent_tokens = 0.0, 0
        was_training = model.training
        model.eval()
        for _ in range(num_batches):
            input_tokens = self.sample_inputs(batch_size, device)
            generated, target_tokens, entropy, ent_tokens = self._greedy_generate(
                model, input_tokens)
            total_entropy += entropy
            total_ent_tokens += ent_tokens
            total_correct += (generated == target_tokens).float().sum().item()
            total_tokens += batch_size * H
            total_exact += (generated == target_tokens).all(dim=1).float().sum().item()
            total_seqs += batch_size
        result = {'entropy': total_entropy / total_ent_tokens}
        if self.binary_reward:
            result['test_error'] = 1.0 - total_exact / total_seqs
        else:
            result['test_error'] = 1.0 - total_correct / total_tokens
        model.train(was_training)
        return result


# -- Masked Reversal ---------------------------------------------------------


class MaskedReversal(TokenReversal):
    """Partial-reward autoregressive credit benchmark.

    Reversal where only the last score_len output positions affect reward.
    The model must generate all H positions autoregressively but doesn't
    know which positions are scored.

    Note: this is NOT a benchmark where only scored positions deserve
    gradient. In an autoregressive model, unscored prefix tokens causally
    condition the scored suffix. The correct claim is weaker: DGToken
    should concentrate more budget on tokens that most affect future
    scored reward.
    """

    def __init__(self, vocab_size: int, seq_len: int, score_len: int,
                 binary_reward: bool = False, reward_noise: float = 0.0,
                 reward_noise_mode: str = 'none'):
        super().__init__(
            vocab_size, seq_len, binary_reward=binary_reward,
            reward_noise=reward_noise, reward_noise_mode=reward_noise_mode)
        if not 1 <= score_len <= seq_len:
            raise ValueError('score_len must be in [1, seq_len]')
        self.score_len = score_len

    def reward_from_actions(self, actions, targets):
        scored = actions[:, -self.score_len:] == targets[:, -self.score_len:]
        if self.binary_reward:
            return scored.all(dim=1).float()
        return scored.float().mean(dim=1)

    def token_candidate_rewards(self, candidate_actions, targets, step, alive):
        scored = step >= self.seq_len - self.score_len
        if not scored:
            rewards = torch.zeros_like(candidate_actions, dtype=torch.float)
            return rewards, alive
        correct = (candidate_actions == targets[:, step:step + 1]).float()
        if self.binary_reward:
            rewards = alive[:, None] * correct
            return rewards, rewards[:, 0]
        return correct, alive

    def sample_batch(self, model, batch_size, device, group_size=1):
        batch = super().sample_batch(model, batch_size, device, group_size)
        score_mask = torch.zeros_like(batch.actions, dtype=torch.bool)
        score_mask[:, -self.score_len:] = True
        # actor_expected_reward from parent is full-sequence P(success), but
        # masked reward is suffix-only. The exact suffix-marginal is not
        # cheaply available, so we explicitly drop it and fall back to the
        # collision baseline.
        return Batch(
            obs=batch.obs, actions=batch.actions, rewards=batch.rewards,
            labels=batch.labels, actor_logp_a=batch.actor_logp_a,
            actor_baseline=batch.actor_baseline, group_ids=batch.group_ids,
            score_mask=score_mask, actor_expected_reward=None,
            informative_group_rate=batch.informative_group_rate,
            retained_group_rate=batch.retained_group_rate,
            used_group_fallback=batch.used_group_fallback,
            age=batch.age)

    @torch.no_grad()
    def evaluate(self, model, device, num_batches=10, batch_size=100):
        H, M, S = self.seq_len, self.vocab_size, self.score_len
        scored_correct, scored_total = 0, 0
        scored_exact, scored_seqs = 0, 0
        unscored_correct, unscored_total = 0, 0
        total_entropy, total_ent_tokens = 0.0, 0
        was_training = model.training
        model.eval()
        for _ in range(num_batches):
            input_tokens = torch.randint(M, (batch_size, H), device=device)
            generated, target_tokens, entropy, ent_tokens = self._greedy_generate(
                model, input_tokens)
            total_entropy += entropy
            total_ent_tokens += ent_tokens
            correct = (generated == target_tokens).float()
            scored_correct += correct[:, -S:].sum().item()
            scored_total += batch_size * S
            scored_exact += correct[:, -S:].all(dim=1).float().sum().item()
            scored_seqs += batch_size
            if S < H:
                unscored_correct += correct[:, :-S].sum().item()
                unscored_total += batch_size * (H - S)
        if self.binary_reward:
            result = {'test_error': 1.0 - scored_exact / scored_seqs}
        else:
            result = {'test_error': 1.0 - scored_correct / scored_total}
        result['entropy'] = total_entropy / total_ent_tokens
        if unscored_total > 0:
            result['test_error_unscored'] = 1.0 - unscored_correct / unscored_total
        model.train(was_training)
        return result


class RewardChainReversal(TokenReversal):
    """Reversal with ordered checkpoint rewards.

    The scalar reward averages three verifiable checks: first-half exactness,
    second-half exactness, and full-response exactness. This gives a toy
    reward-chain setting between final-dot reward and dense token supervision.
    """

    def _chain_reward(self, actions, targets):
        H = actions.size(1)
        mid = H // 2
        first = (actions[:, :mid] == targets[:, :mid]).all(dim=1).float()
        second = (actions[:, mid:] == targets[:, mid:]).all(dim=1).float()
        final = (actions == targets).all(dim=1).float()
        return (first + second + final) / 3.0

    def reward_from_actions(self, actions, targets):
        return self._chain_reward(actions, targets)

    @torch.no_grad()
    def evaluate(self, model, device, num_batches=10, batch_size=100):
        H, M = self.seq_len, self.vocab_size
        total_exact, total_seqs, total_chain = 0, 0, 0.0
        total_entropy, total_ent_tokens = 0.0, 0
        was_training = model.training
        model.eval()
        for _ in range(num_batches):
            input_tokens = torch.randint(M, (batch_size, H), device=device)
            generated, target_tokens, entropy, ent_tokens = self._greedy_generate(
                model, input_tokens)
            total_entropy += entropy
            total_ent_tokens += ent_tokens
            total_exact += (generated == target_tokens).all(dim=1).float().sum().item()
            total_chain += self._chain_reward(generated, target_tokens).sum().item()
            total_seqs += batch_size
        model.train(was_training)
        return {
            'test_error': 1.0 - total_exact / total_seqs,
            'chain_reward': total_chain / total_seqs,
            'entropy': total_entropy / total_ent_tokens,
        }


class ChainArithmetic(RewardChainReversal):
    """Copy operands, then emit a modular checksum with chain rewards."""

    def __init__(self, vocab_size: int, seq_len: int, binary_reward: bool = False,
                 reward_noise: float = 0.0, reward_noise_mode: str = 'none'):
        if seq_len < 2:
            raise ValueError('chain_arithmetic requires seq_len >= 2')
        super().__init__(
            vocab_size, seq_len, binary_reward=binary_reward,
            reward_noise=reward_noise, reward_noise_mode=reward_noise_mode)

    def target_tokens(self, input_tokens):
        targets = input_tokens.clone()
        targets[:, -1] = input_tokens[:, :-1].sum(dim=1) % self.vocab_size
        return targets

    def _chain_reward(self, actions, targets):
        copied = (actions[:, :-1] == targets[:, :-1]).all(dim=1).float()
        answer = (actions[:, -1] == targets[:, -1]).float()
        final = (actions == targets).all(dim=1).float()
        return (copied + answer + final) / 3.0


class FormatAnswerArithmetic(RewardChainReversal):
    """Emit an answer tag, then a modular sum, with format and answer checks."""

    def __init__(self, vocab_size: int, seq_len: int, binary_reward: bool = False,
                 reward_noise: float = 0.0, reward_noise_mode: str = 'none'):
        if seq_len < 2:
            raise ValueError('format_answer requires seq_len >= 2')
        super().__init__(
            vocab_size, seq_len, binary_reward=binary_reward,
            reward_noise=reward_noise, reward_noise_mode=reward_noise_mode)
        self.sep_token = vocab_size
        self.answer_token = vocab_size + 1
        self.effective_vocab = vocab_size + 2
        self.num_actions = self.effective_vocab

    def target_tokens(self, input_tokens):
        targets = torch.zeros_like(input_tokens)
        targets[:, 0] = self.answer_token
        targets[:, 1] = (input_tokens[:, 0] + input_tokens[:, 1]) % self.vocab_size
        if self.seq_len > 2:
            targets[:, 2:] = input_tokens[:, 2:]
        return targets

    def _chain_reward(self, actions, targets):
        format_ok = (actions[:, 0] == targets[:, 0]).float()
        answer_ok = (actions[:, 1] == targets[:, 1]).float()
        final = (actions == targets).all(dim=1).float()
        return (format_ok + answer_ok + final) / 3.0


# -- LM Bandit ---------------------------------------------------------------


class CausalLMWrapper(nn.Module):
    """Wraps a HuggingFace CausalLM. forward(input_ids) -> logits [B, T, V]."""
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, input_ids):
        return self.hf_model(input_ids=input_ids).logits


class LMBandit:
    """Next-token prediction as a contextual bandit with a pre-trained LM."""

    def __init__(self, model_name: str, context_len: int = 128, kl_weight: float = 0.0,
                 max_eval_contexts: int = 500, reward_noise: float = 0.0,
                 reward_noise_mode: str = 'none'):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.context_len = context_len
        self.kl_weight = kl_weight
        self.max_eval_contexts = max_eval_contexts
        self.reward_noise = reward_noise
        self.reward_noise_mode = reward_noise_mode

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = tokenizer.vocab_size
        self.num_actions = self.vocab_size

        raw = load_dataset('wikitext', 'wikitext-2-raw-v1')
        self.train_tokens = self._tokenize_split(raw['train'], tokenizer)
        self.test_tokens = self._tokenize_split(raw['test'], tokenizer)

        self._test_difficulty = None
        self.ref_model = None

        print(f'LMBandit: {model_name}, vocab={self.vocab_size}, '
              f'train={len(self.train_tokens)}, test={len(self.test_tokens)}, '
              f'context_len={context_len}, kl_weight={kl_weight}')

    @staticmethod
    def _tokenize_split(split, tokenizer):
        # Encode incrementally with newline separators so the tokenizer sees
        # natural paragraph boundaries without building one giant string
        chunks = []
        for t in split['text']:
            if t.strip():
                prefix = '\n' if chunks else ''
                chunks.append(torch.tensor(
                    tokenizer.encode(prefix + t), dtype=torch.long))
        return torch.cat(chunks)

    def make_model(self) -> nn.Module:
        from transformers import AutoModelForCausalLM
        # FP32 params for stable Adam; BF16 compute via autocast
        hf_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        wrapper = CausalLMWrapper(hf_model)
        if self.kl_weight > 0:
            ref_hf = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=torch.bfloat16)
            self.ref_model = CausalLMWrapper(ref_hf)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        return wrapper

    def _ref_log_probs(self, contexts, device):
        """Reference model log-probs on contexts. Returns [N, V] or None."""
        if self.kl_weight <= 0 or self.ref_model is None:
            return None
        with torch.no_grad():
            self.ref_model.to(device)
            return F.log_softmax(self.ref_model(contexts)[:, -1, :].float(), dim=-1)

    def _compute_rewards(self, actions, labels, actor_lp, ref_lp):
        """Compute rewards with optional KL penalty. Handles [B] and [N, K] actions."""
        base = (actions == (labels if actions.dim() == 1 else labels.unsqueeze(1))).float()
        if ref_lp is None:
            return base
        # reshape to 2D for gather: [N, 1] or [N, K]
        idx = actions.reshape(actions.shape[0], -1)
        kl = actor_lp.gather(-1, idx) - ref_lp.gather(-1, idx)
        return base - self.kl_weight * kl.reshape_as(base)

    def _compute_expected_reward(self, actor_probs, actor_lp, labels, ref_lp):
        """Exact E[R|x] under the actor. Returns [N] tensor.

        For kl_weight=0: pi(label|x).
        For kl_weight>0: pi(label|x) - beta * KL(pi || ref).
        """
        p_label = actor_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        if ref_lp is None:
            return p_label
        kl = (actor_probs * (actor_lp - ref_lp)).sum(-1)
        return p_label - self.kl_weight * kl

    def sample_batch(self, model: nn.Module, batch_size: int,
                     device: torch.device, group_size: int = 1) -> Batch:
        if group_size > 1:
            return self._sample_grouped(model, batch_size, group_size, device)

        C = self.context_len
        starts = torch.randint(0, len(self.train_tokens) - C - 1, (batch_size,))
        contexts = torch.stack([self.train_tokens[s:s + C] for s in starts]).to(device)
        labels = torch.stack([self.train_tokens[s + C] for s in starts]).to(device)

        was_training = model.training
        model.eval()
        with torch.no_grad(), torch.amp.autocast(
                'cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            actor_logits = model(contexts)[:, -1, :]
            actor_lp = F.log_softmax(actor_logits.float(), dim=-1)
            actor_probs = actor_lp.exp()
            actions = torch.distributions.Categorical(probs=actor_probs).sample()
            logp_a = actor_lp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            baseline = (actor_probs ** 2).sum(-1)
        model.train(was_training)

        ref_lp = self._ref_log_probs(contexts, device)
        rewards = self._compute_rewards(actions, labels, actor_lp, ref_lp)
        rewards = corrupt_rewards(
            rewards, actions, labels, self.reward_noise, self.reward_noise_mode)
        expected_reward = self._compute_expected_reward(
            actor_probs, actor_lp, labels, ref_lp)
        if self.reward_noise > 0:
            expected_reward = None
        return Batch(obs=contexts, actions=actions, rewards=rewards,
                     labels=labels, actor_logp_a=logp_a, actor_baseline=baseline,
                     actor_expected_reward=expected_reward)

    def _sample_grouped(self, model, batch_size, group_size, device) -> Batch:
        """K rollouts per context for grouped methods."""
        C = self.context_len
        num_contexts = batch_size // group_size
        starts = torch.randint(0, len(self.train_tokens) - C - 1, (num_contexts,))
        contexts = torch.stack([self.train_tokens[s:s + C] for s in starts]).to(device)
        labels = torch.stack([self.train_tokens[s + C] for s in starts]).to(device)

        was_training = model.training
        model.eval()
        with torch.no_grad(), torch.amp.autocast(
                'cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            actor_logits = model(contexts)[:, -1, :]
            actor_lp = F.log_softmax(actor_logits.float(), dim=-1)
            actor_probs = actor_lp.exp()
            actor_bl = (actor_probs ** 2).sum(-1)  # [N]
            actions = torch.stack([
                torch.distributions.Categorical(probs=actor_probs).sample()
                for _ in range(group_size)], dim=1)  # [N, K]
        model.train(was_training)

        # Filter on raw correctness BEFORE applying KL penalty
        raw_correct = (actions == labels.unsqueeze(1)).float()
        raw_keep = (raw_correct.sum(1) > 0) & (raw_correct.sum(1) < group_size)
        informative_rate = raw_keep.float().mean().item()
        fallback = raw_keep.sum() == 0
        if fallback:
            raw_keep[:] = True
        ref_lp = self._ref_log_probs(contexts, device)
        rewards = self._compute_rewards(actions, labels, actor_lp, ref_lp)
        rewards = corrupt_rewards(
            rewards, actions, labels, self.reward_noise, self.reward_noise_mode)

        K = group_size
        n_valid = raw_keep.sum().item()

        contexts_f = contexts[raw_keep].unsqueeze(1).expand(-1, K, -1).reshape(-1, C)
        labels_f = labels[raw_keep].unsqueeze(1).expand(-1, K).reshape(-1)
        actions_f = actions[raw_keep].reshape(-1)
        rewards_f = rewards[raw_keep].reshape(-1)
        logp_a_f = actor_lp[raw_keep].gather(-1, actions[raw_keep]).reshape(-1)
        baseline_f = actor_bl[raw_keep].unsqueeze(1).expand(-1, K).reshape(-1)
        group_ids = torch.arange(n_valid, device=device).unsqueeze(1).expand(-1, K).reshape(-1)

        expected_reward = self._compute_expected_reward(
            actor_probs, actor_lp, labels, ref_lp)
        expected_reward_f = None
        if self.reward_noise <= 0:
            expected_reward_f = expected_reward[raw_keep].unsqueeze(1).expand(-1, K).reshape(-1)

        return Batch(obs=contexts_f, actions=actions_f, rewards=rewards_f,
                     labels=labels_f, actor_logp_a=logp_a_f,
                     actor_baseline=baseline_f, group_ids=group_ids,
                     actor_expected_reward=expected_reward_f,
                     informative_group_rate=informative_rate,
                     retained_group_rate=n_valid / num_contexts,
                     used_group_fallback=fallback.item())

    def compute_logits(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        return model(batch.obs)[:, -1, :]

    def compute_logits_oracle(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        return model(batch.obs)[:, -1, :]  # same as compute_logits for bandits

    def compute_difficulty(self, model: nn.Module, device: torch.device,
                           batch_size: int = 16):
        """Compute difficulty terciles from model's current state.

        Call once per seed BEFORE training. Resets cached difficulty.
        """
        self._test_difficulty = None
        C = self.context_len
        n_eval = min((len(self.test_tokens) - C - 1) // C, self.max_eval_contexts)
        starts = torch.arange(n_eval) * C

        losses = []
        was_training = model.training
        model.eval()
        with torch.no_grad(), torch.amp.autocast(
                'cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            for i in range(0, len(starts), batch_size):
                bs = starts[i:i + batch_size]
                ctx = torch.stack([self.test_tokens[s:s + C] for s in bs]).to(device)
                lab = torch.stack([self.test_tokens[s + C] for s in bs]).to(device)
                lp = F.log_softmax(model(ctx)[:, -1, :].float(), dim=-1)
                losses.extend((-lp.gather(1, lab.unsqueeze(1)).squeeze(1)).tolist())
        model.train(was_training)

        n = len(losses)
        ranked = sorted(range(n), key=lambda i: losses[i])
        self._test_difficulty = torch.zeros(n, dtype=torch.long)
        for rank, idx in enumerate(ranked):
            if rank < n // 3:
                self._test_difficulty[idx] = 0
            elif rank < 2 * n // 3:
                self._test_difficulty[idx] = 1
            else:
                self._test_difficulty[idx] = 2

    @torch.no_grad()
    def evaluate(self, model: nn.Module, device: torch.device,
                 batch_size: int = 16) -> dict[str, float]:
        C = self.context_len
        n_eval = min((len(self.test_tokens) - C - 1) // C, self.max_eval_contexts)
        starts = torch.arange(n_eval) * C
        all_correct, all_log_prob, all_entropy = [], [], []

        was_training = model.training
        model.eval()
        with torch.amp.autocast(
                'cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            for i in range(0, len(starts), batch_size):
                bs = starts[i:i + batch_size]
                ctx = torch.stack([self.test_tokens[s:s + C] for s in bs]).to(device)
                lab = torch.stack([self.test_tokens[s + C] for s in bs]).to(device)
                logits = model(ctx)[:, -1, :]
                all_correct.extend((logits.argmax(-1) == lab).tolist())
                lp = F.log_softmax(logits.float(), dim=-1)
                all_log_prob.extend(lp.gather(1, lab.unsqueeze(1)).squeeze(1).tolist())
                all_entropy.extend((-(lp.exp() * lp).sum(dim=-1)).tolist())
        model.train(was_training)

        n = len(all_correct)
        # Perplexity over non-overlapping windows (one token per window).
        # Consistent across methods but not comparable to published full-sequence perplexity.
        result = {
            'test_error': 1.0 - sum(all_correct) / n,
            'perplexity': math.exp(-sum(all_log_prob) / n),
            'entropy': sum(all_entropy) / n,
        }

        if self._test_difficulty is not None:
            diff = self._test_difficulty[:n]
            for level, name in [(0, 'easy'), (1, 'medium'), (2, 'hard')]:
                mask = (diff == level)
                if mask.sum() > 0:
                    lc = [all_correct[i] for i in range(n) if mask[i]]
                    result[f'error_{name}'] = 1.0 - sum(lc) / len(lc)

        return result
