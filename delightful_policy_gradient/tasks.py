"""Tasks: MNISTBandit, TokenReversal, and LMBandit.

Each task provides:
  - sample_batch(model, batch_size, device, group_size=1) → Batch
      Uses model directly as actor in eval() mode. No stale params —
      staleness is handled by the experience queue in the training loop.
  - compute_logits(model, batch) → Tensor
      Learner forward pass. Logits aligned with batch.actions.
  - compute_logits_oracle(model, batch) → Tensor
      For CE: conditions on ground truth, not actor-generated sequence.
      Same as compute_logits for bandit tasks. Different for sequential tasks.
  - evaluate(model, device) → dict
  - make_model() → nn.Module
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets


@dataclass
class Batch:
    """A batch of bandit/RL experience.

    Stores sufficient statistics instead of full actor distributions:
      actor_logp_a: [B] or [B, T] — log-prob of taken action under actor policy
      actor_baseline: [B] — Σπ(a)² expected baseline under actor policy
    This scales with batch size, not vocab size (75,000× smaller for Qwen).
    """
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    labels: torch.Tensor
    actor_logp_a: torch.Tensor   # [B] or [B, T]
    actor_baseline: torch.Tensor  # [B]
    group_ids: torch.Tensor | None = None
    score_mask: torch.Tensor | None = None  # [B, T] bool: which positions are scored
    actor_expected_reward: torch.Tensor | None = None  # [B]: exact E[R|x] under actor
    # Sampler-side group diagnostics (plain scalars, not tensors)
    informative_group_rate: float | None = None  # pre-filter mixed-group rate
    retained_group_rate: float | None = None     # post-filter fraction kept
    used_group_fallback: bool = False            # True if empty-batch safeguard fired

    def to(self, device) -> 'Batch':
        return Batch(
            obs=self.obs.to(device), actions=self.actions.to(device),
            rewards=self.rewards.to(device), labels=self.labels.to(device),
            actor_logp_a=self.actor_logp_a.to(device),
            actor_baseline=self.actor_baseline.to(device),
            group_ids=self.group_ids.to(device) if self.group_ids is not None else None,
            score_mask=self.score_mask.to(device) if self.score_mask is not None else None,
            actor_expected_reward=self.actor_expected_reward.to(device) if self.actor_expected_reward is not None else None,
            informative_group_rate=self.informative_group_rate,
            retained_group_rate=self.retained_group_rate,
            used_group_fallback=self.used_group_fallback)

    def select(self, mask: torch.Tensor) -> 'Batch':
        return Batch(
            obs=self.obs[mask], actions=self.actions[mask],
            rewards=self.rewards[mask], labels=self.labels[mask],
            actor_logp_a=self.actor_logp_a[mask],
            actor_baseline=self.actor_baseline[mask],
            group_ids=self.group_ids[mask] if self.group_ids is not None else None,
            score_mask=self.score_mask[mask] if self.score_mask is not None else None,
            actor_expected_reward=self.actor_expected_reward[mask] if self.actor_expected_reward is not None else None,
            informative_group_rate=self.informative_group_rate,
            retained_group_rate=self.retained_group_rate,
            used_group_fallback=self.used_group_fallback)


# ── MNIST Contextual Bandit ──────────────────────────────────────────────────


class MNISTBandit:
    """MNIST as a one-step contextual bandit."""
    num_actions = 10


    def __init__(self, data_dir: str = './data'):
        train = datasets.MNIST(data_dir, train=True, download=True)
        test = datasets.MNIST(data_dir, train=False, download=True)
        self.train_images = train.data.float().reshape(-1, 784) / 255.0
        self.train_labels = train.targets
        self.test_images = test.data.float().reshape(-1, 784) / 255.0
        self.test_labels = test.targets

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

        model.eval()
        with torch.no_grad():
            actor_logits = model(images)
            actor_lp = F.log_softmax(actor_logits, dim=-1)
            actor_probs = F.softmax(actor_logits, dim=-1)
            actions = torch.distributions.Categorical(logits=actor_logits).sample()
            logp_a = actor_lp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            baseline = (actor_probs ** 2).sum(-1)
            p_success = actor_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        model.train()

        return Batch(obs=images, actions=actions,
                     rewards=(actions == labels).float(),
                     labels=labels, actor_logp_a=logp_a, actor_baseline=baseline,
                     actor_expected_reward=p_success)

    def _sample_grouped(self, model, batch_size, group_size, device) -> Batch:
        """K actions per context for DAPO. Returns flattened batch with group_ids."""
        num_contexts = batch_size // group_size
        idx = torch.randint(len(self.train_images), (num_contexts,))
        images = self.train_images[idx].to(device)
        labels = self.train_labels[idx].to(device)

        model.eval()
        with torch.no_grad():
            actor_logits = model(images)
            actor_lp = F.log_softmax(actor_logits, dim=-1)
            actor_bl = (F.softmax(actor_logits, dim=-1) ** 2).sum(-1)  # [N]
            actions = torch.stack([
                torch.distributions.Categorical(logits=actor_logits).sample()
                for _ in range(group_size)], dim=1)  # [N, K]
        model.train()

        rewards = (actions == labels.unsqueeze(1)).float()  # [N, K]

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
        # Per-action log-probs: actor_lp [N, V], actions [N, K] → gather gives [N, K]
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
        pass  # no difficulty tracking for MNIST

    @torch.no_grad()
    def evaluate(self, model: nn.Module, device: torch.device) -> dict[str, float]:
        model.eval()
        logits = model(self.test_images.to(device))
        acc = (logits.argmax(-1) == self.test_labels.to(device)).float().mean().item()
        return {'test_error': 1.0 - acc}


# ── Token Reversal ───────────────────────────────────────────────────────────


class TokenReversal:
    """Token reversal task from DG paper Section 5."""


    def __init__(self, vocab_size: int, seq_len: int, binary_reward: bool = False):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.binary_reward = binary_reward
        self.effective_vocab = vocab_size + 1
        self.sep_token = vocab_size
        self.num_actions = self.effective_vocab

    def make_model(self, d_model: int, nhead: int, num_layers: int) -> nn.Module:
        from .models import CausalTransformer
        return CausalTransformer(
            vocab_size=self.effective_vocab, d_model=d_model,
            nhead=nhead, num_layers=num_layers,
            max_seq_len=self.seq_len * 2 + 1)

    def _rollout(self, model, input_tokens, device):
        """Single autoregressive rollout. Returns sufficient stats, not full distributions."""
        H = self.seq_len
        B = input_tokens.shape[0]
        target_tokens = input_tokens.flip(1)
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
        baseline = torch.stack(per_token_baseline, dim=1).mean(dim=1)  # [B]
        correct = (actions == target_tokens).float()
        if self.binary_reward:
            rewards = correct.all(dim=1).float()  # exact sequence match
        else:
            rewards = correct.mean(dim=1)  # fraction correct
        obs = torch.cat([input_tokens, sep, actions], dim=1)
        return actions, logp_a, baseline, rewards, obs, target_tokens

    def sample_batch(self, model: nn.Module, batch_size: int,
                     device: torch.device, group_size: int = 1) -> Batch:
        H, M = self.seq_len, self.vocab_size
        model.eval()

        if group_size == 1:
            input_tokens = torch.randint(M, (batch_size, H), device=device)
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

            model.train()
            return Batch(obs=obs, actions=actions, rewards=rewards,
                         labels=labels, actor_logp_a=logp_a, actor_baseline=baseline,
                         actor_expected_reward=p_success)

        # K rollouts per input — how DAPO/GRPO work in practice
        num_contexts = batch_size // group_size
        input_tokens = torch.randint(M, (num_contexts, H), device=device)

        all_a, all_logp, all_bl, all_r, all_obs, all_lab = [], [], [], [], [], []
        for _ in range(group_size):
            a, lp, bl, r, o, l = self._rollout(model, input_tokens, device)
            all_a.append(a); all_logp.append(lp); all_bl.append(bl)
            all_r.append(r); all_obs.append(o); all_lab.append(l)
        model.train()

        actions = torch.stack(all_a, dim=1)      # [N, K, H]
        logp_a = torch.stack(all_logp, dim=1)    # [N, K, H]
        baselines = torch.stack(all_bl, dim=1)   # [N, K]
        rewards = torch.stack(all_r, dim=1)       # [N, K]
        obs = torch.stack(all_obs, dim=1)         # [N, K, 2H+1]
        labels = torch.stack(all_lab, dim=1)      # [N, K, H]

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
            actor_baseline=baselines[raw_keep].reshape(-1),
            group_ids=group_ids,
            informative_group_rate=informative_rate,
            retained_group_rate=n_valid / num_contexts,
            used_group_fallback=fallback.item())

    def compute_logits(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        """Teacher-forced on ACTOR-GENERATED prefix → logits at output positions."""
        logits_full = model(batch.obs[:, :-1])
        return logits_full[:, self.seq_len:, :]

    def compute_difficulty(self, model: nn.Module, device: torch.device):
        pass  # no difficulty tracking for token reversal

    def compute_logits_oracle(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        """Teacher-forced on GROUND TRUTH prefix → true supervised oracle.

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
    def evaluate(self, model: nn.Module, device: torch.device,
                 num_batches: int = 10, batch_size: int = 100) -> dict[str, float]:
        H, M = self.seq_len, self.vocab_size
        total_correct, total_tokens, total_exact, total_seqs = 0, 0, 0, 0
        model.eval()
        for _ in range(num_batches):
            input_tokens = torch.randint(M, (batch_size, H), device=device)
            target_tokens = input_tokens.flip(1)
            sep = torch.full((batch_size, 1), self.sep_token, device=device, dtype=torch.long)
            prefix = torch.cat([input_tokens, sep], dim=1)
            for _ in range(H):
                next_token = model(prefix)[:, -1].argmax(dim=-1)
                prefix = torch.cat([prefix, next_token.unsqueeze(1)], dim=1)
            generated = prefix[:, H + 1:]
            total_correct += (generated == target_tokens).float().sum().item()
            total_tokens += batch_size * H
            total_exact += (generated == target_tokens).all(dim=1).float().sum().item()
            total_seqs += batch_size
        if self.binary_reward:
            return {'test_error': 1.0 - total_exact / total_seqs}
        return {'test_error': 1.0 - total_correct / total_tokens}


# ── Masked Reversal ─────────────────────────────────────────────────────────


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
                 binary_reward: bool = False):
        super().__init__(vocab_size, seq_len, binary_reward=binary_reward)
        assert 1 <= score_len <= seq_len
        self.score_len = score_len

    def _rollout(self, model, input_tokens, device):
        actions, logp_a, baseline, _, obs, targets = super()._rollout(
            model, input_tokens, device)
        scored = actions[:, -self.score_len:] == targets[:, -self.score_len:]
        if self.binary_reward:
            rewards = scored.all(dim=1).float()  # all scored positions correct
        else:
            rewards = scored.float().mean(dim=1)
        return actions, logp_a, baseline, rewards, obs, targets

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
            used_group_fallback=batch.used_group_fallback)

    @torch.no_grad()
    def evaluate(self, model, device, num_batches=10, batch_size=100):
        H, M, S = self.seq_len, self.vocab_size, self.score_len
        scored_correct, scored_total = 0, 0
        scored_exact, scored_seqs = 0, 0
        unscored_correct, unscored_total = 0, 0
        model.eval()
        for _ in range(num_batches):
            input_tokens = torch.randint(M, (batch_size, H), device=device)
            target_tokens = input_tokens.flip(1)
            sep = torch.full((batch_size, 1), self.sep_token, device=device, dtype=torch.long)
            prefix = torch.cat([input_tokens, sep], dim=1)
            for _ in range(H):
                next_token = model(prefix)[:, -1].argmax(dim=-1)
                prefix = torch.cat([prefix, next_token.unsqueeze(1)], dim=1)
            generated = prefix[:, H + 1:]
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
        if unscored_total > 0:
            result['test_error_unscored'] = 1.0 - unscored_correct / unscored_total
        return result


# ── LM Bandit ────────────────────────────────────────────────────────────────


class CausalLMWrapper(nn.Module):
    """Wraps a HuggingFace CausalLM. forward(input_ids) → logits [B, T, V]."""
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, input_ids):
        return self.hf_model(input_ids=input_ids).logits


class LMBandit:
    """Next-token prediction as a contextual bandit with a pre-trained LM."""


    def __init__(self, model_name: str, context_len: int = 128, kl_weight: float = 0.0,
                 max_eval_contexts: int = 500):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.context_len = context_len
        self.kl_weight = kl_weight
        self.max_eval_contexts = max_eval_contexts

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
        chunks = [torch.tensor(tokenizer.encode(t), dtype=torch.long)
                  for t in split['text'] if t.strip()]
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

    def _compute_rewards(self, actions, labels, actor_lp, contexts, device):
        """Compute rewards with optional KL penalty. Handles [B] and [N, K] actions."""
        base = (actions == (labels if actions.dim() == 1 else labels.unsqueeze(1))).float()
        if self.kl_weight <= 0 or self.ref_model is None:
            return base
        with torch.no_grad():
            self.ref_model.to(device)
            ref_lp = F.log_softmax(self.ref_model(contexts)[:, -1, :].float(), dim=-1)
            # reshape to 2D for gather: [N, 1] or [N, K]
            idx = actions.reshape(actions.shape[0], -1)
            kl = actor_lp.gather(-1, idx) - ref_lp.gather(-1, idx)
            return base - self.kl_weight * kl.reshape_as(base)

    def sample_batch(self, model: nn.Module, batch_size: int,
                     device: torch.device, group_size: int = 1) -> Batch:
        if group_size > 1:
            return self._sample_grouped(model, batch_size, group_size, device)

        C = self.context_len
        starts = torch.randint(0, len(self.train_tokens) - C - 1, (batch_size,))
        contexts = torch.stack([self.train_tokens[s:s + C] for s in starts]).to(device)
        labels = torch.stack([self.train_tokens[s + C] for s in starts]).to(device)

        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            actor_logits = model(contexts)[:, -1, :]
            actor_probs = F.softmax(actor_logits.float(), dim=-1)
            actor_lp = torch.log(actor_probs)
            actions = torch.distributions.Categorical(probs=actor_probs).sample()
            logp_a = actor_lp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            baseline = (actor_probs ** 2).sum(-1)
        model.train()

        rewards = self._compute_rewards(actions, labels, actor_lp, contexts, device)
        p_success = actor_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1) if self.kl_weight <= 0 else None
        return Batch(obs=contexts, actions=actions, rewards=rewards,
                     labels=labels, actor_logp_a=logp_a, actor_baseline=baseline,
                     actor_expected_reward=p_success)

    def _sample_grouped(self, model, batch_size, group_size, device) -> Batch:
        """K actions per context for DAPO. Same pattern as MNISTBandit._sample_grouped."""
        C = self.context_len
        num_contexts = batch_size // group_size
        starts = torch.randint(0, len(self.train_tokens) - C - 1, (num_contexts,))
        contexts = torch.stack([self.train_tokens[s:s + C] for s in starts]).to(device)
        labels = torch.stack([self.train_tokens[s + C] for s in starts]).to(device)

        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            actor_logits = model(contexts)[:, -1, :]
            actor_probs = F.softmax(actor_logits.float(), dim=-1)
            actor_lp = torch.log(actor_probs)
            actor_bl = (actor_probs ** 2).sum(-1)  # [N]
            actions = torch.stack([
                torch.distributions.Categorical(probs=actor_probs).sample()
                for _ in range(group_size)], dim=1)  # [N, K]
        model.train()

        # Filter on raw correctness BEFORE applying KL penalty
        raw_correct = (actions == labels.unsqueeze(1)).float()
        raw_keep = (raw_correct.sum(1) > 0) & (raw_correct.sum(1) < group_size)
        informative_rate = raw_keep.float().mean().item()
        fallback = raw_keep.sum() == 0
        if fallback:
            raw_keep[:] = True
        rewards = self._compute_rewards(actions, labels, actor_lp, contexts, device)

        K = group_size
        n_valid = raw_keep.sum().item()

        contexts_f = contexts[raw_keep].unsqueeze(1).expand(-1, K, -1).reshape(-1, C)
        labels_f = labels[raw_keep].unsqueeze(1).expand(-1, K).reshape(-1)
        actions_f = actions[raw_keep].reshape(-1)
        rewards_f = rewards[raw_keep].reshape(-1)
        logp_a_f = actor_lp[raw_keep].gather(-1, actions[raw_keep]).reshape(-1)
        baseline_f = actor_bl[raw_keep].unsqueeze(1).expand(-1, K).reshape(-1)
        group_ids = torch.arange(n_valid, device=device).unsqueeze(1).expand(-1, K).reshape(-1)

        return Batch(obs=contexts_f, actions=actions_f, rewards=rewards_f,
                     labels=labels_f, actor_logp_a=logp_a_f,
                     actor_baseline=baseline_f, group_ids=group_ids,
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
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, len(starts), batch_size):
                bs = starts[i:i + batch_size]
                ctx = torch.stack([self.test_tokens[s:s + C] for s in bs]).to(device)
                lab = torch.stack([self.test_tokens[s + C] for s in bs]).to(device)
                lp = F.log_softmax(model(ctx)[:, -1, :].float(), dim=-1)
                losses.extend((-lp.gather(1, lab.unsqueeze(1)).squeeze(1)).tolist())

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
        all_correct, all_log_prob = [], []

        model.eval()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, len(starts), batch_size):
                bs = starts[i:i + batch_size]
                ctx = torch.stack([self.test_tokens[s:s + C] for s in bs]).to(device)
                lab = torch.stack([self.test_tokens[s + C] for s in bs]).to(device)
                logits = model(ctx)[:, -1, :]
                all_correct.extend((logits.argmax(-1) == lab).tolist())
                lp = F.log_softmax(logits.float(), dim=-1)
                all_log_prob.extend(lp.gather(1, lab.unsqueeze(1)).squeeze(1).tolist())

        n = len(all_correct)
        # Perplexity over non-overlapping windows (one token per window).
        # Consistent across methods but not comparable to published full-sequence perplexity.
        result = {
            'test_error': 1.0 - sum(all_correct) / n,
            'perplexity': math.exp(-sum(all_log_prob) / n),
        }

        if self._test_difficulty is not None:
            diff = self._test_difficulty[:n]
            for level, name in [(0, 'easy'), (1, 'medium'), (2, 'hard')]:
                mask = (diff == level)
                if mask.sum() > 0:
                    lc = [all_correct[i] for i in range(n) if mask[i]]
                    result[f'error_{name}'] = 1.0 - sum(lc) / len(lc)

        return result
