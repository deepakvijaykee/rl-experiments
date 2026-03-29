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

    def to(self, device) -> 'Batch':
        return Batch(
            obs=self.obs.to(device), actions=self.actions.to(device),
            rewards=self.rewards.to(device), labels=self.labels.to(device),
            actor_logp_a=self.actor_logp_a.to(device),
            actor_baseline=self.actor_baseline.to(device),
            group_ids=self.group_ids.to(device) if self.group_ids is not None else None)

    def select(self, mask: torch.Tensor) -> 'Batch':
        return Batch(
            obs=self.obs[mask], actions=self.actions[mask],
            rewards=self.rewards[mask], labels=self.labels[mask],
            actor_logp_a=self.actor_logp_a[mask],
            actor_baseline=self.actor_baseline[mask],
            group_ids=self.group_ids[mask] if self.group_ids is not None else None)


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
            actions = torch.distributions.Categorical(logits=actor_logits).sample()
            logp_a = actor_lp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            baseline = (F.softmax(actor_logits, dim=-1) ** 2).sum(-1)
        model.train()

        return Batch(obs=images, actions=actions,
                     rewards=(actions == labels).float(),
                     labels=labels, actor_logp_a=logp_a, actor_baseline=baseline)

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

        has_mix = (rewards.sum(1) > 0) & (rewards.sum(1) < group_size)
        if has_mix.sum() == 0:
            has_mix[:] = True

        n_valid = has_mix.sum().item()
        K = group_size

        images_f = images[has_mix].unsqueeze(1).expand(-1, K, -1).reshape(-1, 784)
        labels_f = labels[has_mix].unsqueeze(1).expand(-1, K).reshape(-1)
        actions_f = actions[has_mix].reshape(-1)
        rewards_f = rewards[has_mix].reshape(-1)
        # Per-action log-probs: actor_lp [N, V], actions [N, K] → gather gives [N, K]
        logp_a_f = actor_lp[has_mix].gather(-1, actions[has_mix]).reshape(-1)
        baseline_f = actor_bl[has_mix].unsqueeze(1).expand(-1, K).reshape(-1)
        group_ids = torch.arange(n_valid, device=device).unsqueeze(1).expand(-1, K).reshape(-1)

        return Batch(obs=images_f, actions=actions_f, rewards=rewards_f,
                     labels=labels_f, actor_logp_a=logp_a_f,
                     actor_baseline=baseline_f, group_ids=group_ids)

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


    def __init__(self, vocab_size: int, seq_len: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
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
        rewards = (actions == target_tokens).float().mean(dim=1)
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
            model.train()
            return Batch(obs=obs, actions=actions, rewards=rewards,
                         labels=labels, actor_logp_a=logp_a, actor_baseline=baseline)

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
        has_var = rewards.std(1) > 1e-6
        if has_var.sum() == 0:
            has_var[:] = True

        n_valid = has_var.sum().item()
        K = group_size
        group_ids = torch.arange(n_valid, device=device).unsqueeze(1).expand(-1, K).reshape(-1)

        return Batch(
            obs=obs[has_var].reshape(-1, obs.size(-1)),
            actions=actions[has_var].reshape(-1, H),
            rewards=rewards[has_var].reshape(-1),
            labels=labels[has_var].reshape(-1, H),
            actor_logp_a=logp_a[has_var].reshape(-1, H),
            actor_baseline=baselines[has_var].reshape(-1),
            group_ids=group_ids)

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
        total_correct, total_tokens = 0, 0
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
        return {'test_error': 1.0 - total_correct / total_tokens}


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
        return Batch(obs=contexts, actions=actions, rewards=rewards,
                     labels=labels, actor_logp_a=logp_a, actor_baseline=baseline)

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
        has_mix = (raw_correct.sum(1) > 0) & (raw_correct.sum(1) < group_size)
        if has_mix.sum() == 0:
            has_mix[:] = True
        rewards = self._compute_rewards(actions, labels, actor_lp, contexts, device)

        K = group_size
        n_valid = has_mix.sum().item()

        contexts_f = contexts[has_mix].unsqueeze(1).expand(-1, K, -1).reshape(-1, C)
        labels_f = labels[has_mix].unsqueeze(1).expand(-1, K).reshape(-1)
        actions_f = actions[has_mix].reshape(-1)
        rewards_f = rewards[has_mix].reshape(-1)
        logp_a_f = actor_lp[has_mix].gather(-1, actions[has_mix]).reshape(-1)
        baseline_f = actor_bl[has_mix].unsqueeze(1).expand(-1, K).reshape(-1)
        group_ids = torch.arange(n_valid, device=device).unsqueeze(1).expand(-1, K).reshape(-1)

        return Batch(obs=contexts_f, actions=actions_f, rewards=rewards_f,
                     labels=labels_f, actor_logp_a=logp_a_f,
                     actor_baseline=baseline_f, group_ids=group_ids)

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
