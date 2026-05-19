"""Toy autoregressive tasks for on-policy distillation.

The teacher is an oracle distribution over the next token at each
student-sampled prefix. It is intentionally simple: the goal is to isolate OPD
mechanics, not to benchmark teacher capability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OPDBatch:
    """Student rollouts plus teacher distributions on the visited states."""

    obs: torch.Tensor                  # [B, input_len + 1 + H]
    actions: torch.Tensor              # [B, H]
    labels: torch.Tensor               # [B, H]
    rewards: torch.Tensor              # [B]
    actor_logp_a: torch.Tensor         # [B, H]
    teacher_log_probs: torch.Tensor    # [B, H, V]
    teacher_logp_a: torch.Tensor       # [B, H]

    def to(self, device) -> OPDBatch:
        return replace(
            self,
            obs=self.obs.to(device),
            actions=self.actions.to(device),
            labels=self.labels.to(device),
            rewards=self.rewards.to(device),
            actor_logp_a=self.actor_logp_a.to(device),
            teacher_log_probs=self.teacher_log_probs.to(device),
            teacher_logp_a=self.teacher_logp_a.to(device),
        )


def smoothed_label_log_probs(
        labels: torch.Tensor,
        vocab_size: int,
        epsilon: float) -> torch.Tensor:
    """Return a full-support oracle teacher distribution."""
    if vocab_size < 2:
        raise ValueError("vocab_size must be >= 2")
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("teacher_epsilon must be in (0, 1)")
    if labels.numel() and (labels.min() < 0 or labels.max() >= vocab_size):
        raise ValueError("labels must be in the teacher vocabulary")
    off_prob = epsilon / (vocab_size - 1)
    probs = torch.full(
        (*labels.shape, vocab_size),
        off_prob,
        dtype=torch.float,
        device=labels.device,
    )
    probs.scatter_(-1, labels.unsqueeze(-1), 1.0 - epsilon)
    return probs.log()


def soft_label_log_probs(
        labels: torch.Tensor,
        content_vocab_size: int,
        action_vocab_size: int,
        temperature: float,
        special_token_weight: float) -> torch.Tensor:
    """Return a graded full-support teacher over content plus special tokens."""
    if content_vocab_size < 2:
        raise ValueError("content_vocab_size must be >= 2")
    if action_vocab_size < content_vocab_size:
        raise ValueError("action_vocab_size must cover content tokens")
    if temperature <= 0:
        raise ValueError("teacher_temperature must be > 0")
    if special_token_weight <= 0:
        raise ValueError("special_token_weight must be > 0")
    if labels.numel() and (labels.min() < 0 or labels.max() >= content_vocab_size):
        raise ValueError("labels must be in the content vocabulary")

    token_ids = torch.arange(content_vocab_size, device=labels.device)
    distance = (token_ids - labels.unsqueeze(-1)).abs()
    cyclic_distance = torch.minimum(distance, content_vocab_size - distance)
    # The tiny token-id factor breaks symmetric ties without changing the
    # teacher's main distance-based preference.
    tie_break = 1.0 + 1e-3 * token_ids.float()
    content_weights = torch.exp(
        -cyclic_distance.float() / temperature) * tie_break
    if action_vocab_size > content_vocab_size:
        special_shape = (*labels.shape, action_vocab_size - content_vocab_size)
        special_weights = torch.full(
            special_shape,
            special_token_weight,
            dtype=torch.float,
            device=labels.device,
        )
        weights = torch.cat([content_weights, special_weights], dim=-1)
    else:
        weights = content_weights
    probs = weights / weights.sum(dim=-1, keepdim=True)
    return probs.log()


class ReversalTask:
    """Reverse a random token string after a separator token."""

    name = "reversal"

    def __init__(self, vocab_size: int = 2, seq_len: int = 10,
                 teacher_epsilon: float = 1e-3):
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.teacher_epsilon = teacher_epsilon
        self.sep_token = vocab_size
        self.num_actions = vocab_size + 1

    def make_model(self, d_model: int, nhead: int, num_layers: int) -> nn.Module:
        from rl_sandbox.models import CausalTransformer
        return CausalTransformer(
            vocab_size=self.num_actions,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_seq_len=self.seq_len * 2 + 1,
        )

    def sample_inputs(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(self.vocab_size, (batch_size, self.seq_len), device=device)

    def target_tokens(self, input_tokens: torch.Tensor) -> torch.Tensor:
        return input_tokens.flip(1)

    def reward_from_actions(self, actions: torch.Tensor,
                            labels: torch.Tensor) -> torch.Tensor:
        return (actions == labels).float().mean(dim=1)

    def teacher_log_probs_for_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return smoothed_label_log_probs(
            labels, self.num_actions, self.teacher_epsilon)

    @torch.no_grad()
    def sample_batch(self, model: nn.Module, batch_size: int,
                     device: torch.device) -> OPDBatch:
        H = self.seq_len
        inputs = self.sample_inputs(batch_size, device)
        labels = self.target_tokens(inputs)
        sep = torch.full((batch_size, 1), self.sep_token,
                         device=device, dtype=torch.long)
        prefix = torch.cat([inputs, sep], dim=1)
        actions, actor_logp = [], []

        was_training = model.training
        model.eval()
        for _ in range(H):
            logits = model(prefix)[:, -1]
            log_probs = F.log_softmax(logits, dim=-1)
            next_token = torch.distributions.Categorical(logits=logits).sample()
            actions.append(next_token)
            actor_logp.append(log_probs.gather(
                -1, next_token.unsqueeze(-1)).squeeze(-1))
            prefix = torch.cat([prefix, next_token.unsqueeze(1)], dim=1)
        model.train(was_training)

        actions_t = torch.stack(actions, dim=1)
        actor_logp_t = torch.stack(actor_logp, dim=1)
        teacher_log_probs = self.teacher_log_probs_for_labels(labels)
        teacher_logp_a = teacher_log_probs.gather(
            -1, actions_t.unsqueeze(-1)).squeeze(-1)
        rewards = self.reward_from_actions(actions_t, labels)
        obs = torch.cat([inputs, sep, actions_t], dim=1)
        return OPDBatch(
            obs=obs,
            actions=actions_t,
            labels=labels,
            rewards=rewards,
            actor_logp_a=actor_logp_t,
            teacher_log_probs=teacher_log_probs,
            teacher_logp_a=teacher_logp_a,
        )

    def compute_logits(self, model: nn.Module, batch: OPDBatch) -> torch.Tensor:
        logits_full = model(batch.obs[:, :-1])
        return logits_full[:, self.seq_len:, :]

    @torch.no_grad()
    def _greedy_generate(self, model: nn.Module, inputs: torch.Tensor):
        H = self.seq_len
        batch_size = inputs.size(0)
        device = inputs.device
        labels = self.target_tokens(inputs)
        sep = torch.full((batch_size, 1), self.sep_token,
                         device=device, dtype=torch.long)
        prefix = torch.cat([inputs, sep], dim=1)
        total_entropy = 0.0

        for _ in range(H):
            logits = model(prefix)[:, -1]
            log_probs = F.log_softmax(logits, dim=-1)
            total_entropy += -(log_probs.exp() * log_probs).sum(dim=-1).sum().item()
            next_token = logits.argmax(dim=-1)
            prefix = torch.cat([prefix, next_token.unsqueeze(1)], dim=1)
        return prefix[:, H + 1:], labels, total_entropy, batch_size * H

    @torch.no_grad()
    def evaluate(self, model: nn.Module, device: torch.device,
                 num_batches: int = 10, batch_size: int = 100) -> dict[str, float]:
        total_correct, total_tokens = 0.0, 0
        total_entropy, entropy_tokens = 0.0, 0
        was_training = model.training
        model.eval()
        for _ in range(num_batches):
            inputs = self.sample_inputs(batch_size, device)
            generated, labels, entropy, count = self._greedy_generate(model, inputs)
            total_correct += (generated == labels).float().sum().item()
            total_tokens += batch_size * self.seq_len
            total_entropy += entropy
            entropy_tokens += count
        model.train(was_training)
        return {
            "test_error": 1.0 - total_correct / total_tokens,
            "entropy": total_entropy / entropy_tokens,
        }


class SoftReversalTask(ReversalTask):
    """Reversal with a graded teacher over nearby content tokens."""

    name = "soft_reversal"

    def __init__(
            self,
            vocab_size: int = 8,
            seq_len: int = 8,
            teacher_temperature: float = 1.0,
            special_token_weight: float = 0.02):
        super().__init__(
            vocab_size=vocab_size,
            seq_len=seq_len,
            teacher_epsilon=1e-3,
        )
        self.teacher_temperature = teacher_temperature
        self.special_token_weight = special_token_weight

    def teacher_log_probs_for_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return soft_label_log_probs(
            labels,
            content_vocab_size=self.vocab_size,
            action_vocab_size=self.num_actions,
            temperature=self.teacher_temperature,
            special_token_weight=self.special_token_weight,
        )


class FormatAnswerTask(ReversalTask):
    """Emit an answer tag, then a modular checksum, then copy the suffix."""

    name = "format_answer"

    def __init__(self, vocab_size: int = 5, seq_len: int = 3,
                 teacher_epsilon: float = 1e-3):
        if seq_len < 2:
            raise ValueError("format_answer requires seq_len >= 2")
        super().__init__(
            vocab_size=vocab_size,
            seq_len=seq_len,
            teacher_epsilon=teacher_epsilon,
        )
        self.answer_token = vocab_size + 1
        self.num_actions = vocab_size + 2

    def target_tokens(self, input_tokens: torch.Tensor) -> torch.Tensor:
        targets = torch.zeros_like(input_tokens)
        targets[:, 0] = self.answer_token
        targets[:, 1] = (input_tokens[:, 0] + input_tokens[:, 1]) % self.vocab_size
        if self.seq_len > 2:
            targets[:, 2:] = input_tokens[:, 2:]
        return targets

    def reward_from_actions(self, actions: torch.Tensor,
                            labels: torch.Tensor) -> torch.Tensor:
        format_ok = (actions[:, 0] == labels[:, 0]).float()
        answer_ok = (actions[:, 1] == labels[:, 1]).float()
        final = (actions == labels).all(dim=1).float()
        return (format_ok + answer_ok + final) / 3.0
