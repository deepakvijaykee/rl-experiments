"""Set-valued toy task for Vector Policy Optimization.

The task is a contextual bandit with multi-answer rollouts. Each candidate
action has a vector reward, and a rollout emits a set of candidates. The model
is intentionally small: prompt and answer-slot embeddings feed a shared action
head, giving each slot enough capacity to specialize without introducing an LM
training stack.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VPOBatch:
    """Grouped multi-answer rollouts with vector rewards."""

    prompt_ids: torch.Tensor             # [B]
    actions: torch.Tensor                # [B, M]
    vector_rewards: torch.Tensor         # [B, M, D]
    actor_logp_a: torch.Tensor           # [B, M]
    group_ids: torch.Tensor              # [B]
    scalarization_weights: torch.Tensor  # [num_groups, K, D]
    gold_weights: torch.Tensor           # [D]

    def to(self, device: torch.device) -> VPOBatch:
        return replace(
            self,
            prompt_ids=self.prompt_ids.to(device),
            actions=self.actions.to(device),
            vector_rewards=self.vector_rewards.to(device),
            actor_logp_a=self.actor_logp_a.to(device),
            group_ids=self.group_ids.to(device),
            scalarization_weights=self.scalarization_weights.to(device),
            gold_weights=self.gold_weights.to(device),
        )


class SetPolicy(nn.Module):
    """Prompt-conditioned categorical policy for a fixed number of answer slots."""

    def __init__(
            self,
            num_prompts: int,
            num_slots: int,
            num_actions: int,
            hidden: int):
        super().__init__()
        self.num_slots = num_slots
        self.prompt_embed = nn.Embedding(num_prompts, hidden)
        self.slot_embed = nn.Embedding(num_slots, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        slots = torch.arange(self.num_slots, device=prompt_ids.device)
        hidden = (
            self.prompt_embed(prompt_ids).unsqueeze(1)
            + self.slot_embed(slots).unsqueeze(0)
        )
        return self.head(hidden)


class ParetoFrontTask:
    """A two-objective toy where useful search requires multiple trade-offs."""

    name = "pareto_front"
    reward_dim = 2
    gold_weights = torch.tensor([0.5, 0.5])

    def __init__(
            self,
            num_prompts: int = 8,
            num_candidates: int = 3,
            num_weight_samples: int = 16):
        if num_prompts < 1:
            raise ValueError("num_prompts must be >= 1")
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")
        if num_weight_samples < 1:
            raise ValueError("num_weight_samples must be >= 1")
        self.num_prompts = num_prompts
        self.num_candidates = num_candidates
        self.num_weight_samples = num_weight_samples
        self.reward_table = self._build_reward_table(num_prompts)
        self.num_actions = self.reward_table.size(1)

    @staticmethod
    def _build_reward_table(num_prompts: int) -> torch.Tensor:
        base = torch.tensor([
            [1.00, 0.00],
            [0.85, 0.25],
            [0.60, 0.60],
            [0.25, 0.85],
            [0.00, 1.00],
            [0.30, 0.20],
            [0.05, 0.05],
        ])
        rows = []
        for prompt_id in range(num_prompts):
            table = base.roll(shifts=prompt_id % base.size(0), dims=0)
            if prompt_id % 2:
                table = table.flip(dims=(1,))
            rows.append(table)
        return torch.stack(rows)

    def make_model(self, hidden: int) -> nn.Module:
        return SetPolicy(
            num_prompts=self.num_prompts,
            num_slots=self.num_candidates,
            num_actions=self.num_actions,
            hidden=hidden,
        )

    def reward_vectors(
            self,
            prompt_ids: torch.Tensor,
            actions: torch.Tensor) -> torch.Tensor:
        table = self.reward_table.to(prompt_ids.device)
        prompt_table = table[prompt_ids]
        expanded_actions = actions.unsqueeze(-1).expand(
            *actions.shape, self.reward_dim)
        return prompt_table.gather(1, expanded_actions)

    def sample_prompts(
            self,
            num_prompts: int,
            device: torch.device) -> torch.Tensor:
        return torch.randint(self.num_prompts, (num_prompts,), device=device)

    def sample_weights(
            self,
            num_groups: int,
            device: torch.device) -> torch.Tensor:
        concentration = torch.ones(self.reward_dim, device=device)
        return torch.distributions.Dirichlet(concentration).sample(
            (num_groups, self.num_weight_samples))

    @torch.no_grad()
    def sample_batch(
            self,
            model: nn.Module,
            batch_size: int,
            group_size: int,
            device: torch.device) -> VPOBatch:
        if group_size <= 1:
            raise ValueError("group_size must be > 1")
        if batch_size % group_size != 0:
            raise ValueError("batch_size must be divisible by group_size")
        num_groups = batch_size // group_size
        group_prompts = self.sample_prompts(num_groups, device)
        prompt_ids = group_prompts.repeat_interleave(group_size)
        group_ids = torch.arange(num_groups, device=device).repeat_interleave(group_size)

        was_training = model.training
        model.eval()
        logits = model(prompt_ids)
        log_probs = F.log_softmax(logits, dim=-1)
        flat_logits = logits.reshape(-1, logits.size(-1))
        actions = torch.distributions.Categorical(logits=flat_logits).sample()
        actions = actions.reshape(batch_size, self.num_candidates)
        actor_logp_a = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        model.train(was_training)

        return VPOBatch(
            prompt_ids=prompt_ids,
            actions=actions,
            vector_rewards=self.reward_vectors(prompt_ids, actions),
            actor_logp_a=actor_logp_a,
            group_ids=group_ids,
            scalarization_weights=self.sample_weights(num_groups, device),
            gold_weights=self.gold_weights.to(device),
        )

    def compute_logits(self, model: nn.Module, batch: VPOBatch) -> torch.Tensor:
        return model(batch.prompt_ids)

    @torch.no_grad()
    def evaluate(
            self,
            model: nn.Module,
            device: torch.device,
            pool_sets: int = 12) -> dict[str, float]:
        prompt_ids = torch.arange(self.num_prompts, device=device)
        all_scores = []
        all_rewards = []
        was_training = model.training
        model.eval()
        for _ in range(pool_sets):
            logits = model(prompt_ids)
            flat_logits = logits.reshape(-1, logits.size(-1))
            actions = torch.distributions.Categorical(logits=flat_logits).sample()
            actions = actions.reshape(self.num_prompts, self.num_candidates)
            rewards = self.reward_vectors(prompt_ids, actions)
            scores = (rewards * self.gold_weights.to(device)).sum(dim=-1)
            all_scores.append(scores)
            all_rewards.append(rewards)
        model.train(was_training)

        score_pool = torch.cat(all_scores, dim=1)
        reward_sets = torch.stack(all_rewards, dim=1).reshape(
            -1, self.num_candidates, self.reward_dim)
        k3 = min(3, score_pool.size(1))
        k9 = min(9, score_pool.size(1))
        return {
            "mean_scalar_reward": score_pool.mean().item(),
            "best_at_1": score_pool[:, :1].max(dim=1).values.mean().item(),
            "best_at_3": score_pool[:, :k3].max(dim=1).values.mean().item(),
            "best_at_9": score_pool[:, :k9].max(dim=1).values.mean().item(),
            "pool_diversity_l1": pairwise_l1(reward_sets).item(),
        }


def pairwise_l1(reward_sets: torch.Tensor) -> torch.Tensor:
    """Mean pairwise L1 distance inside each generated candidate set."""
    if reward_sets.size(1) < 2:
        return torch.zeros((), device=reward_sets.device)
    diffs = (
        reward_sets.unsqueeze(2) - reward_sets.unsqueeze(1)
    ).abs().sum(dim=-1)
    mask = torch.triu(
        torch.ones(
            reward_sets.size(1),
            reward_sets.size(1),
            device=reward_sets.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    return diffs[:, mask].mean()
