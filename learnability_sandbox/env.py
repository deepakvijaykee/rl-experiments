"""Analytically tractable unique-path decision environment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LayeredEnvironmentConfig:
    num_prompts: int = 128
    horizon: int = 3
    branching_factor: int = 4
    initial_correct_probability: float = 0.5
    task_seed: int = 0

    def __post_init__(self) -> None:
        if self.num_prompts < 1:
            raise ValueError("num_prompts must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.branching_factor < 2:
            raise ValueError("branching_factor must be >= 2")
        if not 0.0 < self.initial_correct_probability < 1.0:
            raise ValueError("initial_correct_probability must be in (0, 1)")


@dataclass(frozen=True)
class RolloutBatch:
    """Grouped trajectories padded to ``horizon`` decisions."""

    prompt_ids: torch.Tensor
    actions: torch.Tensor
    action_mask: torch.Tensor
    rewards: torch.Tensor


class TabularLayeredPolicy(nn.Module):
    """One categorical policy per prompt and decision depth."""

    def __init__(
        self,
        target_actions: torch.Tensor,
        branching_factor: int,
        initial_correct_probability: float,
    ) -> None:
        super().__init__()
        num_prompts, horizon = target_actions.shape
        logits = torch.zeros(
            num_prompts,
            horizon,
            branching_factor,
            device=target_actions.device,
        )
        correct_logit = math.log(
            initial_correct_probability
            * (branching_factor - 1)
            / (1.0 - initial_correct_probability)
        )
        logits.scatter_(-1, target_actions.unsqueeze(-1), correct_logit)
        self.logits = nn.Parameter(logits)

    def forward(self, prompt_ids: torch.Tensor, depth: int) -> torch.Tensor:
        return self.logits[prompt_ids, depth]


class LayeredDecisionEnvironment:
    """Unique-path task where one wrong action terminates the episode."""

    def __init__(
        self,
        config: LayeredEnvironmentConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        generator = torch.Generator().manual_seed(config.task_seed)
        self._target_actions = torch.randint(
            config.branching_factor,
            (config.num_prompts, config.horizon),
            generator=generator,
        ).to(device)

    def make_policy(self) -> TabularLayeredPolicy:
        return TabularLayeredPolicy(
            self._target_actions,
            self.config.branching_factor,
            self.config.initial_correct_probability,
        )

    @torch.no_grad()
    def rollout(
        self,
        policy: TabularLayeredPolicy,
        prompt_ids: torch.Tensor,
        group_size: int,
    ) -> RolloutBatch:
        targets = self._target_actions[prompt_ids]
        num_groups = prompt_ids.numel()
        shape = (num_groups, group_size, self.config.horizon)
        actions = torch.empty(shape, dtype=torch.long, device=prompt_ids.device)
        action_mask = torch.empty(shape, dtype=torch.bool, device=prompt_ids.device)
        active = torch.ones(
            num_groups,
            group_size,
            dtype=torch.bool,
            device=prompt_ids.device,
        )

        for depth in range(self.config.horizon):
            probabilities = F.softmax(policy(prompt_ids, depth), dim=-1)
            sampled = torch.multinomial(
                probabilities.unsqueeze(1)
                .expand(-1, group_size, -1)
                .reshape(-1, self.config.branching_factor),
                num_samples=1,
            ).reshape(num_groups, group_size)
            actions[:, :, depth] = sampled
            action_mask[:, :, depth] = active
            active &= sampled.eq(targets[:, depth].unsqueeze(1))

        return RolloutBatch(
            prompt_ids=prompt_ids,
            actions=actions,
            action_mask=action_mask,
            rewards=active.float(),
        )

    @torch.no_grad()
    def exact_metrics(
        self,
        policy: TabularLayeredPolicy,
        group_size: int,
    ) -> dict[str, float]:
        probabilities = F.softmax(policy.logits, dim=-1)
        correct_probabilities = probabilities.gather(
            -1,
            self._target_actions.unsqueeze(-1),
        ).squeeze(-1)
        success_probability = correct_probabilities.prod(dim=1)
        mixed_group_probability = (
            1.0
            - success_probability.pow(group_size)
            - (1.0 - success_probability).pow(group_size)
        )

        reach_probability = torch.ones_like(correct_probabilities)
        if self.config.horizon > 1:
            reach_probability[:, 1:] = correct_probabilities[:, :-1].cumprod(dim=1)

        metrics = {
            "success_probability": success_probability.mean().item(),
            "pass_at_group_size": (
                1.0 - (1.0 - success_probability).pow(group_size)
            ).mean().item(),
            "predicted_mixed_group_rate": mixed_group_probability.mean().item(),
            "mean_state_correct_probability": correct_probabilities.mean().item(),
            "expected_visited_steps": reach_probability.sum(dim=1).mean().item(),
        }
        for depth in range(self.config.horizon):
            metrics[f"correct_probability_depth_{depth + 1}"] = (
                correct_probabilities[:, depth].mean().item()
            )
            metrics[f"reach_probability_depth_{depth + 1}"] = (
                reach_probability[:, depth].mean().item()
            )
        return metrics

def group_centered_policy_loss(
    policy: TabularLayeredPolicy,
    batch: RolloutBatch,
    normalization: str = "mean",
) -> torch.Tensor:
    """Trajectory policy gradient with a within-prompt group baseline.

    ``normalization`` selects the advantage estimator:

    - ``"mean"``: reward minus the group mean — unbiased, so the expected
      update equals the true policy gradient and cold-task drift inherits the
      gradient's own ``q`` suppression;
    - ``"std"``: the GRPO estimator — mean-centered then divided by the group
      standard deviation, which rescales every mixed group to an O(1) update
      regardless of how rare success is. All-success and all-failure groups
      have zero advantage under both.

    For small ``q`` both drift as ``q̇ ∝ q²``; std normalization multiplies
    the rate by roughly ``√K``. The flag exists to measure that factor, not
    to change the band structure.
    """
    if normalization not in ("mean", "std"):
        raise ValueError("normalization must be mean or std")
    group_size = batch.rewards.size(1)
    sequence_log_probabilities = torch.zeros_like(batch.rewards)
    for depth in range(batch.actions.size(2)):
        log_probabilities = F.log_softmax(policy(batch.prompt_ids, depth), dim=-1)
        chosen_log_probabilities = (
            log_probabilities.unsqueeze(1)
            .expand(-1, group_size, -1)
            .gather(-1, batch.actions[:, :, depth].unsqueeze(-1))
            .squeeze(-1)
        )
        sequence_log_probabilities += (
            chosen_log_probabilities * batch.action_mask[:, :, depth]
        )

    advantages = batch.rewards - batch.rewards.mean(dim=1, keepdim=True)
    if normalization == "std":
        deviation = batch.rewards.std(dim=1, unbiased=False, keepdim=True)
        advantages = torch.where(
            deviation > 0, advantages / deviation, torch.zeros_like(advantages)
        )
    return -(sequence_log_probabilities * advantages).mean()


@torch.no_grad()
def rollout_metrics(batch: RolloutBatch) -> dict[str, float]:
    reward_standard_deviation = batch.rewards.std(dim=1, unbiased=False)
    advantages = batch.rewards - batch.rewards.mean(dim=1, keepdim=True)
    return {
        "sampled_reward": batch.rewards.mean().item(),
        "sampled_mixed_group_rate": (
            reward_standard_deviation > 0
        ).float().mean().item(),
        "sampled_all_zero_group_rate": (
            batch.rewards.eq(0).all(dim=1).float().mean().item()
        ),
        "sampled_all_one_group_rate": (
            batch.rewards.eq(1).all(dim=1).float().mean().item()
        ),
        "sampled_trajectory_length": (
            batch.action_mask.sum(dim=2).float().mean().item()
        ),
        "advantage_abs_mean": advantages.abs().mean().item(),
    }
