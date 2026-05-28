"""Objective math for RLM tree GRPO.

This module is intentionally model-free. The trainer owns tokenization,
generation, and backprop; these helpers own the method invariants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


ROOT_ROLE = "root"
CHILD_ROLE = "child"


@dataclass
class RLMTrainingSegment:
    """One generated policy segment inside an RLM rollout tree."""

    prompt_ids: list[int]
    completion_ids: list[int]
    old_logprobs: list[float]
    group_id: int
    tree_id: int
    role: str
    segment_weight: float
    advantage: float = 0.0

    def __post_init__(self):
        if self.role not in {ROOT_ROLE, CHILD_ROLE}:
            raise ValueError(f"unknown RLM segment role: {self.role}")
        if not self.prompt_ids:
            raise ValueError("RLMTrainingSegment requires prompt tokens")
        if not self.completion_ids:
            raise ValueError("RLMTrainingSegment requires generated completion tokens")
        if len(self.completion_ids) != len(self.old_logprobs):
            raise ValueError("completion_ids and old_logprobs must have equal length")
        if self.segment_weight <= 0:
            raise ValueError("segment_weight must be positive")


def group_relative_advantages(
        rewards: torch.Tensor,
        group_ids: torch.Tensor,
        scale_rewards: bool = True,
        eps: float = 1e-8) -> torch.Tensor:
    """Compute GRPO advantages independently within each prompt group."""
    if rewards.dim() != 1 or group_ids.dim() != 1:
        raise ValueError("rewards and group_ids must be 1D")
    if rewards.numel() != group_ids.numel():
        raise ValueError("rewards and group_ids must have equal length")

    advantages = torch.zeros_like(rewards, dtype=torch.float)
    for group_id in group_ids.unique(sorted=True):
        mask = group_ids == group_id
        group_rewards = rewards[mask].float()
        centered = group_rewards - group_rewards.mean()
        if scale_rewards:
            std = group_rewards.std(unbiased=False)
            centered = torch.where(
                std > eps,
                centered / (std + eps),
                torch.zeros_like(centered),
            )
        advantages[mask] = centered
    return advantages


def rlm_segment_weights(
        root_segment_count: int,
        child_rollout_segment_counts: Sequence[int],
        train_child_trajectories: bool) -> tuple[float, list[float]]:
    """Return per-segment weights for one RLM rollout tree.

    The root trajectory may be represented by multiple model calls, such as a
    planner turn and a final-answer turn. Their weights sum to 1. Each child
    rollout receives 1/k of the child gradient budget, and that child share is
    split across the child's own generated turns.
    """
    if root_segment_count <= 0:
        raise ValueError("RLM rollouts must contain at least one root segment")
    root_weight = 1.0 / root_segment_count
    if not train_child_trajectories or not child_rollout_segment_counts:
        return root_weight, []
    child_count = len(child_rollout_segment_counts)
    child_weights: list[float] = []
    for segment_count in child_rollout_segment_counts:
        if segment_count <= 0:
            raise ValueError("child rollouts must contain at least one segment")
        child_weights.extend([1.0 / child_count / segment_count] * segment_count)
    return root_weight, child_weights


def grpo_segment_losses(
        current_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        clip_epsilon: float,
        ref_logprobs: torch.Tensor | None = None,
        kl_coef: float = 0.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Return one GRPO loss per generated segment.

    Each segment is normalized by its own generated-token count. The caller is
    responsible for applying RLM tree weights and dividing by the number of
    rollout trees.
    """
    if current_logprobs.shape != old_logprobs.shape:
        raise ValueError("current_logprobs and old_logprobs must have equal shape")
    if current_logprobs.shape != mask.shape:
        raise ValueError("mask must match logprob shape")

    while advantages.dim() < current_logprobs.dim():
        advantages = advantages.unsqueeze(-1)
    advantages = advantages.expand_as(current_logprobs)

    log_ratio = (current_logprobs - old_logprobs).clamp(min=-20, max=20)
    ratio = log_ratio.exp()
    clipped = ratio.clamp(min=1.0 - clip_epsilon, max=1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)

    kl = torch.zeros_like(surrogate)
    if ref_logprobs is not None and kl_coef > 0:
        if ref_logprobs.shape != current_logprobs.shape:
            raise ValueError("ref_logprobs must match current_logprobs")
        logp_ref_minus_current = (ref_logprobs - current_logprobs).clamp(min=-20, max=20)
        kl = logp_ref_minus_current.exp() - logp_ref_minus_current - 1.0

    mask_f = mask.float()
    token_loss = -(surrogate - kl_coef * kl) * mask_f
    denom = mask_f.sum(dim=1).clamp(min=1.0)
    losses = token_loss.sum(dim=1) / denom
    total_denom = mask_f.sum().clamp(min=1.0)
    metrics = {
        "ratio_mean": (ratio * mask_f).sum().item() / total_denom.item(),
        "clip_frac": (
            ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).float()
            * mask_f).sum().item() / total_denom.item(),
        "kl_mean": (kl * mask_f).sum().item() / total_denom.item(),
    }
    return losses, metrics
