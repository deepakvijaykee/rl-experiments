"""Objective math for the VPO sandbox."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gather_log_probs(log_probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def vpo_set_scores(
        vector_rewards: torch.Tensor,
        scalarization_weights: torch.Tensor,
        group_ids: torch.Tensor) -> torch.Tensor:
    """Equation 1 Monte-Carlo estimate: E_w max_y w^T r(x, y)."""
    unique_groups = group_ids.unique()
    if scalarization_weights.size(0) != unique_groups.numel():
        raise ValueError("scalarization_weights must have one row per group")
    scores = torch.zeros(vector_rewards.size(0), device=vector_rewards.device)
    for weight_idx, gid in enumerate(unique_groups):
        mask = group_ids == gid
        scalarized = torch.einsum(
            "bmd,kd->bmk",
            vector_rewards[mask].float(),
            scalarization_weights[weight_idx].float(),
        )
        scores[mask] = scalarized.max(dim=1).values.mean(dim=-1)
    return scores


def fixed_scalar_set_scores(
        vector_rewards: torch.Tensor,
        scalar_weights: torch.Tensor,
        reduce: str) -> torch.Tensor:
    scalarized = (vector_rewards.float() * scalar_weights.float()).sum(dim=-1)
    if reduce == "single":
        if scalarized.size(1) != 1:
            raise ValueError("single-answer scalar GRPO requires num_candidates=1")
        return scalarized[:, 0]
    if reduce == "max":
        if scalarized.size(1) <= 1:
            raise ValueError("multi-answer RLVR requires num_candidates > 1")
        return scalarized.max(dim=1).values
    raise ValueError(f"unknown scalar set reduction: {reduce}")


def group_advantage(scores: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    advantage = torch.zeros_like(scores)
    for gid in group_ids.unique():
        mask = group_ids == gid
        group_scores = scores[mask]
        centered = group_scores - group_scores.mean()
        std = group_scores.std(unbiased=False)
        if std > 1e-8:
            advantage[mask] = centered / (std + 1e-8)
    return advantage


def ppo_surrogate(
        logp_a: torch.Tensor,
        old_logp_a: torch.Tensor,
        advantage: torch.Tensor,
        clip_epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    log_ratio = (logp_a - old_logp_a).clamp(min=-20, max=20)
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    objective = torch.minimum(
        ratio * advantage.detach(),
        clipped * advantage.detach(),
    )
    return -objective.mean(), ratio


class ScalarGRPOLoss:
    """Single-answer scalar GRPO baseline."""

    name = "ScalarGRPO"

    def __init__(self, clip_epsilon: float = 0.2):
        self.clip_epsilon = clip_epsilon

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, batch.actions)
        scores = fixed_scalar_set_scores(
            batch.vector_rewards, batch.gold_weights, reduce="single")
        advantage = group_advantage(scores, batch.group_ids).unsqueeze(1)
        loss, ratio = ppo_surrogate(
            logp_a, batch.actor_logp_a, advantage, self.clip_epsilon)
        return loss, {
            "reward": scores.mean().item(),
            "adv_abs_mean": advantage.abs().mean().item(),
            "ratio_mean": ratio.mean().item(),
        }


class MultiRLVRLoss:
    """Multi-answer scalar baseline using best candidate under fixed weights."""

    name = "MultiRLVR"

    def __init__(self, clip_epsilon: float = 0.2):
        self.clip_epsilon = clip_epsilon

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, batch.actions)
        scores = fixed_scalar_set_scores(
            batch.vector_rewards, batch.gold_weights, reduce="max")
        advantage = group_advantage(scores, batch.group_ids).unsqueeze(1)
        loss, ratio = ppo_surrogate(
            logp_a, batch.actor_logp_a, advantage, self.clip_epsilon)
        return loss, {
            "reward": scores.mean().item(),
            "adv_abs_mean": advantage.abs().mean().item(),
            "ratio_mean": ratio.mean().item(),
        }


class VPOGRPOLoss:
    """VPO set reward plugged into the GRPO clipped surrogate."""

    name = "VPOGRPO"

    def __init__(self, clip_epsilon: float = 0.2):
        self.clip_epsilon = clip_epsilon

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, batch.actions)
        scores = vpo_set_scores(
            batch.vector_rewards,
            batch.scalarization_weights,
            batch.group_ids,
        )
        advantage = group_advantage(scores, batch.group_ids).unsqueeze(1)
        loss, ratio = ppo_surrogate(
            logp_a, batch.actor_logp_a, advantage, self.clip_epsilon)
        gold_scores = fixed_scalar_set_scores(
            batch.vector_rewards, batch.gold_weights, reduce="max")
        return loss, {
            "reward": scores.mean().item(),
            "gold_set_reward": gold_scores.mean().item(),
            "adv_abs_mean": advantage.abs().mean().item(),
            "ratio_mean": ratio.mean().item(),
        }

