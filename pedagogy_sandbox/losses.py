"""Objective helpers for the Pedagogical RL sandbox."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gather_log_probs(log_probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    if log_probs.shape[:-1] != actions.shape:
        raise ValueError("log_probs prefix shape must match actions shape")
    return log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def group_advantage(scores: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    if scores.dim() != 1:
        raise ValueError("scores must be 1D")
    if group_ids.dim() != 1:
        raise ValueError("group_ids must be 1D")
    if scores.numel() != group_ids.numel():
        raise ValueError("scores and group_ids must have equal length")
    advantage = torch.zeros_like(scores, dtype=torch.float)
    for gid in group_ids.unique(sorted=True):
        mask = group_ids == gid
        group_scores = scores[mask].float()
        centered = group_scores - group_scores.mean()
        std = group_scores.std(unbiased=False)
        advantage[mask] = torch.where(
            std > 1e-8,
            centered / (std + 1e-8),
            torch.zeros_like(centered),
        )
    return advantage


def ppo_surrogate(
        logp_a: torch.Tensor,
        old_logp_a: torch.Tensor,
        advantage: torch.Tensor,
        clip_epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    if logp_a.shape != old_logp_a.shape:
        raise ValueError("logp_a and old_logp_a must have equal shape")
    while advantage.dim() < logp_a.dim():
        advantage = advantage.unsqueeze(-1)
    log_ratio = (logp_a - old_logp_a).clamp(min=-20, max=20)
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    objective = torch.minimum(
        ratio * advantage.detach(),
        clipped * advantage.detach(),
    )
    return -objective.mean(), ratio


def grpo_loss(
        logits: torch.Tensor,
        actions: torch.Tensor,
        old_logp_a: torch.Tensor,
        rewards: torch.Tensor,
        group_ids: torch.Tensor,
        clip_epsilon: float) -> tuple[torch.Tensor, dict[str, float]]:
    log_probs = F.log_softmax(logits, dim=-1)
    logp_a = gather_log_probs(log_probs, actions)
    advantage = group_advantage(rewards, group_ids)
    loss, ratio = ppo_surrogate(
        logp_a, old_logp_a, advantage, clip_epsilon)
    return loss, {
        "loss": loss.item(),
        "reward": rewards.mean().item(),
        "adv_abs_mean": advantage.abs().mean().item(),
        "ratio_mean": ratio.mean().item(),
    }


def spike_learnability(
        student_logits: torch.Tensor,
        actions: torch.Tensor,
        beta: float,
        learnability_lambda: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if beta <= 0:
        raise ValueError("spike_beta must be > 0")
    if learnability_lambda < 0:
        raise ValueError("learnability_lambda must be non-negative")
    log_probs = F.log_softmax(student_logits.float(), dim=-1)
    chosen_logp = gather_log_probs(log_probs, actions)
    best_logp = log_probs.max(dim=-1).values
    surprise_gap = best_logp - chosen_logp
    spike_penalty = (
        torch.logsumexp(beta * surprise_gap, dim=-1)
        - math.log(surprise_gap.size(-1))
    ) / beta
    learnability = torch.exp(-learnability_lambda * spike_penalty)
    diagnostics = {
        "student_logp": chosen_logp,
        "avg_surprisal": -chosen_logp.mean(dim=-1),
        "avg_surprise_gap": surprise_gap.mean(dim=-1),
        "max_surprise_gap": surprise_gap.max(dim=-1).values,
        "spike_penalty": spike_penalty,
        "learnability": learnability,
    }
    return learnability, diagnostics


def pedagogical_rewards(
        task_rewards: torch.Tensor,
        student_logits: torch.Tensor,
        actions: torch.Tensor,
        beta: float,
        learnability_lambda: float) -> tuple[torch.Tensor, dict[str, float]]:
    learnability, diagnostics = spike_learnability(
        student_logits, actions, beta, learnability_lambda)
    rewards = task_rewards.float() * learnability
    return rewards, {
        "task_reward": task_rewards.float().mean().item(),
        "pedagogy_reward": rewards.mean().item(),
        "learnability": diagnostics["learnability"].mean().item(),
        "spike_penalty": diagnostics["spike_penalty"].mean().item(),
        "avg_surprise_gap": diagnostics["avg_surprise_gap"].mean().item(),
        "max_surprise_gap": diagnostics["max_surprise_gap"].mean().item(),
        "avg_student_surprisal": diagnostics["avg_surprisal"].mean().item(),
    }


def additive_teacher_rewards(
        task_rewards: torch.Tensor,
        student_logits: torch.Tensor,
        actions: torch.Tensor,
        penalty_lambda: float) -> tuple[torch.Tensor, dict[str, float]]:
    if penalty_lambda < 0:
        raise ValueError("learnability_lambda must be non-negative")
    log_probs = F.log_softmax(student_logits.float(), dim=-1)
    student_logp = gather_log_probs(log_probs, actions)
    avg_surprisal = -student_logp.mean(dim=-1)
    rewards = task_rewards.float() - penalty_lambda * avg_surprisal
    return rewards, {
        "task_reward": task_rewards.float().mean().item(),
        "teacher_additive_reward": rewards.mean().item(),
        "avg_student_surprisal": avg_surprisal.mean().item(),
    }


def gated_imitation_loss(
        student_logits: torch.Tensor,
        actions: torch.Tensor,
        gate_kappa: float,
        gate_gamma: float) -> tuple[torch.Tensor, dict[str, float]]:
    if gate_kappa <= 0:
        raise ValueError("gate_kappa must be > 0")
    log_probs = F.log_softmax(student_logits, dim=-1)
    logp_a = gather_log_probs(log_probs, actions)
    weights = torch.sigmoid(gate_kappa * (logp_a.detach() - gate_gamma))
    token_loss = -logp_a
    denom = weights.sum(dim=-1).clamp(min=1e-8)
    loss = ((weights * token_loss).sum(dim=-1) / denom).mean()
    return loss, {
        "assim_loss": loss.item(),
        "assim_gate_mean": weights.mean().item(),
        "assim_logp_mean": logp_a.mean().item(),
    }


def sft_loss(
        student_logits: torch.Tensor,
        actions: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    if student_logits.shape[:-1] != actions.shape:
        raise ValueError("student_logits prefix shape must match actions shape")
    loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        actions.reshape(-1),
    )
    return loss, {"sft_loss": loss.item()}
