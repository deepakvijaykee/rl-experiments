"""Objective math for the OPD sandbox.

The sandbox keeps two OPD forms separate:

- `OPDReverseKL`: exact full-vocabulary KL(pi_student || pi_teacher).
- `OPDTopKReverseKL`: support-truncated reverse-KL contribution.
- `OPDPG`: sampled-token reverse-KL reward with a clipped IS surrogate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gather_log_probs(log_probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Gather log-probs for sampled actions along the vocabulary dimension."""
    return log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def ppo_surrogate(
        logp_a: torch.Tensor,
        old_logp_a: torch.Tensor,
        advantage: torch.Tensor,
        clip_epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a clipped sampled-token policy-gradient loss and ratio."""
    log_ratio = (logp_a - old_logp_a).clamp(min=-20, max=20)
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    objective = torch.minimum(ratio * advantage.detach(),
                              clipped * advantage.detach())
    return -objective.mean(), ratio


class OPDReverseKLLoss:
    """Full-vocabulary reverse KL on student-sampled states."""
    name = "OPDReverseKL"

    def __call__(self, logits, batch):
        if logits.shape != batch.teacher_log_probs.shape:
            raise ValueError("teacher_log_probs must match logits shape")
        student_log_probs = F.log_softmax(logits.float(), dim=-1)
        student_probs = student_log_probs.exp()
        teacher_log_probs = batch.teacher_log_probs.float()
        teacher_probs = teacher_log_probs.exp()

        kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        loss = kl.mean()
        teacher_entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
        student_entropy = -(student_probs * student_log_probs).sum(dim=-1)
        top1_agreement = (
            student_log_probs.argmax(dim=-1) == teacher_log_probs.argmax(dim=-1)
        ).float().mean()
        return loss, {
            "opd_reverse_kl": loss.item(),
            "teacher_entropy": teacher_entropy.mean().item(),
            "student_entropy": student_entropy.mean().item(),
            "top1_agreement": top1_agreement.item(),
            "reward": batch.rewards.mean().item(),
        }


class OPDTopKReverseKLLoss:
    """Reverse-KL contribution restricted to a top-k support.

    This is an unnormalized truncation of the full expectation under the
    student distribution. It becomes exact full-vocabulary reverse KL when
    `k >= vocab_size`; for smaller k it is a support-restricted diagnostic
    objective, not a proper normalized KL.
    """
    name = "OPDTopKReverseKL"

    def __init__(self, k: int, support: str = "student"):
        if k < 1:
            raise ValueError("top_k must be >= 1")
        if support not in {"student", "teacher", "intersection"}:
            raise ValueError("top_k_support must be student, teacher, or intersection")
        self.k = k
        self.support = support

    def _support_mask(
            self,
            student_log_probs: torch.Tensor,
            teacher_log_probs: torch.Tensor) -> torch.Tensor:
        k = min(self.k, student_log_probs.size(-1))
        student_topk = student_log_probs.topk(k, dim=-1).indices
        teacher_topk = teacher_log_probs.topk(k, dim=-1).indices
        student_mask = torch.zeros_like(student_log_probs).scatter_(
            -1, student_topk, 1.0)
        teacher_mask = torch.zeros_like(student_log_probs).scatter_(
            -1, teacher_topk, 1.0)
        if self.support == "student":
            return student_mask
        if self.support == "teacher":
            return teacher_mask
        return student_mask * teacher_mask

    def __call__(self, logits, batch):
        if logits.shape != batch.teacher_log_probs.shape:
            raise ValueError("teacher_log_probs must match logits shape")
        student_log_probs = F.log_softmax(logits.float(), dim=-1)
        student_probs = student_log_probs.exp()
        teacher_log_probs = batch.teacher_log_probs.float()
        teacher_probs = teacher_log_probs.exp()

        mask = self._support_mask(student_log_probs, teacher_log_probs)
        contribution = (
            mask * student_probs * (student_log_probs - teacher_log_probs)
        ).sum(dim=-1)
        loss = contribution.mean()
        student_topk_mass = (mask * student_probs).sum(dim=-1)
        teacher_topk_mass = (mask * teacher_probs).sum(dim=-1)
        top1_agreement = (
            student_log_probs.argmax(dim=-1) == teacher_log_probs.argmax(dim=-1)
        ).float().mean()
        return loss, {
            "opd_topk_reverse_kl": loss.item(),
            "student_topk_mass": student_topk_mass.mean().item(),
            "teacher_topk_mass": teacher_topk_mass.mean().item(),
            "topk_support_size": mask.sum(dim=-1).float().mean().item(),
            "top1_agreement": top1_agreement.item(),
            "reward": batch.rewards.mean().item(),
        }


class OPDPGLoss:
    """Sampled-token OPD via reverse-KL reward.

    The sampled token advantage is log pi_teacher(a|s) - log pi_old(a|s), the
    negative of the sampled reverse-KL contribution.
    """
    name = "OPDPG"

    def __init__(self, clip_epsilon: float = 0.2):
        self.clip_epsilon = clip_epsilon

    def __call__(self, logits, batch):
        log_probs = F.log_softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, batch.actions)
        advantage = batch.teacher_logp_a - batch.actor_logp_a
        loss, ratio = ppo_surrogate(
            logp_a, batch.actor_logp_a, advantage, self.clip_epsilon)
        sampled_reverse_kl = batch.actor_logp_a - batch.teacher_logp_a
        return loss, {
            "sampled_reverse_kl": sampled_reverse_kl.mean().item(),
            "opd_advantage_mean": advantage.mean().item(),
            "ratio_mean": ratio.mean().item(),
            "reward": batch.rewards.mean().item(),
        }


def opd_diagnostics(logits, batch, overlap_k: int = 2) -> dict[str, float]:
    """Method-independent OPD diagnostics on the current learner logits."""
    if logits.shape != batch.teacher_log_probs.shape:
        raise ValueError("teacher_log_probs must match logits shape")
    if overlap_k < 1:
        raise ValueError("overlap_k must be >= 1")

    student_log_probs = F.log_softmax(logits.float(), dim=-1)
    student_probs = student_log_probs.exp()
    teacher_log_probs = batch.teacher_log_probs.float()
    teacher_probs = teacher_log_probs.exp()
    reverse_kl = (
        student_probs * (student_log_probs - teacher_log_probs)
    ).sum(dim=-1)
    forward_kl = (
        teacher_probs * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)
    teacher_entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
    student_entropy = -(student_probs * student_log_probs).sum(dim=-1)
    sampled_reverse_kl = batch.actor_logp_a - batch.teacher_logp_a
    top1_agreement = (
        student_log_probs.argmax(dim=-1) == teacher_log_probs.argmax(dim=-1)
    ).float().mean()

    k = min(overlap_k, logits.size(-1))
    student_top = student_log_probs.topk(k, dim=-1).indices
    teacher_top = teacher_log_probs.topk(k, dim=-1).indices
    overlap = (
        student_top.unsqueeze(-1) == teacher_top.unsqueeze(-2)
    ).any(dim=-1).float().mean()

    return {
        "diag_reverse_kl": reverse_kl.mean().item(),
        "diag_forward_kl": forward_kl.mean().item(),
        "diag_sampled_reverse_kl": sampled_reverse_kl.mean().item(),
        "diag_teacher_entropy": teacher_entropy.mean().item(),
        "diag_student_entropy": student_entropy.mean().item(),
        "diag_entropy_gap": (student_entropy - teacher_entropy).mean().item(),
        "diag_top1_agreement": top1_agreement.item(),
        f"diag_overlap_at_{k}": overlap.item(),
    }
