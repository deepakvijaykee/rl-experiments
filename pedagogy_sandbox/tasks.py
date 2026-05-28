"""Toy privileged-context task for Pedagogical RL.

The task is small by design. The student sees two input tokens and must emit a
short trajectory whose final token is the modular sum. The teacher sees the
same input plus the privileged final answer. Scratch tokens before the final
answer are unconstrained, which gives the teacher room to choose trajectories
that differ in learnability under the student.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_sandbox.models import CausalTransformer


@dataclass
class PedagogyBatch:
    """Rollouts with both student and privileged-teacher views."""

    student_obs: torch.Tensor
    teacher_obs: torch.Tensor
    actions: torch.Tensor
    answers: torch.Tensor
    task_rewards: torch.Tensor
    actor_logp_a: torch.Tensor
    group_ids: torch.Tensor
    informative_group_rate: float

    def to(self, device: torch.device) -> PedagogyBatch:
        return replace(
            self,
            student_obs=self.student_obs.to(device),
            teacher_obs=self.teacher_obs.to(device),
            actions=self.actions.to(device),
            answers=self.answers.to(device),
            task_rewards=self.task_rewards.to(device),
            actor_logp_a=self.actor_logp_a.to(device),
            group_ids=self.group_ids.to(device),
        )


class HintedArithmeticTask:
    """Modular arithmetic with privileged answer context for the teacher."""

    name = "hinted_arithmetic"

    def __init__(self, content_vocab_size: int = 5, completion_len: int = 3):
        if content_vocab_size < 2:
            raise ValueError("content_vocab_size must be >= 2")
        if completion_len < 1:
            raise ValueError("completion_len must be >= 1")
        self.content_vocab_size = content_vocab_size
        self.completion_len = completion_len
        self.query_sep = content_vocab_size
        self.hint_sep = content_vocab_size + 1
        self.num_actions = content_vocab_size + 2
        self.student_prompt_len = 3
        self.teacher_prompt_len = 5

    @property
    def max_seq_len(self) -> int:
        return self.teacher_prompt_len + self.completion_len

    def make_model(self, d_model: int, nhead: int, num_layers: int) -> nn.Module:
        return CausalTransformer(
            vocab_size=self.num_actions,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_seq_len=self.max_seq_len,
        )

    def sample_inputs(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(
            self.content_vocab_size, (batch_size, 2), device=device)

    def answer_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs[:, 0] + inputs[:, 1]) % self.content_vocab_size

    def student_prompts(self, inputs: torch.Tensor) -> torch.Tensor:
        sep = torch.full(
            (inputs.size(0), 1),
            self.query_sep,
            dtype=torch.long,
            device=inputs.device,
        )
        return torch.cat([inputs, sep], dim=1)

    def teacher_prompts(
            self,
            inputs: torch.Tensor,
            answers: torch.Tensor) -> torch.Tensor:
        hint = torch.full(
            (inputs.size(0), 1),
            self.hint_sep,
            dtype=torch.long,
            device=inputs.device,
        )
        sep = torch.full(
            (inputs.size(0), 1),
            self.query_sep,
            dtype=torch.long,
            device=inputs.device,
        )
        return torch.cat([inputs, hint, answers.unsqueeze(1), sep], dim=1)

    def reward_from_actions(
            self,
            actions: torch.Tensor,
            answers: torch.Tensor) -> torch.Tensor:
        return (actions[:, -1] == answers).float()

    @torch.no_grad()
    def _rollout_from_prompts(
            self,
            model: nn.Module,
            prompts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        was_training = model.training
        model.eval()
        prefix = prompts
        actions = []
        logp_actions = []
        for _ in range(self.completion_len):
            logits = model(prefix)[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            next_token = torch.distributions.Categorical(logits=logits).sample()
            actions.append(next_token)
            logp_actions.append(log_probs.gather(
                -1, next_token.unsqueeze(-1)).squeeze(-1))
            prefix = torch.cat([prefix, next_token.unsqueeze(1)], dim=1)
        model.train(was_training)
        return (
            torch.stack(actions, dim=1),
            torch.stack(logp_actions, dim=1),
            prefix,
        )

    @staticmethod
    def _informative_group_rate(
            rewards: torch.Tensor,
            group_size: int) -> float:
        if group_size == 1:
            return 0.0
        grouped = rewards.reshape(-1, group_size)
        mixed = (grouped.min(dim=1).values < grouped.max(dim=1).values)
        return mixed.float().mean().item()

    def sample_teacher_rollouts(
            self,
            teacher: nn.Module,
            batch_size: int,
            group_size: int,
            device: torch.device) -> PedagogyBatch:
        if group_size < 1:
            raise ValueError("group_size must be >= 1")
        if batch_size % group_size != 0:
            raise ValueError("batch_size must be divisible by group_size")
        num_groups = batch_size // group_size
        group_inputs = self.sample_inputs(num_groups, device)
        group_answers = self.answer_tokens(group_inputs)
        inputs = group_inputs.repeat_interleave(group_size, dim=0)
        answers = group_answers.repeat_interleave(group_size)

        teacher_prompts = self.teacher_prompts(inputs, answers)
        actions, actor_logp_a, teacher_obs = self._rollout_from_prompts(
            teacher, teacher_prompts)
        student_obs = torch.cat([self.student_prompts(inputs), actions], dim=1)
        rewards = self.reward_from_actions(actions, answers)
        group_ids = torch.arange(
            num_groups, device=device).repeat_interleave(group_size)
        return PedagogyBatch(
            student_obs=student_obs,
            teacher_obs=teacher_obs,
            actions=actions,
            answers=answers,
            task_rewards=rewards,
            actor_logp_a=actor_logp_a,
            group_ids=group_ids,
            informative_group_rate=self._informative_group_rate(
                rewards, group_size),
        )

    def sample_student_rollouts(
            self,
            student: nn.Module,
            batch_size: int,
            group_size: int,
            device: torch.device) -> PedagogyBatch:
        if group_size < 1:
            raise ValueError("group_size must be >= 1")
        if batch_size % group_size != 0:
            raise ValueError("batch_size must be divisible by group_size")
        num_groups = batch_size // group_size
        group_inputs = self.sample_inputs(num_groups, device)
        group_answers = self.answer_tokens(group_inputs)
        inputs = group_inputs.repeat_interleave(group_size, dim=0)
        answers = group_answers.repeat_interleave(group_size)

        student_prompts = self.student_prompts(inputs)
        actions, actor_logp_a, student_obs = self._rollout_from_prompts(
            student, student_prompts)
        rewards = self.reward_from_actions(actions, answers)
        group_ids = torch.arange(
            num_groups, device=device).repeat_interleave(group_size)
        return PedagogyBatch(
            student_obs=student_obs,
            teacher_obs=student_obs,
            actions=actions,
            answers=answers,
            task_rewards=rewards,
            actor_logp_a=actor_logp_a,
            group_ids=group_ids,
            informative_group_rate=self._informative_group_rate(
                rewards, group_size),
        )

    def compute_student_logits(
            self,
            model: nn.Module,
            batch: PedagogyBatch) -> torch.Tensor:
        logits = model(batch.student_obs[:, :-1])
        return logits[:, self.student_prompt_len - 1:, :]

    def compute_teacher_logits(
            self,
            model: nn.Module,
            batch: PedagogyBatch) -> torch.Tensor:
        logits = model(batch.teacher_obs[:, :-1])
        return logits[:, self.teacher_prompt_len - 1:, :]

    @torch.no_grad()
    def _greedy_from_prompts(
            self,
            model: nn.Module,
            prompts: torch.Tensor) -> tuple[torch.Tensor, float]:
        was_training = model.training
        model.eval()
        prefix = prompts
        generated = []
        total_entropy = 0.0
        for _ in range(self.completion_len):
            logits = model(prefix)[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            total_entropy += (
                -(log_probs.exp() * log_probs).sum(dim=-1).sum().item())
            next_token = logits.argmax(dim=-1)
            generated.append(next_token)
            prefix = torch.cat([prefix, next_token.unsqueeze(1)], dim=1)
        model.train(was_training)
        actions = torch.stack(generated, dim=1)
        entropy = total_entropy / (prompts.size(0) * self.completion_len)
        return actions, entropy

    @torch.no_grad()
    def evaluate_student(
            self,
            model: nn.Module,
            device: torch.device,
            num_batches: int = 10,
            batch_size: int = 100) -> dict[str, float]:
        total_reward = 0.0
        total_entropy = 0.0
        total_count = 0
        for _ in range(num_batches):
            inputs = self.sample_inputs(batch_size, device)
            answers = self.answer_tokens(inputs)
            actions, entropy = self._greedy_from_prompts(
                model, self.student_prompts(inputs))
            total_reward += self.reward_from_actions(actions, answers).sum().item()
            total_entropy += entropy * batch_size
            total_count += batch_size
        reward = total_reward / total_count
        return {
            "test_error": 1.0 - reward,
            "reward": reward,
            "entropy": total_entropy / total_count,
        }

    @torch.no_grad()
    def evaluate_teacher(
            self,
            model: nn.Module,
            device: torch.device,
            num_batches: int = 10,
            batch_size: int = 100) -> dict[str, float]:
        total_reward = 0.0
        total_entropy = 0.0
        total_count = 0
        for _ in range(num_batches):
            inputs = self.sample_inputs(batch_size, device)
            answers = self.answer_tokens(inputs)
            actions, entropy = self._greedy_from_prompts(
                model, self.teacher_prompts(inputs, answers))
            total_reward += self.reward_from_actions(actions, answers).sum().item()
            total_entropy += entropy * batch_size
            total_count += batch_size
        reward = total_reward / total_count
        return {
            "teacher_test_error": 1.0 - reward,
            "teacher_reward_eval": reward,
            "teacher_entropy": total_entropy / total_count,
        }

