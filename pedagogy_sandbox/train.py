"""Training loop for the Pedagogical RL toy sandbox."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import time
from dataclasses import dataclass
from typing import get_type_hints

import pandas as pd
import torch

from . import losses as L
from .tasks import HintedArithmeticTask


METHODS = {"PedagogicalRL", "TeacherRL", "StudentGRPO"}
TYPE_MAP = {int: int, float: float, str: str, bool: bool}


@dataclass
class Config:
    method: str = "PedagogicalRL"
    num_steps: int = 300
    batch_size: int = 96
    group_size: int = 8
    teacher_steps: int = 1
    student_steps: int = 1
    teacher_lr: float = 1e-3
    student_lr: float = 1e-3
    eval_every: int = 20
    num_seeds: int = 3
    seed: int = 0
    output: str = "pedagogy_results.csv"
    verbose: bool = True
    content_vocab_size: int = 5
    completion_len: int = 3
    d_model: int = 64
    nhead: int = 2
    num_layers: int = 2
    clip_epsilon: float = 0.2
    spike_beta: float = 5.0
    learnability_lambda: float = 0.5
    gate_kappa: float = 4.0
    gate_gamma: float = -2.5
    max_grad_norm: float = 1.0


def validate_config(config: Config):
    if config.method not in METHODS:
        raise ValueError(f"unknown method: {config.method}")
    if config.num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.group_size <= 1:
        raise ValueError("group_size must be > 1")
    if config.batch_size % config.group_size != 0:
        raise ValueError("batch_size must be divisible by group_size")
    if config.teacher_steps < 0:
        raise ValueError("teacher_steps must be non-negative")
    if config.student_steps < 1:
        raise ValueError("student_steps must be >= 1")
    if config.teacher_lr <= 0:
        raise ValueError("teacher_lr must be > 0")
    if config.student_lr <= 0:
        raise ValueError("student_lr must be > 0")
    if config.eval_every < 1:
        raise ValueError("eval_every must be >= 1")
    if config.num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")
    if config.content_vocab_size < 2:
        raise ValueError("content_vocab_size must be >= 2")
    if config.completion_len < 1:
        raise ValueError("completion_len must be >= 1")
    if config.d_model < 1:
        raise ValueError("d_model must be >= 1")
    if config.nhead < 1:
        raise ValueError("nhead must be >= 1")
    if config.d_model % config.nhead != 0:
        raise ValueError("d_model must be divisible by nhead")
    if config.num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if config.clip_epsilon < 0:
        raise ValueError("clip_epsilon must be non-negative")
    if config.spike_beta <= 0:
        raise ValueError("spike_beta must be > 0")
    if config.learnability_lambda < 0:
        raise ValueError("learnability_lambda must be non-negative")
    if config.gate_kappa <= 0:
        raise ValueError("gate_kappa must be > 0")
    if config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be > 0")
    if config.method != "StudentGRPO" and config.teacher_steps < 1:
        raise ValueError("teacher_steps must be >= 1 for teacher-based methods")


def _update_model(
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        parameters,
        max_grad_norm: float):
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_grad_norm)
    optimizer.step()


def _student_grpo_step(
        task: HintedArithmeticTask,
        student: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Config,
        device: torch.device) -> dict[str, float]:
    batch = task.sample_student_rollouts(
        student, config.batch_size, config.group_size, device)
    logits = task.compute_student_logits(student, batch)
    loss, metrics = L.grpo_loss(
        logits,
        batch.actions,
        batch.actor_logp_a,
        batch.task_rewards,
        batch.group_ids,
        config.clip_epsilon,
    )
    _update_model(loss, optimizer, student.parameters(), config.max_grad_norm)
    metrics["mixed_group_rate"] = batch.informative_group_rate
    return {f"student_{key}": value for key, value in metrics.items()}


def _teacher_step(
        task: HintedArithmeticTask,
        teacher: torch.nn.Module,
        student: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Config,
        device: torch.device) -> dict[str, float]:
    batch = task.sample_teacher_rollouts(
        teacher, config.batch_size, config.group_size, device)
    with torch.no_grad():
        student_logits = task.compute_student_logits(student, batch)
        if config.method == "PedagogicalRL":
            rewards, reward_metrics = L.pedagogical_rewards(
                batch.task_rewards,
                student_logits,
                batch.actions,
                config.spike_beta,
                config.learnability_lambda,
            )
        elif config.method == "TeacherRL":
            rewards, reward_metrics = L.additive_teacher_rewards(
                batch.task_rewards,
                student_logits,
                batch.actions,
                config.learnability_lambda,
            )
        else:
            raise ValueError(f"teacher step is undefined for {config.method}")
    teacher_logits = task.compute_teacher_logits(teacher, batch)
    loss, update_metrics = L.grpo_loss(
        teacher_logits,
        batch.actions,
        batch.actor_logp_a,
        rewards,
        batch.group_ids,
        config.clip_epsilon,
    )
    _update_model(loss, optimizer, teacher.parameters(), config.max_grad_norm)
    metrics = {
        **reward_metrics,
        **{f"teacher_{key}": value for key, value in update_metrics.items()},
        "teacher_mixed_group_rate": batch.informative_group_rate,
    }
    return metrics


def _assimilation_step(
        task: HintedArithmeticTask,
        teacher: torch.nn.Module,
        student: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Config,
        device: torch.device) -> dict[str, float]:
    batch = task.sample_teacher_rollouts(
        teacher, config.batch_size, group_size=1, device=device)
    logits = task.compute_student_logits(student, batch)
    if config.method == "PedagogicalRL":
        loss, metrics = L.gated_imitation_loss(
            logits, batch.actions, config.gate_kappa, config.gate_gamma)
    elif config.method == "TeacherRL":
        loss, metrics = L.sft_loss(logits, batch.actions)
    else:
        raise ValueError(f"assimilation is undefined for {config.method}")
    _update_model(loss, optimizer, student.parameters(), config.max_grad_norm)
    metrics["teacher_sample_task_reward"] = batch.task_rewards.mean().item()
    return metrics


def train_one_seed(
        task: HintedArithmeticTask,
        config: Config,
        seed: int,
        device: torch.device) -> list[dict[str, float]]:
    torch.manual_seed(seed)
    base_model = task.make_model(config.d_model, config.nhead, config.num_layers)
    student = copy.deepcopy(base_model).to(device)
    teacher = copy.deepcopy(base_model).to(device)
    student_optimizer = torch.optim.Adam(student.parameters(), lr=config.student_lr)
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=config.teacher_lr)
    rows = []

    for step in range(config.num_steps):
        eval_due = step % config.eval_every == 0
        if eval_due:
            eval_metrics = task.evaluate_student(student, device)
            if config.method != "StudentGRPO":
                eval_metrics.update(task.evaluate_teacher(teacher, device))

        step_metrics: dict[str, float] = {}
        if config.method == "StudentGRPO":
            for _ in range(config.student_steps):
                step_metrics.update(_student_grpo_step(
                    task, student, student_optimizer, config, device))
        else:
            for _ in range(config.teacher_steps):
                step_metrics.update(_teacher_step(
                    task, teacher, student, teacher_optimizer, config, device))
            for _ in range(config.student_steps):
                step_metrics.update(_assimilation_step(
                    task, teacher, student, student_optimizer, config, device))

        if eval_due:
            row = {"step": step, **step_metrics, **eval_metrics}
            rows.append(row)
            if config.verbose and step % (config.eval_every * 10) == 0:
                print(
                    f"  step {step:5d} "
                    f"test_error={eval_metrics['test_error']:.4f}"
                )

    final_metrics = task.evaluate_student(student, device)
    if config.method != "StudentGRPO":
        final_metrics.update(task.evaluate_teacher(teacher, device))
    rows.append({"step": config.num_steps, **final_metrics})
    return rows


def run_config(config: Config) -> pd.DataFrame:
    validate_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = HintedArithmeticTask(
        content_vocab_size=config.content_vocab_size,
        completion_len=config.completion_len,
    )
    all_rows = []
    for seed_offset in range(config.num_seeds):
        seed = config.seed + seed_offset
        if config.verbose:
            print(f"{config.method} seed={seed}")
        rows = train_one_seed(task, config, seed, device)
        for row in rows:
            row.update({
                "seed": seed,
                "method": config.method,
                "task": task.name,
            })
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(config.output, index=False)
    return pd.DataFrame(all_rows)


def parse_bool(text: str) -> bool:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    type_hints = get_type_hints(Config)
    for field in dataclasses.fields(Config):
        field_type = type_hints[field.name]
        if field_type is bool:
            parser.add_argument(
                f"--{field.name}", type=parse_bool,
                default=field.default, metavar="BOOL")
        else:
            parser.add_argument(
                f"--{field.name}",
                type=TYPE_MAP.get(field_type, str),
                default=field.default)
    args = parser.parse_args()
    arg_values = vars(args)
    config = Config(**{
        field.name: arg_values[field.name]
        for field in dataclasses.fields(Config)
    })
    start = time.time()
    df = run_config(config)
    df.to_csv(config.output, index=False)
    print(f"Saved {len(df)} rows to {config.output} ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
