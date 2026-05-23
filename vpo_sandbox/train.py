"""Training entry point for the VPO sandbox."""

from __future__ import annotations

import argparse
import dataclasses
import time
from dataclasses import dataclass
from typing import get_type_hints

import pandas as pd
import torch

from . import losses as L
from .tasks import ParetoFrontTask


TASKS = {
    "pareto_front": lambda c: ParetoFrontTask(
        num_prompts=c.num_prompts,
        num_candidates=c.num_candidates,
        num_weight_samples=c.num_weight_samples,
    ),
}

LOSSES = {
    "ScalarGRPO": lambda c: L.ScalarGRPOLoss(clip_epsilon=c.clip_epsilon),
    "MultiRLVR": lambda c: L.MultiRLVRLoss(clip_epsilon=c.clip_epsilon),
    "VPOGRPO": lambda c: L.VPOGRPOLoss(clip_epsilon=c.clip_epsilon),
}


@dataclass
class Config:
    task: str = "pareto_front"
    method: str = "VPOGRPO"
    num_steps: int = 500
    batch_size: int = 128
    group_size: int = 8
    inner_epochs: int = 4
    lr: float = 3e-3
    eval_every: int = 20
    num_seeds: int = 3
    seed: int = 0
    output: str = "vpo_results.csv"
    verbose: bool = True
    hidden: int = 32
    num_prompts: int = 8
    num_candidates: int = 3
    num_weight_samples: int = 16
    clip_epsilon: float = 0.2


def validate_config(config: Config) -> None:
    if config.task not in TASKS:
        raise ValueError(f"unknown VPO task: {config.task}")
    if config.method not in LOSSES:
        raise ValueError(f"unknown VPO method: {config.method}")
    if config.num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.group_size <= 1:
        raise ValueError("group_size must be > 1")
    if config.batch_size % config.group_size != 0:
        raise ValueError("batch_size must be divisible by group_size")
    if config.inner_epochs < 1:
        raise ValueError("inner_epochs must be >= 1")
    if config.lr <= 0:
        raise ValueError("lr must be > 0")
    if config.eval_every < 1:
        raise ValueError("eval_every must be >= 1")
    if config.num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")
    if config.hidden < 1:
        raise ValueError("hidden must be >= 1")
    if config.num_prompts < 1:
        raise ValueError("num_prompts must be >= 1")
    if config.num_candidates < 1:
        raise ValueError("num_candidates must be >= 1")
    if config.num_weight_samples < 1:
        raise ValueError("num_weight_samples must be >= 1")
    if config.clip_epsilon < 0:
        raise ValueError("clip_epsilon must be non-negative")
    if config.method == "ScalarGRPO" and config.num_candidates != 1:
        raise ValueError("ScalarGRPO requires num_candidates=1")
    if config.method in {"MultiRLVR", "VPOGRPO"} and config.num_candidates <= 1:
        raise ValueError(f"{config.method} requires num_candidates > 1")


def train_one_seed(
        task: ParetoFrontTask,
        loss_fn,
        model: torch.nn.Module,
        config: Config,
        seed: int,
        device: torch.device) -> list[dict]:
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    rows = []

    for step in range(config.num_steps):
        eval_due = step % config.eval_every == 0
        if eval_due:
            eval_metrics = task.evaluate(model, device)

        batch = task.sample_batch(
            model, config.batch_size, config.group_size, device)
        for _ in range(config.inner_epochs):
            logits = task.compute_logits(model, batch)
            loss, metrics = loss_fn(logits, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if eval_due:
            row = {"step": step, "loss": loss.item(), **metrics, **eval_metrics}
            rows.append(row)
            if config.verbose and step % (config.eval_every * 10) == 0:
                print(
                    f"  step {step:5d} "
                    f"best_at_9={eval_metrics['best_at_9']:.4f} "
                    f"diversity={eval_metrics['pool_diversity_l1']:.4f} "
                    f"loss={loss.item():.4f}"
                )

    rows.append({"step": config.num_steps, **task.evaluate(model, device)})
    return rows


def run_config(config: Config) -> pd.DataFrame:
    validate_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = TASKS[config.task](config)
    all_rows = []
    for i in range(config.num_seeds):
        seed = config.seed + i
        if config.verbose:
            print(f"{config.method} task={config.task} seed={seed}")
        torch.manual_seed(seed)
        model = task.make_model(hidden=config.hidden)
        loss_fn = LOSSES[config.method](config)
        rows = train_one_seed(task, loss_fn, model, config, seed, device)
        for row in rows:
            row.update({"seed": seed, "method": config.method, "task": config.task})
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(config.output, index=False)
    return pd.DataFrame(all_rows)


TYPE_MAP = {int: int, float: float, str: str, bool: bool}


def parse_bool(text: str) -> bool:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    type_hints = get_type_hints(Config)
    for field in dataclasses.fields(Config):
        field_type = type_hints[field.name]
        if field_type is bool:
            parser.add_argument(
                f"--{field.name}",
                type=parse_bool,
                default=field.default,
                metavar="BOOL",
            )
        else:
            parser.add_argument(
                f"--{field.name}",
                type=TYPE_MAP.get(field_type, str),
                default=field.default,
            )
    args = parser.parse_args()
    config = Config(**{
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(Config)
    })

    t0 = time.time()
    df = run_config(config)
    df.to_csv(config.output, index=False)
    print(f"Saved {len(df)} rows to {config.output} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
