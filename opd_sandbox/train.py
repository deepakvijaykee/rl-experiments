"""Training entry point for the OPD sandbox."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import pandas as pd
import torch

from . import losses as L
from .cli_utils import add_dataclass_args, config_from_args
from .tasks import FormatAnswerTask, ReversalTask, SoftReversalTask


TASKS = {
    "reversal": lambda c: ReversalTask(
        vocab_size=c.vocab_size,
        seq_len=c.seq_len,
        teacher_epsilon=c.teacher_epsilon,
    ),
    "soft_reversal": lambda c: SoftReversalTask(
        vocab_size=c.vocab_size,
        seq_len=c.seq_len,
        teacher_temperature=c.teacher_temperature,
        special_token_weight=c.special_token_weight,
    ),
    "format_answer": lambda c: FormatAnswerTask(
        vocab_size=c.vocab_size,
        seq_len=c.seq_len,
        teacher_epsilon=c.teacher_epsilon,
    ),
}

LOSSES = {
    "OPDReverseKL": lambda c: L.OPDReverseKLLoss(),
    "OPDTopKReverseKL": lambda c: L.OPDTopKReverseKLLoss(
        k=c.top_k,
        support=c.top_k_support,
    ),
    "OPDPG": lambda c: L.OPDPGLoss(clip_epsilon=c.clip_epsilon),
}


@dataclass
class Config:
    task: str = "reversal"
    method: str = "OPDReverseKL"
    num_steps: int = 500
    batch_size: int = 64
    lr: float = 1e-3
    eval_every: int = 20
    num_seeds: int = 3
    seed: int = 0
    output: str = "opd_results.csv"
    verbose: bool = True
    vocab_size: int = 2
    seq_len: int = 10
    d_model: int = 64
    nhead: int = 2
    num_layers: int = 2
    teacher_epsilon: float = 1e-3
    clip_epsilon: float = 0.2
    overlap_k: int = 2
    top_k: int = 2
    top_k_support: str = "student"
    teacher_temperature: float = 1.0
    special_token_weight: float = 0.02


def validate_config(config: Config):
    if config.task not in TASKS:
        raise ValueError(f"unknown OPD task: {config.task}")
    if config.method not in LOSSES:
        raise ValueError(f"unknown OPD method: {config.method}")
    if config.num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.lr <= 0:
        raise ValueError("lr must be > 0")
    if config.eval_every < 1:
        raise ValueError("eval_every must be >= 1")
    if config.num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")
    if config.vocab_size < 2:
        raise ValueError("vocab_size must be >= 2")
    if config.seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if config.d_model < 1:
        raise ValueError("d_model must be >= 1")
    if config.nhead < 1:
        raise ValueError("nhead must be >= 1")
    if config.d_model % config.nhead != 0:
        raise ValueError("d_model must be divisible by nhead")
    if config.num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if config.task != "soft_reversal" and (
            config.teacher_epsilon <= 0 or config.teacher_epsilon >= 1):
        raise ValueError("teacher_epsilon must be in (0, 1)")
    if config.clip_epsilon < 0:
        raise ValueError("clip_epsilon must be non-negative")
    if config.overlap_k < 1:
        raise ValueError("overlap_k must be >= 1")
    if config.top_k < 1:
        raise ValueError("top_k must be >= 1")
    if config.top_k_support not in {"student", "teacher", "intersection"}:
        raise ValueError("top_k_support must be student, teacher, or intersection")
    if config.teacher_temperature <= 0:
        raise ValueError("teacher_temperature must be > 0")
    if config.special_token_weight <= 0:
        raise ValueError("special_token_weight must be > 0")


def train_one_seed(task, loss_fn, model, config: Config,
                   seed: int, device: torch.device) -> list[dict]:
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    rows = []

    for step in range(config.num_steps):
        eval_due = step % config.eval_every == 0
        if eval_due:
            eval_metrics = task.evaluate(model, device)

        batch = task.sample_batch(model, config.batch_size, device)
        logits = task.compute_logits(model, batch)
        loss, metrics = loss_fn(logits, batch)
        diagnostics = L.opd_diagnostics(logits.detach(), batch, config.overlap_k)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if eval_due:
            row = {
                "step": step,
                "loss": loss.item(),
                **metrics,
                **diagnostics,
                **eval_metrics,
            }
            rows.append(row)
            if config.verbose and step % (config.eval_every * 10) == 0:
                print(
                    f"  step {step:5d} "
                    f"test_error={eval_metrics['test_error']:.4f} "
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
        model = task.make_model(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
        )
        loss_fn = LOSSES[config.method](config)
        rows = train_one_seed(task, loss_fn, model, config, seed, device)
        for row in rows:
            row.update({"seed": seed, "method": config.method, "task": config.task})
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(config.output, index=False)
    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataclass_args(parser, Config)
    args = parser.parse_args()
    config = config_from_args(Config, args)

    t0 = time.time()
    df = run_config(config)
    df.to_csv(config.output, index=False)
    print(f"Saved {len(df)} rows to {config.output} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
