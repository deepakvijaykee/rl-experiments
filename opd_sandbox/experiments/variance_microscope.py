"""Measure gradient-estimator variance for OPD-style objectives.

This experiment is a scoped mechanism test, not a faithful MiniLLM
implementation. It compares three estimators on the same student rollouts:

- `sequence_pg`: sampled cumulative-return score estimator.
- `token_pg`: sampled one-step score estimator, equivalent to gamma=0 credit.
- `full_vocab_rkl`: exact per-token KL(pi_student || pi_teacher).

The output CSV reports variance of a fixed random projection of the gradient,
plus gradient-norm variance, as the rollout horizon changes.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
import plotnine as gg
import torch
import torch.nn.functional as F

from rl_sandbox.models import CausalTransformer

from ..cli_utils import add_dataclass_args, config_from_args, parse_int_list
from ..losses import gather_log_probs
from ..tasks import OPDBatch, ReversalTask


Estimator = Callable[[torch.Tensor, OPDBatch], torch.Tensor]


@dataclass
class Config:
    horizons: str = "4,8,16,32"
    num_batches: int = 40
    batch_size: int = 16
    num_seeds: int = 3
    seed: int = 0
    vocab_size: int = 2
    d_model: int = 32
    nhead: int = 2
    num_layers: int = 1
    teacher_epsilon: float = 1e-3
    output_dir: str = "opd_results"
    verbose: bool = True


def parse_horizons(text: str) -> list[int]:
    return parse_int_list(text, "horizons")


def validate_config(config: Config):
    horizons = parse_horizons(config.horizons)
    if config.num_batches < 2:
        raise ValueError("num_batches must be >= 2 for variance estimates")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")
    if config.vocab_size < 2:
        raise ValueError("vocab_size must be >= 2")
    if config.d_model < 1:
        raise ValueError("d_model must be >= 1")
    if config.nhead < 1:
        raise ValueError("nhead must be >= 1")
    if config.d_model % config.nhead != 0:
        raise ValueError("d_model must be divisible by nhead")
    if config.num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if config.teacher_epsilon <= 0 or config.teacher_epsilon >= 1:
        raise ValueError("teacher_epsilon must be in (0, 1)")
    return horizons


def sampled_rewards(logits: torch.Tensor, batch: OPDBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Return current sampled log-probs and sampled reverse-KL rewards."""
    log_probs = F.log_softmax(logits, dim=-1)
    logp_a = gather_log_probs(log_probs, batch.actions)
    reward = batch.teacher_logp_a - batch.actor_logp_a
    return logp_a, reward


def sequence_pg_loss(logits: torch.Tensor, batch: OPDBatch) -> torch.Tensor:
    """Sampled cumulative-return score estimator for reverse-KL reward."""
    logp_a, reward = sampled_rewards(logits, batch)
    returns = torch.flip(torch.cumsum(torch.flip(reward, dims=[1]), dim=1), dims=[1])
    return -(returns.detach() * logp_a).sum(dim=1).mean()


def token_pg_loss(logits: torch.Tensor, batch: OPDBatch) -> torch.Tensor:
    """Sampled one-step score estimator with gamma=0 credit."""
    logp_a, reward = sampled_rewards(logits, batch)
    return -(reward.detach() * logp_a).mean()


def full_vocab_rkl_loss(logits: torch.Tensor, batch: OPDBatch) -> torch.Tensor:
    """Exact full-vocabulary KL(pi_student || pi_teacher), averaged per token."""
    if logits.shape != batch.teacher_log_probs.shape:
        raise ValueError("teacher_log_probs must match logits shape")
    student_log_probs = F.log_softmax(logits.float(), dim=-1)
    student_probs = student_log_probs.exp()
    teacher_log_probs = batch.teacher_log_probs.float()
    kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    return kl.mean()


ESTIMATORS: dict[str, Estimator] = {
    "sequence_pg": sequence_pg_loss,
    "token_pg": token_pg_loss,
    "full_vocab_rkl": full_vocab_rkl_loss,
}


def make_model(task: ReversalTask, max_horizon: int, config: Config) -> CausalTransformer:
    return CausalTransformer(
        vocab_size=task.num_actions,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        max_seq_len=max_horizon * 2 + 1,
    )


def make_projection_direction(
        model: torch.nn.Module,
        seed: int,
        device: torch.device) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = []
    total_sq = 0.0
    for parameter in model.parameters():
        part = torch.randn(parameter.shape, generator=generator, dtype=torch.float32)
        total_sq += part.square().sum().item()
        direction.append(part.to(device))
    scale = math.sqrt(total_sq)
    return [part / scale for part in direction]


def grad_stats(model: torch.nn.Module,
               direction: list[torch.Tensor]) -> tuple[float, float]:
    projection = 0.0
    norm_sq = 0.0
    for parameter, part in zip(model.parameters(), direction):
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        projection += (grad * part).sum().item()
        norm_sq += grad.square().sum().item()
    return projection, math.sqrt(norm_sq)


def measure_batch(model: torch.nn.Module,
                  task: ReversalTask,
                  batch: OPDBatch,
                  direction: list[torch.Tensor]) -> list[dict[str, float | str]]:
    rows = []
    for name, loss_fn in ESTIMATORS.items():
        model.zero_grad(set_to_none=True)
        logits = task.compute_logits(model, batch)
        loss = loss_fn(logits, batch)
        if not torch.isfinite(loss):
            raise ValueError(f"{name} produced non-finite loss")
        loss.backward()
        projection, grad_norm = grad_stats(model, direction)
        rows.append({
            "estimator": name,
            "loss": loss.item(),
            "projection": projection,
            "grad_norm": grad_norm,
        })
    model.zero_grad(set_to_none=True)
    return rows


def summarize(raw_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_rows)
    grouped = (
        df
        .groupby(["horizon", "seed", "estimator", "batch_size"])
        .agg(
            num_batches=("batch_index", "count"),
            loss_mean=("loss", "mean"),
            projection_mean=("projection", "mean"),
            projection_var=("projection", "var"),
            grad_norm_mean=("grad_norm", "mean"),
            grad_norm_var=("grad_norm", "var"),
        )
        .reset_index()
    )
    grouped["projection_std"] = grouped["projection_var"].pow(0.5)
    grouped["grad_norm_std"] = grouped["grad_norm_var"].pow(0.5)
    return grouped


def run_experiment(config: Config) -> pd.DataFrame:
    horizons = validate_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_horizon = max(horizons)
    raw_rows = []

    for seed_offset in range(config.num_seeds):
        seed = config.seed + seed_offset
        torch.manual_seed(seed)
        base_task = ReversalTask(
            vocab_size=config.vocab_size,
            seq_len=max_horizon,
            teacher_epsilon=config.teacher_epsilon,
        )
        model = make_model(base_task, max_horizon, config).to(device)
        direction = make_projection_direction(model, seed + 10_000, device)
        model.eval()

        for horizon in horizons:
            task = ReversalTask(
                vocab_size=config.vocab_size,
                seq_len=horizon,
                teacher_epsilon=config.teacher_epsilon,
            )
            if config.verbose:
                print(f"seed={seed} horizon={horizon}")
            for batch_index in range(config.num_batches):
                batch = task.sample_batch(model, config.batch_size, device)
                for row in measure_batch(model, task, batch, direction):
                    row.update({
                        "horizon": horizon,
                        "seed": seed,
                        "batch_index": batch_index,
                        "batch_size": config.batch_size,
                    })
                    raw_rows.append(row)

    return summarize(raw_rows)


def plot_results(df: pd.DataFrame, path: Path):
    if df.empty:
        raise ValueError("variance microscope has no rows to plot")
    plot_df = (
        df
        .groupby(["horizon", "estimator"])
        .projection_var
        .mean()
        .reset_index()
    )
    p = (
        gg.ggplot(plot_df, gg.aes(x="horizon", y="projection_var", color="estimator"))
        + gg.geom_line(size=1)
        + gg.geom_point(size=2)
        + gg.scale_x_log10()
        + gg.scale_y_log10()
        + gg.xlab("rollout horizon")
        + gg.ylab("gradient projection variance")
        + gg.theme_bw(base_size=12)
        + gg.theme(figure_size=(7, 4.5))
    )
    p.save(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataclass_args(parser, Config)
    args = parser.parse_args()
    config = config_from_args(Config, args)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = run_experiment(config)
    csv_path = output_dir / "variance_microscope.csv"
    plot_path = output_dir / "variance_microscope.png"
    df.to_csv(csv_path, index=False)
    plot_results(df, plot_path)
    print(f"Saved {len(df)} rows to {csv_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
