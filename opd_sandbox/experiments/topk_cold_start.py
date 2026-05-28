"""Test whether a full-vocab OPD warmup makes top-k truncation safe."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
import plotnine as gg
import torch

from .. import losses as L
from ..cli_utils import add_dataclass_args, config_from_args
from ..tasks import FormatAnswerTask, ReversalTask
from .topk_stability import parse_top_ks


TASKS = {
    "reversal": lambda c: ReversalTask(
        vocab_size=c.vocab_size,
        seq_len=c.seq_len,
        teacher_epsilon=c.teacher_epsilon,
    ),
    "format_answer": lambda c: FormatAnswerTask(
        vocab_size=c.vocab_size,
        seq_len=c.seq_len,
        teacher_epsilon=c.teacher_epsilon,
    ),
}


@dataclass
class Config:
    task: str = "reversal"
    top_ks: str = "1,2,4"
    num_steps: int = 300
    warmup_steps: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    eval_every: int = 20
    num_seeds: int = 3
    seed: int = 0
    verbose: bool = True
    vocab_size: int = 8
    seq_len: int = 8
    d_model: int = 64
    nhead: int = 2
    num_layers: int = 2
    teacher_epsilon: float = 1e-3
    overlap_k: int = 4
    output_dir: str = "opd_results"


def validate_config(config: Config) -> list[int]:
    top_ks = parse_top_ks(config.top_ks)
    if config.task not in TASKS:
        raise ValueError(f"unknown OPD task: {config.task}")
    if config.num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if config.warmup_steps >= config.num_steps:
        raise ValueError("warmup_steps must be smaller than num_steps")
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
    if config.teacher_epsilon <= 0 or config.teacher_epsilon >= 1:
        raise ValueError("teacher_epsilon must be in (0, 1)")
    if config.overlap_k < 1:
        raise ValueError("overlap_k must be >= 1")
    return top_ks


def make_model(task, config: Config) -> torch.nn.Module:
    return task.make_model(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
    )


def phase_for_step(variant: str, step: int, warmup_steps: int) -> str:
    if variant.startswith("warm_topk") and step < warmup_steps:
        return "warmup_full_vocab"
    return variant


def loss_for_phase(phase: str, top_k: int | None):
    if phase == "full_vocab_rkl" or phase == "warmup_full_vocab":
        return L.OPDReverseKLLoss()
    if phase.startswith("cold_topk") or phase.startswith("warm_topk"):
        if top_k is None:
            raise ValueError("top-k phase requires top_k")
        return L.OPDTopKReverseKLLoss(k=top_k)
    raise ValueError(f"unknown phase: {phase}")


def train_one_seed(config: Config, variant: str, top_k: int | None,
                   seed: int, device: torch.device) -> list[dict]:
    torch.manual_seed(seed)
    task = TASKS[config.task](config)
    model = make_model(task, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    rows = []

    for step in range(config.num_steps):
        phase = phase_for_step(variant, step, config.warmup_steps)
        eval_due = step % config.eval_every == 0
        if eval_due:
            eval_metrics = task.evaluate(model, device)

        batch = task.sample_batch(model, config.batch_size, device)
        logits = task.compute_logits(model, batch)
        loss_fn = loss_for_phase(phase, top_k)
        loss, metrics = loss_fn(logits, batch)
        diagnostics = L.opd_diagnostics(logits.detach(), batch, config.overlap_k)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if eval_due:
            rows.append({
                "step": step,
                "phase": phase,
                "loss": loss.item(),
                **metrics,
                **diagnostics,
                **eval_metrics,
            })
            if config.verbose and step % (config.eval_every * 10) == 0:
                print(
                    f"{variant} seed={seed} step={step} "
                    f"phase={phase} test_error={eval_metrics['test_error']:.4f}"
                )

    rows.append({
        "step": config.num_steps,
        "phase": phase_for_step(variant, config.num_steps, config.warmup_steps),
        **task.evaluate(model, device),
    })
    return rows


def run_variant(config: Config, variant: str, top_k: int | None,
                device: torch.device, output_dir: Path) -> pd.DataFrame:
    rows = []
    for seed_offset in range(config.num_seeds):
        seed = config.seed + seed_offset
        seed_rows = train_one_seed(config, variant, top_k, seed, device)
        for row in seed_rows:
            row.update({
                "seed": seed,
                "variant": variant,
                "top_k": top_k or 0,
                "task": config.task,
            })
        rows.extend(seed_rows)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"topk_cold_start_{variant}.csv", index=False)
    return df


def run_experiment(config: Config) -> pd.DataFrame:
    top_ks = validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = [run_variant(config, "full_vocab_rkl", None, device, output_dir)]
    for k in top_ks:
        frames.append(run_variant(config, f"cold_topk_k{k}", k, device, output_dir))
    for k in top_ks:
        frames.append(run_variant(config, f"warm_topk_k{k}", k, device, output_dir))

    combined = pd.concat(frames, ignore_index=True)
    combined_path = output_dir / "topk_cold_start.csv"
    combined.to_csv(combined_path, index=False)
    plot_results(combined, output_dir / "topk_cold_start.png", config.overlap_k)
    return combined


def plot_results(df: pd.DataFrame, path: Path, overlap_k: int):
    metrics = [
        ("test_error", "test error"),
        ("diag_top1_agreement", "top-1 agreement"),
        (f"diag_overlap_at_{overlap_k}", f"overlap@{overlap_k}"),
        ("diag_reverse_kl", "reverse KL"),
    ]
    rows = []
    for metric, label in metrics:
        if metric not in df.columns:
            continue
        subset = df[["step", "seed", "variant", metric]].dropna()
        for record in subset.to_dict("records"):
            rows.append({
                "step": record["step"],
                "seed": record["seed"],
                "variant": record["variant"],
                "metric": label,
                "value": record[metric],
            })
    if not rows:
        raise ValueError("top-k cold-start run has no plottable rows")

    plot_df = pd.DataFrame(rows)
    agg = (
        plot_df
        .groupby(["metric", "variant", "step"])
        .value
        .agg(["mean", "std"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    agg["ymin"] = agg["mean"] - agg["std"]
    agg["ymax"] = agg["mean"] + agg["std"]

    p = (
        gg.ggplot(agg, gg.aes(x="step", y="mean", color="variant"))
        + gg.geom_line(size=1)
        + gg.geom_ribbon(
            gg.aes(ymin="ymin", ymax="ymax", fill="variant"),
            alpha=0.10,
        )
        + gg.facet_wrap("metric", scales="free_y", ncol=1)
        + gg.ylab("")
        + gg.theme_bw(base_size=11)
        + gg.theme(figure_size=(8, 10))
    )
    p.save(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataclass_args(parser, Config)
    args = parser.parse_args()
    config = config_from_args(Config, args)
    df = run_experiment(config)
    print(f"Saved {len(df)} rows to {Path(config.output_dir) / 'topk_cold_start.csv'}")


if __name__ == "__main__":
    main()
