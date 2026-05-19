"""Compare top-k support choices under a graded soft teacher.

The earlier top-k stress tests use a smoothed one-hot oracle. That makes support
alignment easy to diagnose, but it also creates arbitrary ties among all wrong
tokens. This experiment uses `SoftReversalTask`, whose teacher prefers the
correct token most, nearby content tokens next, and the separator token last.
"""

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
from ..tasks import SoftReversalTask
from .topk_stability import parse_top_ks
from .topk_warmup_sweep import parse_warmup_steps


SUPPORTS = {"student", "teacher", "intersection"}


@dataclass
class Config:
    top_ks: str = "4"
    supports: str = "student,teacher,intersection"
    warmup_steps: str = "0,100,200,250"
    num_steps: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    eval_every: int = 10
    num_seeds: int = 3
    seed: int = 0
    verbose: bool = True
    vocab_size: int = 8
    seq_len: int = 8
    d_model: int = 64
    nhead: int = 2
    num_layers: int = 2
    teacher_temperature: float = 1.0
    special_token_weight: float = 0.02
    overlap_k: int = 4
    output_dir: str = "opd_results"


def parse_supports(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("supports must not be empty")
    invalid = sorted(set(values) - SUPPORTS)
    if invalid:
        raise argparse.ArgumentTypeError(
            "supports must contain only student, teacher, or intersection"
        )
    return sorted(set(values), key=values.index)


def validate_config(config: Config) -> tuple[list[int], list[str], list[int]]:
    top_ks = parse_top_ks(config.top_ks)
    supports = parse_supports(config.supports)
    warmups = parse_warmup_steps(config.warmup_steps)
    if config.num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if max(warmups) >= config.num_steps:
        raise ValueError("all warmup_steps must be smaller than num_steps")
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
    if config.teacher_temperature <= 0:
        raise ValueError("teacher_temperature must be > 0")
    if config.special_token_weight <= 0:
        raise ValueError("special_token_weight must be > 0")
    if config.overlap_k < 1:
        raise ValueError("overlap_k must be >= 1")
    return top_ks, supports, warmups


def make_task(config: Config) -> SoftReversalTask:
    return SoftReversalTask(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        teacher_temperature=config.teacher_temperature,
        special_token_weight=config.special_token_weight,
    )


def make_model(task: SoftReversalTask, config: Config) -> torch.nn.Module:
    return task.make_model(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
    )


def loss_for_step(step: int, warmup_steps: int, top_k: int | None,
                  support: str = "student"):
    if top_k is None or step < warmup_steps:
        return "full_vocab_rkl", L.OPDReverseKLLoss()
    return f"topk_{support}", L.OPDTopKReverseKLLoss(k=top_k, support=support)


def train_one_seed(
        config: Config,
        top_k: int | None,
        support: str,
        warmup_steps: int,
        seed: int,
        device: torch.device) -> list[dict]:
    torch.manual_seed(seed)
    task = make_task(config)
    model = make_model(task, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    rows = []

    for step in range(config.num_steps):
        phase, loss_fn = loss_for_step(step, warmup_steps, top_k, support)
        eval_due = step % config.eval_every == 0 or step == warmup_steps
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
            rows.append({
                "step": step,
                "phase": phase,
                "is_switch": top_k is not None and step == warmup_steps,
                "loss": loss.item(),
                **metrics,
                **diagnostics,
                **eval_metrics,
            })
            if config.verbose and step % (config.eval_every * 10) == 0:
                label = (
                    "full_vocab"
                    if top_k is None
                    else f"{support}, k={top_k}, warm={warmup_steps}"
                )
                print(
                    f"{label} seed={seed} step={step} phase={phase} "
                    f"test_error={eval_metrics['test_error']:.4f}"
                )

    rows.append({
        "step": config.num_steps,
        "phase": "full_vocab_rkl" if top_k is None else f"topk_{support}",
        "is_switch": False,
        **task.evaluate(model, device),
    })
    return rows


def run_variant(
        config: Config,
        top_k: int | None,
        support: str,
        warmup_steps: int,
        device: torch.device,
        output_dir: Path) -> pd.DataFrame:
    if top_k is None:
        variant = "full_vocab_rkl"
    else:
        variant = f"topk_{support}_k{top_k}_warm{warmup_steps}"
    rows = []
    for seed_offset in range(config.num_seeds):
        seed = config.seed + seed_offset
        seed_rows = train_one_seed(
            config, top_k, support, warmup_steps, seed, device)
        for row in seed_rows:
            row.update({
                "seed": seed,
                "variant": variant,
                "support": support if top_k is not None else "full_vocab",
                "top_k": top_k or 0,
                "warmup_steps": warmup_steps if top_k is not None else -1,
                "teacher_temperature": config.teacher_temperature,
                "special_token_weight": config.special_token_weight,
            })
        rows.extend(seed_rows)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"soft_teacher_topk_{variant}.csv", index=False)
    return df


def run_experiment(config: Config) -> pd.DataFrame:
    top_ks, supports, warmups = validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = [run_variant(config, None, "full_vocab", 0, device, output_dir)]
    for top_k in top_ks:
        for support in supports:
            for warmup_steps in warmups:
                frames.append(
                    run_variant(
                        config,
                        top_k,
                        support,
                        warmup_steps,
                        device,
                        output_dir,
                    )
                )

    combined = pd.concat(frames, ignore_index=True)
    combined_path = output_dir / "soft_teacher_topk.csv"
    combined.to_csv(combined_path, index=False)
    plot_results(combined, output_dir / "soft_teacher_topk.png")
    return combined


def plot_results(df: pd.DataFrame, path: Path):
    final = df[df["step"] == df["step"].max()].copy()
    final = final[final["top_k"] > 0]
    if final.empty:
        raise ValueError("soft-teacher top-k run has no top-k final rows")

    agg = (
        final
        .groupby(["top_k", "support", "warmup_steps"])
        .test_error
        .agg(["mean", "std"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    agg["top_k"] = "k=" + agg["top_k"].astype(str)
    agg["ymin"] = agg["mean"] - agg["std"]
    agg["ymax"] = agg["mean"] + agg["std"]

    p = (
        gg.ggplot(agg, gg.aes(x="warmup_steps", y="mean", color="support"))
        + gg.geom_line(size=1)
        + gg.geom_point(size=2)
        + gg.geom_ribbon(
            gg.aes(ymin="ymin", ymax="ymax", fill="support"),
            alpha=0.10,
        )
        + gg.facet_wrap("top_k", ncol=1)
        + gg.xlab("full-vocab warmup steps before top-k switch")
        + gg.ylab("final test error")
        + gg.theme_bw(base_size=12)
        + gg.theme(figure_size=(7.5, 4.8))
    )
    p.save(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataclass_args(parser, Config)
    args = parser.parse_args()
    config = config_from_args(Config, args)
    df = run_experiment(config)
    print(f"Saved {len(df)} rows to {Path(config.output_dir) / 'soft_teacher_topk.csv'}")


if __name__ == "__main__":
    main()
