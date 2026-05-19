"""Compare full-vocab, top-k, and sampled-token OPD training.

This is an appendix experiment for support truncation. It is deliberately
small: one toy task, one learner architecture, and a few method variants.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
import plotnine as gg

from ..cli_utils import add_dataclass_args, config_from_args, parse_int_list
from ..train import Config as TrainConfig
from ..train import run_config


@dataclass
class Config:
    task: str = "reversal"
    top_ks: str = "1,2,4"
    num_steps: int = 300
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
    clip_epsilon: float = 0.2
    overlap_k: int = 4
    output_dir: str = "opd_results"


def parse_top_ks(text: str) -> list[int]:
    return parse_int_list(text, "top_ks")


def validate_config(config: Config) -> list[int]:
    top_ks = parse_top_ks(config.top_ks)
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
    if config.teacher_epsilon <= 0 or config.teacher_epsilon >= 1:
        raise ValueError("teacher_epsilon must be in (0, 1)")
    if config.clip_epsilon < 0:
        raise ValueError("clip_epsilon must be non-negative")
    if config.overlap_k < 1:
        raise ValueError("overlap_k must be >= 1")
    return top_ks


def train_config(config: Config, method: str, output: Path,
                 top_k: int | None = None) -> TrainConfig:
    return TrainConfig(
        task=config.task,
        method=method,
        num_steps=config.num_steps,
        batch_size=config.batch_size,
        lr=config.lr,
        eval_every=config.eval_every,
        num_seeds=config.num_seeds,
        seed=config.seed,
        output=str(output),
        verbose=config.verbose,
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        teacher_epsilon=config.teacher_epsilon,
        clip_epsilon=config.clip_epsilon,
        overlap_k=config.overlap_k,
        top_k=top_k or config.overlap_k,
    )


def run_variant(config: Config, variant: str, method: str,
                output_dir: Path, top_k: int | None = None) -> pd.DataFrame:
    output = output_dir / f"topk_stability_{variant}.csv"
    df = run_config(train_config(config, method, output, top_k=top_k))
    df["variant"] = variant
    if top_k is not None:
        df["top_k"] = top_k
    else:
        df["top_k"] = 0
    df.to_csv(output, index=False)
    return df


def run_experiment(config: Config) -> pd.DataFrame:
    top_ks = validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        run_variant(config, "full_vocab_rkl", "OPDReverseKL", output_dir),
    ]
    for k in top_ks:
        frames.append(
            run_variant(
                config,
                f"topk_rkl_k{k}",
                "OPDTopKReverseKL",
                output_dir,
                top_k=k,
            )
        )
    frames.append(run_variant(config, "sampled_pg", "OPDPG", output_dir))

    combined = pd.concat(frames, ignore_index=True)
    combined_path = output_dir / "topk_stability.csv"
    combined.to_csv(combined_path, index=False)
    plot_results(combined, output_dir / "topk_stability.png", config.overlap_k)
    return combined


def plot_results(df: pd.DataFrame, path: Path, overlap_k: int):
    metrics = [
        ("test_error", "test error"),
        ("diag_reverse_kl", "reverse KL"),
        ("diag_top1_agreement", "top-1 agreement"),
        (f"diag_overlap_at_{overlap_k}", f"overlap@{overlap_k}"),
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
        raise ValueError("top-k stability run has no plottable rows")

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
    print(f"Saved {len(df)} rows to {Path(config.output_dir) / 'topk_stability.csv'}")


if __name__ == "__main__":
    main()
