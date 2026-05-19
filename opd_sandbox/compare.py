"""Run the first OPD appendix comparison and save a plot."""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
import plotnine as gg

from .cli_utils import add_dataclass_args, config_from_args
from .train import Config, run_config


METHODS = ("OPDReverseKL", "OPDPG")


def run_comparison(config: Config, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for method in METHODS:
        method_config = dataclasses.replace(
            config,
            method=method,
            output=str(output_dir / f"{config.task}_{method}.csv"),
        )
        frames.append(run_config(method_config))
    combined = pd.concat(frames, ignore_index=True)
    combined_path = output_dir / f"{config.task}_opd_compare.csv"
    combined.to_csv(combined_path, index=False)
    plot_comparison(combined, output_dir / f"{config.task}_opd_compare.png")
    return combined


def plot_comparison(df: pd.DataFrame, path: Path):
    metrics = [
        ("test_error", "test error"),
        ("diag_reverse_kl", "reverse KL"),
        ("diag_top1_agreement", "top-1 agreement"),
    ]
    rows = []
    for metric, label in metrics:
        if metric not in df.columns:
            continue
        subset = df[["step", "method", "seed", metric]].dropna()
        for record in subset.to_dict("records"):
            rows.append({
                "step": record["step"],
                "method": record["method"],
                "metric": label,
                "value": record[metric],
            })
    if not rows:
        raise ValueError("comparison has no plottable metric rows")

    plot_df = pd.DataFrame(rows)
    agg = (
        plot_df
        .groupby(["metric", "method", "step"])
        .value
        .agg(["mean", "std"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    agg["ymin"] = agg["mean"] - agg["std"]
    agg["ymax"] = agg["mean"] + agg["std"]

    p = (
        gg.ggplot(agg, gg.aes(x="step", y="mean", color="method"))
        + gg.geom_line(size=1)
        + gg.geom_ribbon(
            gg.aes(ymin="ymin", ymax="ymax", fill="method"),
            alpha=0.12,
        )
        + gg.facet_wrap("metric", scales="free_y", ncol=1)
        + gg.ylab("")
        + gg.theme_bw(base_size=12)
        + gg.theme(figure_size=(8, 9))
    )
    p.save(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("opd_results"),
        help="directory for per-method CSVs, combined CSV, and plot",
    )
    skip = {"method", "output"}
    add_dataclass_args(parser, Config, skip=skip)
    args = parser.parse_args()
    df = run_comparison(config_from_args(Config, args, skip=skip), args.output_dir)
    print(
        f"Saved {len(df)} rows to "
        f"{args.output_dir / f'{args.task}_opd_compare.csv'}"
    )


if __name__ == "__main__":
    main()
