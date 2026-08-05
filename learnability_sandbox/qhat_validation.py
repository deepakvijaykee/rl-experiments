"""Calibration of teacher-forced reflex q̂ against sampled rollout success.

The theory's state variable is the per-task success probability ``q``. On the
SFT/RL vocabulary the rollout grammar generates a latent think block before
each move, so ``exp(-line_nll)`` from ``coordinate_eval`` is the *reflex*
success probability — the direct-move distribution with no thinking — while
the sampled rollout success from ``evaluate_checkpoint`` marginalizes over
think paths. If thinking carries computation, empirical success will exceed
the binomial envelope of the reflex prediction; if the think block is
ornamental, the two agree. Either answer calibrates the cheap instrument:
this comparison is the chess-internal version of the math-proxy question.

Inputs are one coordinate CSV (with ``line_nll``) and one or more per-puzzle
rollout CSVs from the same checkpoint, joined on (data_source, fen).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

BIN_COLORS = {  # categorical slots 1-4, assigned in fixed order
    "test_B1": "#2a78d6",
    "test_B2": "#eb6834",
    "test_B3": "#1baf7a",
    "test_B4": "#eda100",
    "test_B5": "#e87ba4",
}
SIGNIFICANCE = 1e-3


@dataclass(frozen=True)
class Config:
    coords: Path
    rollouts: tuple[Path, ...]
    output: Path
    figure: Path | None = None


def load_joined(config: Config) -> pd.DataFrame:
    coords = pd.read_csv(config.coords)
    if "line_nll" not in coords.columns:
        raise ValueError(f"{config.coords} lacks line_nll; rerun coordinate_eval")
    rollouts = pd.concat([pd.read_csv(p) for p in config.rollouts], ignore_index=True)
    joined = coords.merge(
        rollouts[["data_source", "fen", "board_success_count", "num_samples"]],
        on=["data_source", "fen"],
        validate="one_to_one",
    )
    joined["reflex_q"] = np.exp(-joined.line_nll)
    joined["empirical_p"] = joined.board_success_count / joined.num_samples
    # Binomial tail probabilities under the reflex prediction: how surprising
    # is the observed success count if rollouts sampled the reflex line?
    joined["p_above"] = binom.sf(
        joined.board_success_count - 1, joined.num_samples, joined.reflex_q
    )
    joined["p_below"] = binom.cdf(
        joined.board_success_count, joined.num_samples, joined.reflex_q
    )
    return joined


def summarize(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in [("all", joined), *joined.groupby("data_source")]:
        rows.append({
            "bin": name,
            "num_puzzles": len(frame),
            "mean_reflex_q": frame.reflex_q.mean(),
            "mean_empirical_p": frame.empirical_p.mean(),
            "spearman": frame.reflex_q.corr(frame.empirical_p, method="spearman"),
            "lift_significant": (frame.p_above < SIGNIFICANCE).mean(),
            "deficit_significant": (frame.p_below < SIGNIFICANCE).mean(),
        })
    return pd.DataFrame(rows)


def plot(joined: pd.DataFrame, figure_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    floor = 10 ** np.floor(np.log10(max(joined.reflex_q.min(), 1e-12)))
    grid = np.geomspace(floor, 1.0, 200)

    n_typical = int(joined.num_samples.median())
    left.fill_between(
        grid,
        binom.ppf(0.005, n_typical, grid) / n_typical,
        binom.ppf(0.995, n_typical, grid) / n_typical,
        color="#0b0b0b", alpha=0.08, linewidth=0,
        label=f"99% binomial envelope (n={n_typical})",
    )
    left.plot(grid, grid, color="#52514e", linewidth=1, linestyle="--")
    for bin_name, frame in joined.groupby("data_source"):
        left.scatter(
            frame.reflex_q, frame.empirical_p,
            s=9, alpha=0.45, linewidths=0,
            color=BIN_COLORS.get(bin_name, "#52514e"), label=bin_name,
        )
    left.set_xscale("log")
    left.set_xlabel("reflex success probability  exp(−line NLL)")
    left.set_ylabel("empirical rollout success rate")
    left.set_title("Per-puzzle: sampled success vs reflex prediction")
    left.legend(frameon=False, fontsize=8, loc="upper left")

    edges = np.quantile(np.log10(joined.reflex_q), np.linspace(0, 1, 11))
    groups = joined.groupby(
        pd.cut(np.log10(joined.reflex_q), np.unique(edges)), observed=True
    )
    centers = groups.reflex_q.mean()
    right.plot(grid, grid, color="#52514e", linewidth=1, linestyle="--")
    right.plot(
        centers, groups.empirical_p.mean(),
        color="#2a78d6", linewidth=2, marker="o", markersize=5,
        label="mean empirical per reflex-decile",
    )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("mean reflex success probability (decile)")
    right.set_ylabel("mean empirical success rate")
    right.set_title("Calibration: thinking lift is the gap above the diagonal")
    right.legend(frameon=False, fontsize=8, loc="upper left")

    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(True, which="major", color="#0b0b0b", alpha=0.08, linewidth=0.6)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)


def run_config(config: Config) -> pd.DataFrame:
    joined = load_joined(config)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(config.output, index=False)
    summary = summarize(joined)
    print(summary.to_string(index=False))
    if config.figure is not None:
        plot(joined, config.figure)
    return summary


def parse_config() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coords", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, default=None)
    arguments = parser.parse_args()
    return Config(
        coords=arguments.coords,
        rollouts=tuple(arguments.rollouts),
        output=arguments.output,
        figure=arguments.figure,
    )


def main() -> None:
    config = parse_config()
    started = time.time()
    run_config(config)
    print(f"Saved {config.output} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
