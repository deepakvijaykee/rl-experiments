"""Shell decomposition of reflex-q̂ evolution across an RL checkpoint series.

The reservoir theory is diagonal in tasks: group-centered credit moves a
task only through its own mixed groups, so a task whose success probability
is far below ``1/(K · groups seen)`` statistically cannot have received
direct credit. Partition the panel by the *initial* (pre-RL) reflex q̂ into
shells — saturated, band, cold, deep-cold — and track each shell's q̂ across
released RL checkpoints. Band movement is what direct credit predicts;
coherent movement in the deep-cold shell is off-diagonal generalization
transfer, the correction term the diagonal theory omits.

Reflex q̂ is the direct-move line probability (``exp(-line_nll)``); on the
SFT/RL vocabulary its calibration against sampled success is measured by
``qhat_validation`` and read alongside this output.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SHELL_ORDER = ("deep_cold", "cold", "band", "saturated")
SHELL_COLORS = {
    "deep_cold": "#2a78d6",
    "cold": "#eb6834",
    "band": "#1baf7a",
    "saturated": "#eda100",
}


@dataclass(frozen=True)
class Config:
    coords: tuple[Path, ...]
    labels: tuple[str, ...]
    output: Path
    figure: Path | None = None
    group_size: int = 8
    band_low: float = 0.05
    band_high: float = 0.9
    deep_cold: float = 1e-3

    def __post_init__(self) -> None:
        if len(self.coords) != len(self.labels):
            raise ValueError("coords and labels must align")
        if len(self.coords) < 2:
            raise ValueError("need at least an initial and one later checkpoint")


def assign_shell(q: float, config: Config) -> str:
    if q >= config.band_high:
        return "saturated"
    if q >= config.band_low:
        return "band"
    if q >= config.deep_cold:
        return "cold"
    return "deep_cold"


def load_series(config: Config) -> pd.DataFrame:
    """One row per puzzle with a log10 reflex-q̂ column per checkpoint."""
    merged: pd.DataFrame | None = None
    for path, label in zip(config.coords, config.labels):
        frame = pd.read_csv(path)
        if "line_nll" not in frame.columns:
            raise ValueError(f"{path} lacks line_nll; rerun coordinate_eval")
        frame = frame[["data_source", "fen", "horizon", "line_nll"]].rename(
            columns={"line_nll": f"line_nll_{label}"}
        )
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(
                frame.drop(columns="horizon"),
                on=["data_source", "fen"],
                validate="one_to_one",
            )
    assert merged is not None
    for label in config.labels:
        merged[f"log10_q_{label}"] = -merged[f"line_nll_{label}"] / np.log(10)
    initial = config.labels[0]
    merged["shell"] = [
        assign_shell(q, config) for q in 10 ** merged[f"log10_q_{initial}"]
    ]
    return merged


def contrast_mass(q: np.ndarray, group_size: int) -> np.ndarray:
    return 1.0 - q**group_size - (1.0 - q) ** group_size


def summarize(series: pd.DataFrame, config: Config) -> pd.DataFrame:
    rows = []
    for shell, frame in series.groupby("shell"):
        for label in config.labels:
            log_q = frame[f"log10_q_{label}"]
            q = 10**log_q
            rows.append({
                "shell": shell,
                "checkpoint": label,
                "num_puzzles": len(frame),
                "median_log10_q": log_q.median(),
                "q25_log10_q": log_q.quantile(0.25),
                "q75_log10_q": log_q.quantile(0.75),
                "mean_q": q.mean(),
                "mean_contrast_mass": contrast_mass(q, config.group_size).mean(),
                "fraction_in_band_or_above": (q >= config.band_low).mean(),
            })
    summary = pd.DataFrame(rows)
    summary["shell"] = pd.Categorical(
        summary.shell, categories=SHELL_ORDER, ordered=True
    )
    return summary.sort_values(["shell", "checkpoint"]).reset_index(drop=True)


def plot(series: pd.DataFrame, summary: pd.DataFrame, config: Config) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    positions = np.arange(len(config.labels))

    for shell in SHELL_ORDER:
        frame = summary[summary.shell == shell].set_index("checkpoint").loc[
            list(config.labels)
        ]
        left.plot(
            positions, frame.median_log10_q,
            color=SHELL_COLORS[shell], linewidth=2, marker="o", markersize=5,
            label=f"{shell} (n={int(frame.num_puzzles.iloc[0])})",
        )
        left.fill_between(
            positions, frame.q25_log10_q, frame.q75_log10_q,
            color=SHELL_COLORS[shell], alpha=0.12, linewidth=0,
        )
    left.axhline(np.log10(config.band_low), color="#52514e",
                 linewidth=1, linestyle="--")
    left.annotate("band threshold", xy=(0.02, np.log10(config.band_low)),
                  xycoords=("axes fraction", "data"),
                  textcoords="offset points", xytext=(0, 4),
                  fontsize=8, color="#52514e")
    left.set_xticks(positions, config.labels)
    left.set_xlabel("checkpoint")
    left.set_ylabel("median log10 reflex q̂ (IQR band)")
    left.set_title("Shell evolution under RL (shells fixed at initial q̂)")
    left.legend(frameon=False, fontsize=8)

    bins = np.linspace(
        series[[f"log10_q_{label}" for label in config.labels]].min().min(),
        0.0,
        40,
    )
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#104281"]
    for index, label in enumerate(config.labels):
        color = ramp[min(index, len(ramp) - 1)]
        right.hist(
            series[f"log10_q_{label}"], bins=bins, histtype="step",
            linewidth=1.8, color=color, label=label,
        )
    right.set_xlabel("log10 reflex q̂")
    right.set_ylabel("puzzles")
    right.set_title("Reservoir histogram per checkpoint")
    right.legend(frameon=False, fontsize=8)

    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(True, which="major", color="#0b0b0b", alpha=0.08,
                  linewidth=0.6)
    fig.tight_layout()
    assert config.figure is not None
    config.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.figure)
    plt.close(fig)


def run_config(config: Config) -> pd.DataFrame:
    series = load_series(config)
    summary = summarize(series, config)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(
        config.output.with_name(config.output.stem + "_puzzles.csv"), index=False
    )
    summary.to_csv(config.output, index=False)
    print(summary.to_string(index=False))

    initial, final = config.labels[0], config.labels[-1]
    for shell in SHELL_ORDER:
        frame = series[series.shell == shell]
        if frame.empty:
            continue
        delta = frame[f"log10_q_{final}"] - frame[f"log10_q_{initial}"]
        moved = (delta > 1.0).mean()
        print(f"{shell}: median Δlog10 q̂ = {delta.median():+.3f}, "
              f"fraction moved >1 decade = {moved:.3f}")
    if config.figure is not None:
        plot(series, summary, config)
    return summary


def parse_config() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coords", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--band_low", type=float, default=0.05)
    parser.add_argument("--band_high", type=float, default=0.9)
    parser.add_argument("--deep_cold", type=float, default=1e-3)
    arguments = parser.parse_args()
    return Config(
        coords=tuple(arguments.coords),
        labels=tuple(arguments.labels),
        output=arguments.output,
        figure=arguments.figure,
        group_size=arguments.group_size,
        band_low=arguments.band_low,
        band_high=arguments.band_high,
        deep_cold=arguments.deep_cold,
    )


def main() -> None:
    config = parse_config()
    started = time.time()
    run_config(config)
    print(f"Saved {config.output} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
