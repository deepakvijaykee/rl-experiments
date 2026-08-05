"""Claim-3 test and coarse q-histogram deconvolution on the released curves.

The paper's collected run grid (`rl_curve_points.csv`) reports pass@k for
k in {1, 4, 8, 16} at every eval step of every RL run, all trained at group
size K=8. pass@k is a moment transform of the task-level success
distribution, ``pass@k = 1 - E[(1-q)^k]``, so the published aggregates
carry a coarse image of the q-histogram and its evolution — no model
evaluation required.

Three measurements per run:

1. **Gain profile** (claim 3): the change in pass@k from the first to the
   last eval step, as a function of k. Band-limited conversion at K=8
   predicts gains that shrink with k and vanish within noise at k=16,
   because pass@16 headroom is concentrated in tasks too cold for groups
   of 8 to produce contrast.
2. **Tail-difference depletion**: ``pass@16 - pass@8`` isolates the mass
   with q roughly in the half-decade below the band. Diagonal conversion
   only drains it (its upper edge converts, nothing refills from below);
   broad off-diagonal generalization would refill it from the cold side.
   Its trend over training is therefore a transfer diagnostic.
3. **Shell deconvolution**: nonnegative least squares on atoms at fixed q
   values recovers shell masses from the four moments (with a zero atom
   absorbing frozen mass). Four moments cannot localize structure finely —
   treat the output as shell-level, and check the gain profile directly
   before trusting any finer read.

The per-puzzle binomial standard error uses the panel size implied by
``rl_n_rows``; differences across steps are treated as unpaired, which
overstates the error (the panel is fixed), so significance calls are
conservative.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pandas as pd
from scipy.optimize import nnls

TYPE_MAP = {int: int, float: float, str: str}

PASS_COLUMNS = {1: "rl_pass_at_1", 4: "rl_pass_at_4",
                8: "rl_pass_at_8", 16: "rl_pass_at_16"}
K_VALUES = (1, 4, 8, 16)
# Shell atoms: frozen, deep cold, cold edge, band, saturated.
SHELL_ATOMS = {"frozen": 0.0, "deep_cold": 0.002, "cold": 0.05,
               "band": 0.3, "saturated": 0.95}
SIZE_COLORS = {"20m": "#2a78d6", "50m": "#eb6834",
               "200m": "#1baf7a", "680m": "#eda100"}


@dataclass(frozen=True)
class Config:
    input_path: str = (
        "data/chess_pre_to_post/collected_pre_to_post_data/rl_curve_points.csv"
    )
    output_dir: str = "results/released_curves"
    figure: str = "learnability_sandbox/figures/released_curve_claim3.png"


def deconvolve_shells(pass_at: dict[int, float]) -> dict[str, float]:
    """Shell masses from the four pass@k moments via constrained NNLS."""
    atoms = np.array(list(SHELL_ATOMS.values()))
    design = np.array([
        [1.0 - (1.0 - q) ** k for q in atoms] for k in K_VALUES
    ])
    target = np.array([pass_at[k] for k in K_VALUES])
    # Augment with a heavily weighted total-mass row so weights sum to one;
    # the zero atom gives the frozen remainder somewhere to live.
    design = np.vstack([design, 10.0 * np.ones(len(atoms))])
    target = np.append(target, 10.0)
    weights, _ = nnls(design, target)
    return dict(zip(SHELL_ATOMS, weights))


def analyze_run(frame: pd.DataFrame) -> dict:
    """Gain profile, tail-difference trend, and slope for one run."""
    frame = frame.sort_values("rl_step")
    first, last = frame.iloc[0], frame.iloc[-1]
    num_puzzles = int(last.rl_n_rows) // 16
    row = {
        "run_id": last.run_id,
        "size": last.size_str,
        "alpha": last.alpha_pretrain,
        "pretrain_loss": last.pretrain_loss_eval,
        "first_step": int(first.rl_step),
        "last_step": int(last.rl_step),
        "num_puzzles": num_puzzles,
    }
    for k, column in PASS_COLUMNS.items():
        gain = last[column] - first[column]
        # Unpaired binomial SE at each end; conservative for a fixed panel.
        variance = sum(
            point[column] * (1.0 - point[column]) / num_puzzles
            for point in (first, last)
        )
        row[f"gain_pass_at_{k}"] = gain
        row[f"gain_se_{k}"] = float(np.sqrt(variance))
    tail = frame[PASS_COLUMNS[16]] - frame[PASS_COLUMNS[8]]
    tail_slope = np.polyfit(np.log(frame.rl_step), tail, 1)[0]
    row["tail_difference_first"] = tail.iloc[0]
    row["tail_difference_last"] = tail.iloc[-1]
    row["tail_slope_per_log_step"] = tail_slope
    row["reward_slope_per_log_step"] = np.polyfit(
        np.log(frame.rl_step), frame.rl_pass_at_1, 1
    )[0]
    return row


def shell_evolution(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, point in frame.sort_values("rl_step").iterrows():
        shells = deconvolve_shells(
            {k: point[column] for k, column in PASS_COLUMNS.items()}
        )
        rows.append({
            "run_id": point.run_id,
            "size": point.size_str,
            "alpha": point.alpha_pretrain,
            "rl_step": point.rl_step,
            **shells,
        })
    return pd.DataFrame(rows)


def plot(
    runs: pd.DataFrame,
    curves: pd.DataFrame,
    shells: pd.DataFrame,
    focus_run: str,
    figure_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(14, 4.4), dpi=150)

    for _, row in runs.iterrows():
        gains = [row[f"gain_pass_at_{k}"] for k in K_VALUES]
        left.plot(K_VALUES, gains, color=SIZE_COLORS[row["size"]],
                  linewidth=1.6, marker="o", markersize=4, alpha=0.75)
    left.axhline(0.0, color="#52514e", linewidth=1, linestyle="--")
    left.axvline(8, color="#52514e", linewidth=1, linestyle=":")
    left.annotate("training K", xy=(8, left.get_ylim()[1]),
                  textcoords="offset points", xytext=(4, -12),
                  fontsize=8, color="#52514e")
    left.set_xscale("log", base=2)
    left.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
    left.set_xlabel("k")
    left.set_ylabel("pass@k gain, first to last eval step")
    left.set_title("Gain profile per run (color = model size)")
    handles = [plt.Line2D([], [], color=color, linewidth=2, label=size)
               for size, color in SIZE_COLORS.items()]
    left.legend(handles=handles, frameon=False, fontsize=8)

    for run_id, frame in curves.groupby("run_id"):
        frame = frame.sort_values("rl_step")
        tail = frame[PASS_COLUMNS[16]] - frame[PASS_COLUMNS[8]]
        middle.plot(frame.rl_step, tail,
                    color=SIZE_COLORS[frame.size_str.iloc[0]],
                    linewidth=1.6, alpha=0.75)
    middle.set_xscale("log")
    middle.set_xlabel("RL step")
    middle.set_ylabel("pass@16 − pass@8")
    middle.set_title("Tail-difference mass over training")

    focus = shells[shells.run_id == focus_run].sort_values("rl_step")
    shell_ramp = {"frozen": "#9ec5f4", "deep_cold": "#5598e7",
                  "cold": "#2a78d6", "band": "#1c5cab", "saturated": "#104281"}
    for shell, color in shell_ramp.items():
        right.plot(focus.rl_step, focus[shell], color=color,
                   linewidth=2, marker="o", markersize=4, label=shell)
    right.set_xscale("log")
    right.set_xlabel("RL step")
    right.set_ylabel("deconvolved shell mass")
    right.set_title(f"Shell evolution, {focus_run}")
    right.legend(frameon=False, fontsize=8)

    for axis in (left, middle, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(True, which="major", color="#0b0b0b", alpha=0.08,
                  linewidth=0.6)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)


def run_config(config: Config) -> pd.DataFrame:
    curves = pd.read_csv(config.input_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.DataFrame([
        analyze_run(frame) for _, frame in curves.groupby("run_id")
    ]).sort_values(["size", "alpha"])
    shells = pd.concat([
        shell_evolution(frame) for _, frame in curves.groupby("run_id")
    ], ignore_index=True)
    runs.to_csv(output_dir / "run_gains.csv", index=False)
    shells.to_csv(output_dir / "shell_evolution.csv", index=False)

    display = runs[[
        "run_id", "first_step", "last_step",
        "gain_pass_at_1", "gain_pass_at_8", "gain_pass_at_16", "gain_se_16",
        "tail_slope_per_log_step",
    ]]
    print(display.to_string(index=False))
    significant = (
        runs.gain_pass_at_16.abs() > 2 * runs.gain_se_16
    )
    print(f"\nruns with |pass@16 gain| > 2 SE: {int(significant.sum())} "
          f"of {len(runs)}")
    print(f"median gain ratio pass@16/pass@1: "
          f"{(runs.gain_pass_at_16 / runs.gain_pass_at_1).median():.3f}")
    print(f"runs with negative tail-difference slope: "
          f"{int((runs.tail_slope_per_log_step < 0).sum())} of {len(runs)}")

    focus = "C6p5e18_20m_alpha0.400_beta0.008"
    if focus not in set(runs.run_id):
        focus = runs.run_id.iloc[0]
    plot(runs, curves, shells, focus, Path(config.figure))
    return runs


def parse_config() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    type_hints = get_type_hints(Config)
    for field in dataclasses.fields(Config):
        parser.add_argument(f"--{field.name}",
                            type=TYPE_MAP[type_hints[field.name]],
                            default=field.default)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    config = parse_config()
    started = time.time()
    run_config(config)
    print(f"Done ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
