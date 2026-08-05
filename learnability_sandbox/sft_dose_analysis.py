"""Adaptation dose-response on the released SFT sweep.

The released `all_sft_models.csv` grid contains the closest existing
off-manifold axis to the mid-training question: for many pretrained bases
(fixed compute class, size, and pretraining allocation alpha) the paper
trained SFT models at more than one SFT compute fraction beta, and
separately trained both `thinking` and `nonthinking` SFT variants — all
evaluated with pass@k for k in {1, 4, 8, 16}.

Deconvolving each model's pass@k moments into shell masses (the Result 6
instrument) turns that grid into two measurements:

1. **Dose-response**: how band-relevant mass — the deconvolved contrast
   mass at K=8, and the model-free tail moment ``pass@16 - pass@8`` —
   moves with adaptation dose at a fixed base. The reservoir account of
   mid-training predicts an early rise (cold mass lifted toward the band)
   with a saturating or inverting tail (mode-seeking SFT drains the
   diversity that carries pass@k at large k).
2. **Format contribution**: thinking vs nonthinking at matched base and
   dose isolates what the think format itself adds to the RL-relevant
   histogram, at the SFT stage where both variants are on equal footing.

A base is identified by (compute class, size, alpha); dose lines never mix
compute classes or SFT kinds.
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

from .released_curve_analysis import K_VALUES, SHELL_ATOMS, deconvolve_shells

TYPE_MAP = {int: int, float: float, str: str}

PASS_COLUMNS = {k: f"pass_at_{k}" for k in K_VALUES}
SIZE_COLORS = {"20m": "#2a78d6", "50m": "#eb6834", "100m": "#e87ba4",
               "200m": "#1baf7a", "680m": "#eda100"}
BASE_KEY = ["compute_class", "size_str", "alpha"]


@dataclass(frozen=True)
class Config:
    input_path: str = (
        "data/chess_pre_to_post/collected_pre_to_post_data/all_sft_models.csv"
    )
    output_path: str = "results/released_curves/sft_dose_response.csv"
    figure: str = "learnability_sandbox/figures/sft_dose_response.png"
    group_size: int = 8


def contrast_mass(q: float, group_size: int) -> float:
    return 1.0 - q**group_size - (1.0 - q) ** group_size


def deconvolve_rows(frame: pd.DataFrame, group_size: int) -> pd.DataFrame:
    rows = []
    for _, model in frame.iterrows():
        shells = deconvolve_shells(
            {k: model[column] for k, column in PASS_COLUMNS.items()}
        )
        rows.append({
            **{key: model[key] for key in BASE_KEY},
            "sft_kind": model.sft_kind,
            "beta": model.beta,
            "pass_at_1": model.pass_at_1,
            "tail_moment": model.pass_at_16 - model.pass_at_8,
            **shells,
            "contrast_mass": sum(
                weight * contrast_mass(q, group_size)
                for (name, q), weight in zip(SHELL_ATOMS.items(),
                                             shells.values())
            ),
            "tail_mass": shells["frozen"] + shells["deep_cold"],
        })
    return pd.DataFrame(rows)


def dose_deltas(shells: pd.DataFrame) -> pd.DataFrame:
    """Per-base change of each summary from lowest to highest SFT dose."""
    rows = []
    for (kind, *base), frame in shells.groupby(["sft_kind", *BASE_KEY]):
        if frame.beta.nunique() < 2:
            continue
        frame = frame.sort_values("beta")
        low, high = frame.iloc[0], frame.iloc[-1]
        rows.append({
            "sft_kind": kind,
            **dict(zip(BASE_KEY, base)),
            "beta_low": low.beta,
            "beta_high": high.beta,
            "doses": len(frame),
            "delta_pass_at_1": high.pass_at_1 - low.pass_at_1,
            "delta_contrast_mass": high.contrast_mass - low.contrast_mass,
            "delta_tail_moment": high.tail_moment - low.tail_moment,
            "delta_saturated": high.saturated - low.saturated,
        })
    return pd.DataFrame(rows)


def format_pairs(shells: pd.DataFrame) -> pd.DataFrame:
    """Thinking minus nonthinking at matched base and dose."""
    keys = [*BASE_KEY, "beta"]
    thinking = shells[shells.sft_kind == "thinking"]
    nonthinking = shells[shells.sft_kind == "nonthinking"]
    pairs = thinking.merge(
        nonthinking, on=keys, suffixes=("_think", "_direct"),
        validate="one_to_one",
    )
    for column in ("pass_at_1", "contrast_mass", "tail_moment", "saturated"):
        pairs[f"format_gap_{column}"] = (
            pairs[f"{column}_think"] - pairs[f"{column}_direct"]
        )
    return pairs


def plot(
    shells: pd.DataFrame,
    pairs: pd.DataFrame,
    group_size: int,
    figure_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(14, 4.4), dpi=150)

    dosed = shells.groupby(["sft_kind", *BASE_KEY]).filter(
        lambda frame: frame.beta.nunique() >= 2
    )

    def dose_lines(axis, column):
        for (kind, *base), frame in dosed.groupby(["sft_kind", *BASE_KEY]):
            frame = frame.sort_values("beta")
            axis.plot(
                frame.beta, frame[column],
                color=SIZE_COLORS[base[1]], linewidth=1.6,
                marker="o", markersize=4, alpha=0.8,
                linestyle="-" if kind == "thinking" else "--",
            )
        axis.set_xscale("log")
        axis.set_xlabel("SFT compute fraction beta")

    dose_lines(left, "contrast_mass")
    left.set_ylabel(f"deconvolved contrast mass, K={group_size}")
    left.set_title("Dose-response of RL-touchable mass")
    dosed_sizes = dosed.size_str.unique()
    handles = [plt.Line2D([], [], color=color, linewidth=2, label=size)
               for size, color in SIZE_COLORS.items() if size in dosed_sizes]
    if (dosed.sft_kind == "nonthinking").any():
        handles.append(plt.Line2D([], [], color="#52514e", linewidth=1.6,
                                  linestyle="--", label="nonthinking"))
    left.legend(handles=handles, frameon=False, fontsize=8)

    dose_lines(middle, "tail_moment")
    middle.set_ylabel("pass@16 − pass@8")
    middle.set_title("Dose-response of the sub-band tail (model-free)")

    right.axhline(0.0, color="#52514e", linewidth=1, linestyle="--")
    for _, pair in pairs.iterrows():
        right.scatter(
            pair.format_gap_pass_at_1, pair.format_gap_tail_moment,
            s=42, color=SIZE_COLORS[pair.size_str], alpha=0.85, linewidths=0,
        )
    right.axvline(0.0, color="#52514e", linewidth=1, linestyle="--")
    right.set_xlabel("thinking − nonthinking: pass@1")
    right.set_ylabel("thinking − nonthinking: pass@16 − pass@8")
    right.set_title("Format contribution at matched base and dose")

    for axis in (left, middle, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(True, which="major", color="#0b0b0b", alpha=0.08,
                  linewidth=0.6)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)


def run_config(config: Config) -> pd.DataFrame:
    models = pd.read_csv(config.input_path)
    models = models.dropna(subset=list(PASS_COLUMNS.values()))
    shells = deconvolve_rows(models, config.group_size)
    deltas = dose_deltas(shells)
    pairs = format_pairs(shells)

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shells.to_csv(output_path, index=False)
    deltas.to_csv(output_path.with_name("sft_dose_deltas.csv"), index=False)
    pairs.to_csv(output_path.with_name("sft_format_pairs.csv"), index=False)

    print(f"{len(shells)} evaluated SFT models, "
          f"{len(deltas)} bases with a dose axis, "
          f"{len(pairs)} matched format pairs\n")
    print(deltas.to_string(index=False))
    for column in ("delta_pass_at_1", "delta_contrast_mass",
                   "delta_tail_moment"):
        signs = np.sign(deltas[column])
        print(f"{column}: positive in {int((signs > 0).sum())} "
              f"of {len(deltas)} bases (median {deltas[column].median():+.4f})")
    print()
    summary = pairs[["size_str", "alpha", "beta",
                     "format_gap_pass_at_1", "format_gap_tail_moment"]]
    print(summary.to_string(index=False))
    print(f"\nformat gap pass@1: median "
          f"{pairs.format_gap_pass_at_1.median():+.4f}; "
          f"tail moment: median {pairs.format_gap_tail_moment.median():+.4f}")

    plot(shells, pairs, config.group_size, Path(config.figure))
    return shells


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
