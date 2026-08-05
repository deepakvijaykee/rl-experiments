"""Drift-law measurement in the analytical environment.

The reservoir derivation rests on one local law: under group-centered credit
the expected success-probability drift of a task obeys ``q̇ ∝ q²`` at small
``q`` — mean baseline because the unbiased gradient itself carries ``q``,
std normalization because rescaled mixed groups occur at rate ``Kq`` and the
gradient carries the other factor — with std faster than mean by roughly
``√K / (1 - 1/K)``. Conversion time then scales as ``1/q₀``, and a task
population log-uniform in ``q₀`` yields reward linear in ``log`` compute.

This script measures all three consequences by direct simulation:

1. **Drift scaling**: early drift across a (p₀, D, K, normalization) grid,
   fit ``log(q̇ / D(1-p₀)²)`` against ``log q₀`` — slope should be 2 for both
   normalizations, with a ``√K``-spaced offset between them.
2. **Conversion time**: steps to reach success 0.5 against ``1/q₀``.
3. **Reservoir sweep**: the equal-weight mixture over depths 1..D_max plotted
   against log steps — the log-linear window and its depletion.

Each prompt group covers every prompt once per step and the optimizer is
plain SGD, so the measured drift is the estimator's own expectation rather
than an adaptive-optimizer artifact.
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
import torch

from .env import (
    LayeredDecisionEnvironment,
    LayeredEnvironmentConfig,
    TabularLayeredPolicy,
    group_centered_policy_loss,
)

TYPE_MAP = {int: int, float: float, str: str}


@dataclass(frozen=True)
class Config:
    output_dir: str = "results/analytical_drift"
    figure: str = "learnability_sandbox/figures/analytical_drift.png"
    num_prompts: int = 64
    learning_rate: float = 0.5
    num_seeds: int = 3
    branching_factor: int = 4
    drift_max_steps: int = 2000
    conversion_max_steps: int = 30000
    mixture_max_depth: int = 8
    device: str = "cpu"


def run_training(
    environment: LayeredDecisionEnvironment,
    config: Config,
    group_size: int,
    normalization: str,
    seed: int,
    max_steps: int,
    stop_success: float,
) -> list[tuple[int, float]]:
    """Train and return (step, exact mean success) until a stop condition.

    The SGD learning rate is scaled by ``num_prompts`` so the per-prompt
    effective step matches ``config.learning_rate`` regardless of how many
    prompts share the batch mean.
    """
    torch.manual_seed(seed)
    policy = environment.make_policy()
    optimizer = torch.optim.SGD(
        policy.parameters(), lr=config.learning_rate * config.num_prompts
    )
    prompt_ids = torch.arange(config.num_prompts)
    trajectory = []
    for step in range(max_steps + 1):
        success = environment.exact_metrics(policy, group_size)[
            "success_probability"
        ]
        trajectory.append((step, success))
        if success >= stop_success:
            break
        batch = environment.rollout(policy, prompt_ids, group_size)
        loss = group_centered_policy_loss(policy, batch, normalization)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return trajectory


def make_environment(config: Config, horizon: int, p0: float) -> LayeredDecisionEnvironment:
    return LayeredDecisionEnvironment(
        LayeredEnvironmentConfig(
            num_prompts=config.num_prompts,
            horizon=horizon,
            branching_factor=config.branching_factor,
            initial_correct_probability=p0,
            task_seed=0,
        ),
        torch.device(config.device),
    )


def measure_drift(config: Config) -> pd.DataFrame:
    """Early drift q̇ at the initial condition, averaged over seeds.

    The window ends when success grows by half of itself (drift still local)
    or at ``drift_max_steps``; the estimate is the mean slope over the window.
    """
    rows = []
    for p0 in (0.15, 0.3, 0.5):
        for horizon in (2, 3, 4):
            q0 = p0**horizon
            for group_size in (4, 8, 16):
                for normalization in ("mean", "std"):
                    drifts = []
                    for seed in range(config.num_seeds):
                        environment = make_environment(config, horizon, p0)
                        trajectory = run_training(
                            environment, config, group_size, normalization,
                            seed, config.drift_max_steps, stop_success=1.5 * q0,
                        )
                        steps, success = trajectory[-1]
                        if steps > 0:
                            drifts.append((success - q0) / steps)
                    rows.append({
                        "p0": p0,
                        "horizon": horizon,
                        "q0": q0,
                        "group_size": group_size,
                        "normalization": normalization,
                        "drift": float(np.mean(drifts)),
                        "window_reached": len(drifts),
                    })
                    print(f"  drift p0={p0} D={horizon} K={group_size} "
                          f"{normalization}: {np.mean(drifts):.3e}", flush=True)
    return pd.DataFrame(rows)


def measure_conversion(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conversion times and the depth-mixture reward curve at K=8, p0=0.5."""
    conversion_rows, curve_rows = [], []
    for normalization in ("mean", "std"):
        for horizon in range(1, config.mixture_max_depth + 1):
            q0 = 0.5**horizon
            environment = make_environment(config, horizon, 0.5)
            trajectory = run_training(
                environment, config, 8, normalization,
                seed=0, max_steps=config.conversion_max_steps, stop_success=0.9,
            )
            crossed = [s for s, success in trajectory if success >= 0.5]
            conversion_rows.append({
                "normalization": normalization,
                "horizon": horizon,
                "q0": q0,
                "conversion_step": crossed[0] if crossed else None,
                "final_success": trajectory[-1][1],
                "steps_run": trajectory[-1][0],
            })
            for step, success in trajectory:
                curve_rows.append({
                    "normalization": normalization,
                    "horizon": horizon,
                    "step": step,
                    "success": success,
                })
            print(f"  conversion {normalization} D={horizon}: "
                  f"step {crossed[0] if crossed else 'not reached'}", flush=True)
    return pd.DataFrame(conversion_rows), pd.DataFrame(curve_rows)


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(np.log10(x), np.log10(y), 1)
    return slope, intercept


def report_fits(drift: pd.DataFrame, conversion: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (normalization, group_size), frame in drift.groupby(
        ["normalization", "group_size"]
    ):
        normalized = frame.drift / (frame.horizon * (1 - frame.p0) ** 2)
        slope, intercept = fit_power_law(frame.q0.values, normalized.values)
        rows.append({
            "measurement": "drift_vs_q0",
            "normalization": normalization,
            "group_size": group_size,
            "slope": slope,
            "intercept": intercept,
        })
    converted = conversion.dropna(subset=["conversion_step"])
    converted = converted[converted.conversion_step > 0]
    for normalization, frame in converted.groupby("normalization"):
        slope, intercept = fit_power_law(
            1.0 / frame.q0.values, frame.conversion_step.values
        )
        rows.append({
            "measurement": "conversion_vs_inverse_q0",
            "normalization": normalization,
            "group_size": 8,
            "slope": slope,
            "intercept": intercept,
        })
    fits = pd.DataFrame(rows)
    print(fits.to_string(index=False))
    return fits


def plot(
    drift: pd.DataFrame,
    conversion: pd.DataFrame,
    curves: pd.DataFrame,
    figure_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(14, 4.2), dpi=150)
    norm_colors = {"mean": "#2a78d6", "std": "#eb6834"}

    for (normalization, group_size), frame in drift.groupby(
        ["normalization", "group_size"]
    ):
        normalized = frame.drift / (frame.horizon * (1 - frame.p0) ** 2)
        left.scatter(
            frame.q0, normalized,
            s={4: 14, 8: 26, 16: 42}[group_size],
            color=norm_colors[normalization], alpha=0.6, linewidths=0,
        )
    reference = np.geomspace(drift.q0.min(), drift.q0.max(), 50)
    left.plot(reference, 0.5 * reference**2, color="#52514e",
              linewidth=1, linestyle="--")
    left.annotate("slope 2", xy=(reference[25], 0.5 * reference[25] ** 2),
                  textcoords="offset points", xytext=(6, -10), fontsize=8,
                  color="#52514e")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("initial success probability q0")
    left.set_ylabel("drift / (D (1−p0)²)")
    left.set_title("Early drift scaling (marker size = K)")

    converted = conversion.dropna(subset=["conversion_step"])
    converted = converted[converted.conversion_step > 0]
    for normalization, frame in converted.groupby("normalization"):
        middle.plot(
            1.0 / frame.q0, frame.conversion_step,
            color=norm_colors[normalization], linewidth=2,
            marker="o", markersize=5, label=normalization,
        )
    middle.set_xscale("log")
    middle.set_yscale("log")
    middle.set_xlabel("1 / q0")
    middle.set_ylabel("steps to success 0.5")
    middle.set_title("Conversion time (K=8, p0=0.5)")
    middle.legend(frameon=False, fontsize=9, title="advantage")

    for normalization, frame in curves.groupby("normalization"):
        # Trajectories stop at their own saturation; hold each depth at its
        # final value on a common step grid so the mixture never loses mass.
        grid_curves = (
            frame.pivot(index="step", columns="horizon", values="success")
            .ffill()
        )
        mixture = grid_curves.mean(axis=1)
        right.plot(
            mixture.index + 1, mixture.values,
            color=norm_colors[normalization], linewidth=2, label=normalization,
        )
    right.set_xscale("log")
    right.set_xlabel("training step + 1")
    right.set_ylabel("mixture success (depths 1..8)")
    right.set_title("Reservoir sweep: the log-linear window")
    right.legend(frameon=False, fontsize=9, title="advantage")

    for axis in (left, middle, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(True, which="major", color="#0b0b0b", alpha=0.08,
                  linewidth=0.6)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)


def run_config(config: Config) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("drift grid", flush=True)
    drift = measure_drift(config)
    drift.to_csv(output_dir / "drift.csv", index=False)
    print("conversion and mixture", flush=True)
    conversion, curves = measure_conversion(config)
    conversion.to_csv(output_dir / "conversion.csv", index=False)
    curves.to_csv(output_dir / "mixture_curves.csv", index=False)
    fits = report_fits(drift, conversion)
    fits.to_csv(output_dir / "fits.csv", index=False)
    plot(drift, conversion, curves, Path(config.figure))


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
