"""Train a tabular policy in the analytical unique-path environment."""

from __future__ import annotations

import argparse
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import get_type_hints

import pandas as pd
import torch

from .env import (
    LayeredDecisionEnvironment,
    LayeredEnvironmentConfig,
    RolloutBatch,
    TabularLayeredPolicy,
    group_centered_policy_loss,
    rollout_metrics,
)


TYPE_MAP = {int: int, float: float, str: str}


@dataclass(frozen=True)
class Config:
    num_steps: int = 200
    groups_per_step: int = 64
    group_size: int = 8
    learning_rate: float = 1e-2
    eval_every: int = 10
    num_seeds: int = 3
    seed: int = 0
    output: str = "results/learnability.csv"
    verbose: bool = True
    device: str = "auto"
    num_prompts: int = 128
    horizon: int = 3
    branching_factor: int = 4
    initial_correct_probability: float = 0.5
    task_seed: int = 0
    advantage_normalization: str = "mean"

    def __post_init__(self) -> None:
        if self.advantage_normalization not in {"mean", "std"}:
            raise ValueError("advantage_normalization must be mean or std")
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.groups_per_step < 1:
            raise ValueError("groups_per_step must be >= 1")
        if self.group_size < 2:
            raise ValueError("group_size must be >= 2")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.eval_every < 1:
            raise ValueError("eval_every must be >= 1")
        if self.num_seeds < 1:
            raise ValueError("num_seeds must be >= 1")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")

    def environment_config(self) -> LayeredEnvironmentConfig:
        return LayeredEnvironmentConfig(
            num_prompts=self.num_prompts,
            horizon=self.horizon,
            branching_factor=self.branching_factor,
            initial_correct_probability=self.initial_correct_probability,
            task_seed=self.task_seed,
        )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("device=cuda requested but CUDA is unavailable")
    return torch.device(name)


def sample_training_batch(
    environment: LayeredDecisionEnvironment,
    policy: TabularLayeredPolicy,
    config: Config,
    device: torch.device,
) -> RolloutBatch:
    prompt_ids = torch.randint(
        config.num_prompts,
        (config.groups_per_step,),
        device=device,
    )
    return environment.rollout(policy, prompt_ids, config.group_size)


def measurement_row(
    environment: LayeredDecisionEnvironment,
    policy: TabularLayeredPolicy,
    batch: RolloutBatch,
    loss: torch.Tensor,
    config: Config,
    step: int,
) -> dict[str, float]:
    return {
        "step": step,
        "loss": loss.item(),
        **rollout_metrics(batch),
        **environment.exact_metrics(policy, config.group_size),
    }


def train_one_seed(
    environment: LayeredDecisionEnvironment,
    config: Config,
    seed: int,
    device: torch.device,
) -> list[dict[str, float]]:
    torch.manual_seed(seed)
    policy = environment.make_policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    rows = []

    for step in range(config.num_steps):
        batch = sample_training_batch(environment, policy, config, device)
        loss = group_centered_policy_loss(
            policy, batch, config.advantage_normalization
        )

        if step % config.eval_every == 0:
            row = measurement_row(
                environment,
                policy,
                batch,
                loss,
                config,
                step,
            )
            rows.append(row)
            if config.verbose and step % (config.eval_every * 10) == 0:
                print(
                    f"  step {step:5d} "
                    f"success={row['success_probability']:.4f} "
                    f"mixed={row['predicted_mixed_group_rate']:.4f}"
                )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_batch = sample_training_batch(environment, policy, config, device)
    final_loss = group_centered_policy_loss(
        policy, final_batch, config.advantage_normalization
    )
    rows.append(measurement_row(
        environment,
        policy,
        final_batch,
        final_loss,
        config,
        config.num_steps,
    ))
    return rows


def run_config(config: Config) -> pd.DataFrame:
    device = resolve_device(config.device)
    environment = LayeredDecisionEnvironment(config.environment_config(), device)
    rows = []
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for seed_offset in range(config.num_seeds):
        seed = config.seed + seed_offset
        if config.verbose:
            print(f"group_centered seed={seed} horizon={config.horizon}")
        seed_rows = train_one_seed(environment, config, seed, device)
        for row in seed_rows:
            row.update({
                "seed": seed,
                "method": "group_centered",
                "task": "layered_decision",
                "horizon": config.horizon,
                "branching_factor": config.branching_factor,
                "group_size": config.group_size,
                "groups_per_step": config.groups_per_step,
                "initial_correct_probability": config.initial_correct_probability,
                "advantage_normalization": config.advantage_normalization,
            })
        rows.extend(seed_rows)
        pd.DataFrame(rows).to_csv(output, index=False)

    return pd.DataFrame(rows)


def parse_bool(text: str) -> bool:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_config(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    type_hints = get_type_hints(Config)
    for field in dataclasses.fields(Config):
        field_type = type_hints[field.name]
        argument_type = parse_bool if field_type is bool else TYPE_MAP[field_type]
        parser.add_argument(
            f"--{field.name}",
            type=argument_type,
            default=field.default,
        )
    return Config(**vars(parser.parse_args(argv)))


def main() -> None:
    config = parse_config()
    started = time.time()
    frame = run_config(config)
    print(f"Saved {len(frame)} rows to {config.output} ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
