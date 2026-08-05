"""Compare strict and teacher-forced protocols on released chess puzzles."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import get_type_hints

import pandas as pd

from .chess_env import (
    ChessPuzzle,
    Protocol,
    StrictChessPuzzleEnvironment,
    TeacherForcedReplyEnvironment,
)


ENVIRONMENT_TYPES = {
    Protocol.STRICT_TERMINATION: StrictChessPuzzleEnvironment,
    Protocol.TEACHER_FORCED_REPLY_REPLAY: TeacherForcedReplyEnvironment,
}


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_path: Path
    correct_probability: float = 0.6
    group_size: int = 8
    num_seeds: int = 3
    seed: int = 0
    protocol: str = "both"

    def __post_init__(self) -> None:
        if not 0.0 <= self.correct_probability <= 1.0:
            raise ValueError("correct_probability must be in [0, 1]")
        if self.group_size < 2:
            raise ValueError("group_size must be >= 2")
        if self.num_seeds < 1:
            raise ValueError("num_seeds must be >= 1")
        if self.protocol not in ("both", *Protocol):
            raise ValueError(f"unknown protocol: {self.protocol!r}")


@dataclass(frozen=True)
class PuzzleMechanics:
    puzzle: ChessPuzzle
    legal_actions_by_step: tuple[tuple[str, ...], ...]

    @property
    def decision_depth(self) -> int:
        return sum(len(actions) > 1 for actions in self.legal_actions_by_step)


@dataclass(frozen=True)
class GroupSample:
    reward: float
    mixed_group_rate: float
    trajectory_length: float
    post_error_actions: float


def load_puzzles(path: Path) -> list[ChessPuzzle]:
    frame = pd.read_parquet(path, columns=["reward_model", "extra_info"])
    puzzles = []
    for row_index, row in enumerate(frame.itertuples(index=False)):
        puzzle = ChessPuzzle.from_release(
            row.extra_info["FEN"],
            row.extra_info["Moves"],
        )
        ground_truth = tuple(ast.literal_eval(row.reward_model["ground_truth"]))
        if ground_truth != puzzle.solver_moves:
            raise ValueError(
                f"row {row_index} solver moves disagree with reward ground truth"
            )
        puzzles.append(puzzle)
    return puzzles


def measure_mechanics(puzzle: ChessPuzzle) -> PuzzleMechanics:
    environment = StrictChessPuzzleEnvironment(puzzle)
    legal_actions_by_step = []
    while not environment.terminated:
        legal_actions_by_step.append(environment.legal_actions)
        target_action = puzzle.solver_moves[environment.solver_move_index]
        environment.step(target_action)
    return PuzzleMechanics(
        puzzle=puzzle,
        legal_actions_by_step=tuple(legal_actions_by_step),
    )


def sample_action_group(
    mechanics: PuzzleMechanics,
    correct_probability: float,
    group_size: int,
    random_generator: random.Random,
) -> tuple[tuple[str, ...], ...]:
    action_group = []
    for _ in range(group_size):
        actions = []
        for target_action, legal_actions in zip(
            mechanics.puzzle.solver_moves,
            mechanics.legal_actions_by_step,
            strict=True,
        ):
            alternatives = tuple(
                action for action in legal_actions if action != target_action
            )
            if alternatives and random_generator.random() >= correct_probability:
                actions.append(random_generator.choice(alternatives))
            else:
                actions.append(target_action)
        action_group.append(tuple(actions))
    return tuple(action_group)


def run_action_group(
    mechanics: PuzzleMechanics,
    action_group: tuple[tuple[str, ...], ...],
    protocol: Protocol,
) -> GroupSample:
    environment = ENVIRONMENT_TYPES[protocol](mechanics.puzzle)
    successes = 0
    trajectory_length = 0
    post_error_actions = 0

    for rollout_index, actions in enumerate(action_group):
        if rollout_index > 0:
            environment.reset()
        prefix_correct = True
        for action_index, action in enumerate(actions):
            post_error_actions += int(not prefix_correct)
            trajectory_length += 1
            prefix_correct &= (
                action == mechanics.puzzle.solver_moves[action_index]
            )
            transition = environment.step(action)
            if transition.terminated:
                break
        successes += int(transition.reward)

    group_size = len(action_group)
    return GroupSample(
        reward=successes / group_size,
        mixed_group_rate=float(0 < successes < group_size),
        trajectory_length=trajectory_length / group_size,
        post_error_actions=post_error_actions / group_size,
    )


def predict_protocol_metrics(
    mechanics: PuzzleMechanics,
    correct_probability: float,
    protocol: Protocol,
) -> tuple[float, float, float]:
    step_probabilities = tuple(
        1.0 if len(legal_actions) == 1 else correct_probability
        for legal_actions in mechanics.legal_actions_by_step
    )
    success_probability = 1.0
    strict_trajectory_length = 0.0
    replay_post_error_actions = 0.0
    for step_probability in step_probabilities:
        strict_trajectory_length += success_probability
        replay_post_error_actions += 1.0 - success_probability
        success_probability *= step_probability

    if protocol == Protocol.STRICT_TERMINATION:
        return success_probability, strict_trajectory_length, 0.0
    return (
        success_probability,
        float(len(step_probabilities)),
        replay_post_error_actions,
    )


def summarize(
    rows: list[dict[str, float]],
    horizon: str,
    seed: int,
    protocol: Protocol,
    config: Config,
) -> dict[str, float | int | str]:
    count = len(rows)
    return {
        "seed": seed,
        "horizon": horizon,
        "num_puzzles": count,
        "mean_decision_depth": sum(row["decision_depth"] for row in rows) / count,
        "predicted_reward": sum(row["predicted_reward"] for row in rows) / count,
        "sampled_reward": sum(row["sampled_reward"] for row in rows) / count,
        "predicted_mixed_group_rate": (
            sum(row["predicted_mixed_group_rate"] for row in rows) / count
        ),
        "sampled_mixed_group_rate": (
            sum(row["sampled_mixed_group_rate"] for row in rows) / count
        ),
        "predicted_trajectory_length": (
            sum(row["predicted_trajectory_length"] for row in rows) / count
        ),
        "sampled_trajectory_length": (
            sum(row["sampled_trajectory_length"] for row in rows) / count
        ),
        "predicted_post_error_actions": (
            sum(row["predicted_post_error_actions"] for row in rows) / count
        ),
        "sampled_post_error_actions": (
            sum(row["sampled_post_error_actions"] for row in rows) / count
        ),
        "correct_probability": config.correct_probability,
        "group_size": config.group_size,
        "protocol": protocol.value,
    }


def run_config(config: Config) -> pd.DataFrame:
    mechanics = [
        measure_mechanics(puzzle) for puzzle in load_puzzles(config.input_path)
    ]
    protocols = tuple(Protocol) if config.protocol == "both" else (Protocol(config.protocol),)
    summary_rows = []

    for seed_offset in range(config.num_seeds):
        seed = config.seed + seed_offset
        random_generator = random.Random(seed)
        rows_by_protocol = {protocol: [] for protocol in protocols}
        rows_by_protocol_and_horizon = {
            protocol: defaultdict(list) for protocol in protocols
        }

        for item in mechanics:
            action_group = sample_action_group(
                item,
                config.correct_probability,
                config.group_size,
                random_generator,
            )
            for protocol in protocols:
                sample = run_action_group(item, action_group, protocol)
                (
                    success_probability,
                    predicted_trajectory_length,
                    predicted_post_error_actions,
                ) = predict_protocol_metrics(
                    item,
                    config.correct_probability,
                    protocol,
                )
                row = {
                    "decision_depth": float(item.decision_depth),
                    "predicted_reward": success_probability,
                    "sampled_reward": sample.reward,
                    "predicted_mixed_group_rate": (
                        1.0
                        - success_probability**config.group_size
                        - (1.0 - success_probability) ** config.group_size
                    ),
                    "sampled_mixed_group_rate": sample.mixed_group_rate,
                    "predicted_trajectory_length": predicted_trajectory_length,
                    "sampled_trajectory_length": sample.trajectory_length,
                    "predicted_post_error_actions": predicted_post_error_actions,
                    "sampled_post_error_actions": sample.post_error_actions,
                }
                rows_by_protocol[protocol].append(row)
                rows_by_protocol_and_horizon[protocol][
                    len(item.puzzle.solver_moves)
                ].append(row)

        for protocol in protocols:
            summary_rows.append(
                summarize(rows_by_protocol[protocol], "all", seed, protocol, config)
            )
            for horizon, rows in sorted(
                rows_by_protocol_and_horizon[protocol].items()
            ):
                summary_rows.append(
                    summarize(rows, str(horizon), seed, protocol, config)
                )
        print(f"seed={seed} puzzles={len(mechanics)} protocols={len(protocols)}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(summary_rows)
    frame.to_csv(config.output_path, index=False)
    return frame


def parse_config() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    type_hints = get_type_hints(Config)
    for field in dataclasses.fields(Config):
        if type_hints[field.name] is Path:
            parser.add_argument(f"--{field.name}", type=Path, required=True)
        elif field.name == "protocol":
            parser.add_argument("--protocol",
                                choices=("both", *(p.value for p in Protocol)),
                                default=field.default)
        else:
            parser.add_argument(f"--{field.name}",
                                type=type_hints[field.name],
                                default=field.default)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    config = parse_config()
    started = time.time()
    frame = run_config(config)
    print(
        f"Saved {len(frame)} rows to {config.output_path} "
        f"({time.time() - started:.1f}s)"
    )


if __name__ == "__main__":
    main()
