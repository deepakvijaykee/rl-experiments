"""Evaluate a released-format checkpoint on the reconstructed puzzle bins.

Reproduces the released evaluation: multi-turn interaction with ``<call_env>``
as the stop token, temperature-1 pure sampling (vLLM's defaults disable top-k
and top-p filtering), N samples per puzzle, and the released trajectory
reward. The reply protocol is switchable between the released teacher-forced
replay and the paper's strict termination.

Outputs one summary row per (bin, scorer) with pass@k — the released parity
scorer beside the canonical board scorer — and a per-puzzle CSV with
empirical success rates for phase-diagram analysis.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import get_type_hints

import pandas as pd
import torch
from transformers import Qwen3ForCausalLM

from .chess_env import Protocol
from .puzzle_data import MultiTurnPuzzle, load_multi_turn_puzzles
from .lan_tokenizer import CALL_ENV, LanTokenizer
from .multi_turn import (
    MultiTurnRollout,
    StrictReplies,
    TeacherForcedReplies,
    board_verdict,
    score_solver_sequence,
)

# Reply-protocol factories, keyed by the canonical protocol names.
REPLY_PROTOCOLS = {
    Protocol.STRICT_TERMINATION: lambda p: StrictReplies(p.puzzle),
    Protocol.TEACHER_FORCED_REPLY_REPLAY: lambda p: TeacherForcedReplies(p.env_replies),
}


@dataclass(frozen=True)
class Config:
    checkpoint: Path
    eval_dir: Path
    output: Path
    protocol: str = Protocol.TEACHER_FORCED_REPLY_REPLAY.value
    num_samples: int = 16
    temperature: float = 1.0
    max_env_calls: int = 6
    max_response_tokens: int = 2560
    max_batch_tokens: int = 300_000
    seed: int = 0
    limit_per_bin: int | None = None

    def __post_init__(self) -> None:
        if self.protocol not in REPLY_PROTOCOLS:
            raise ValueError(f"unknown protocol: {self.protocol!r}")
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if self.max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be >= 1")


def load_verified_tokenizer(checkpoint: Path) -> LanTokenizer:
    """The vendored SFT tokenizer, verified against the checkpoint's vocab."""
    tokenizer = LanTokenizer.sft()
    shipped = json.loads((checkpoint / "vocab.json").read_text())
    if shipped != tokenizer.vocab:
        raise ValueError(f"checkpoint {checkpoint} ships a different vocabulary")
    return tokenizer


@dataclass
class PuzzleEval:
    puzzle: MultiTurnPuzzle
    rollouts: list[MultiTurnRollout]

    def released_successes(self) -> list[float]:
        """Parity scores under the released text scorer, defects included —
        the column to compare against published numbers."""
        targets = self.puzzle.target_moves
        return [
            score_solver_sequence(rollout.response_text, targets).score
            for rollout in self.rollouts
        ]

    def board_successes(self) -> list[float]:
        """Canonical scores: submitted moves replayed on the board."""
        return [
            board_verdict(rollout.submitted_moves, self.puzzle.puzzle)
            for rollout in self.rollouts
        ]


def build_evals(
    puzzles: list[MultiTurnPuzzle],
    tokenizer: LanTokenizer,
    config: Config,
) -> list[PuzzleEval]:
    make_replies = REPLY_PROTOCOLS[config.protocol]
    evals = []
    for item in puzzles:
        prompt_ids = tokenizer.encode_prompt(item.prompt)
        rollouts = [
            MultiTurnRollout(
                prompt_ids=prompt_ids,
                tokenizer=tokenizer,
                replies=make_replies(item),
                max_env_calls=config.max_env_calls,
                max_model_tokens=config.max_response_tokens,
            )
            for _ in range(config.num_samples)
        ]
        evals.append(PuzzleEval(item, rollouts))
    return evals


def _memory_chunks(
    active: list[MultiTurnRollout],
    max_batch_tokens: int,
) -> list[list[MultiTurnRollout]]:
    """Split context-sorted rollouts into generate() chunks whose worst-case
    KV footprint — rows x (context width + new tokens) — stays within budget.

    The resource a generation call consumes is tokens, not rows: late rounds
    carry long contexts and must run smaller chunks than early ones. A chunk
    always takes at least one rollout, so an oversized single sample runs
    alone rather than stalling the loop.
    """
    chunks, start = [], 0
    while start < len(active):
        end, new_tokens = start, 0
        while end < len(active):
            width = len(active[end].context)
            new_tokens = max(new_tokens, active[end].remaining_budget)
            rows = end - start + 1
            if rows > 1 and rows * (width + new_tokens) > max_batch_tokens:
                break
            end += 1
        chunks.append(active[start:end])
        start = end
    return chunks


@torch.no_grad()
def run_rollouts(
    model: Qwen3ForCausalLM,
    tokenizer: LanTokenizer,
    rollouts: list[MultiTurnRollout],
    config: Config,
    device: torch.device,
) -> None:
    call_env_id = tokenizer.vocab[CALL_ENV]
    pad_id = tokenizer.bos_id
    round_index = 0
    while True:
        # A rollout can hit <call_env> exactly at its token budget: it is
        # not yet finished yet must generate nothing more. Submitting an
        # empty round records the budget termination; generate() itself
        # rejects max_new_tokens=0.
        for rollout in rollouts:
            if not rollout.finished and rollout.remaining_budget == 0:
                rollout.submit([])
        active = [r for r in rollouts if not r.finished]
        if not active:
            return
        active.sort(key=lambda r: len(r.context))
        started = time.time()
        chunks = _memory_chunks(active, config.max_batch_tokens)
        for chunk_index, chunk in enumerate(chunks):
            width = max(len(r.context) for r in chunk)
            input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for i, rollout in enumerate(chunk):
                ctx = rollout.context
                input_ids[i, width - len(ctx):] = torch.tensor(ctx, dtype=torch.long)
                attention_mask[i, width - len(ctx):] = 1
            output = model.generate(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                do_sample=True,
                temperature=config.temperature,
                top_p=1.0,
                top_k=0,
                max_new_tokens=max(r.remaining_budget for r in chunk),
                eos_token_id=[tokenizer.eos_id, call_env_id],
                pad_token_id=pad_id,
            )
            for i, rollout in enumerate(chunk):
                rollout.submit(output[i, width:].tolist())
            print(
                f"    chunk {chunk_index + 1}/{len(chunks)}: "
                f"{len(chunk)} rollouts, ctx<={width}, "
                f"{time.time() - started:.0f}s elapsed",
                flush=True,
            )
        round_index += 1
        print(
            f"  round {round_index}: {len(active)} active rollouts, "
            f"{time.time() - started:.0f}s",
            flush=True,
        )


def pass_at_k(success_counts: list[int], n: int, k: int) -> float:
    """Unbiased pass@k over per-puzzle success counts out of n samples."""
    total = 0.0
    for c in success_counts:
        total += 1.0 - comb(n - c, k) / comb(n, k)
    return total / len(success_counts)


def summarize(
    bin_name: str,
    scorer: str,
    counts: list[int],
    config: Config,
) -> dict:
    row = {
        "bin": bin_name,
        "scorer": scorer,
        "num_puzzles": len(counts),
        "protocol": config.protocol,
        "reward_mean": sum(counts) / (len(counts) * config.num_samples),
    }
    for k in (1, 4, 8, 16):
        if k <= config.num_samples:
            row[f"pass_at_{k}"] = pass_at_k(counts, config.num_samples, k)
    return row


def run_config(config: Config) -> pd.DataFrame:
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_verified_tokenizer(config.checkpoint)
    model = Qwen3ForCausalLM.from_pretrained(config.checkpoint, dtype=torch.bfloat16)
    model = model.to(device).eval()

    parquets = sorted(config.eval_dir.glob("test_B*_multi_turn.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"no test_B*_multi_turn.parquet under {config.eval_dir} — "
            "run learnability_sandbox.eval_puzzles first"
        )
    summary_rows = []
    puzzle_rows = []
    for parquet in parquets:
        puzzles = load_multi_turn_puzzles(parquet)
        if config.limit_per_bin is not None:
            puzzles = puzzles[: config.limit_per_bin]
        evals = build_evals(puzzles, tokenizer, config)
        print(f"{parquet.name}: {len(puzzles)} puzzles "
              f"x {config.num_samples} samples ({config.protocol})", flush=True)
        run_rollouts(
            model,
            tokenizer,
            [r for e in evals for r in e.rollouts],
            config,
            device,
        )

        released_counts, board_counts = [], []
        for item in evals:
            released = int(sum(item.released_successes()))
            board = int(sum(item.board_successes()))
            released_counts.append(released)
            board_counts.append(board)
            puzzle_rows.append({
                "data_source": item.puzzle.data_source,
                "fen": item.puzzle.puzzle.initial_fen,
                "horizon": len(item.puzzle.target_moves),
                "rating": item.puzzle.rating,
                "released_success_count": released,
                "board_success_count": board,
                "num_samples": config.num_samples,
                "mean_model_tokens": sum(
                    r.n_model_tokens for r in item.rollouts
                ) / config.num_samples,
                "terminations": ",".join(r.termination for r in item.rollouts),
            })
        bin_name = parquet.stem.removesuffix("_multi_turn")
        for scorer, counts in (("released", released_counts), ("board", board_counts)):
            row = summarize(bin_name, scorer, counts, config)
            summary_rows.append(row)
            print(f"  {row}", flush=True)
        # Durable progress: a failure in a later bin must not lose the
        # completed ones on a multi-hour run.
        _write_outputs(summary_rows, puzzle_rows, config)
        torch.cuda.empty_cache()

    for scorer, column in (("released", "released_success_count"),
                           ("board", "board_success_count")):
        counts = [row[column] for row in puzzle_rows]
        summary_rows.append(summarize("all", scorer, counts, config))
    return _write_outputs(summary_rows, puzzle_rows, config)


def _write_outputs(
    summary_rows: list[dict],
    puzzle_rows: list[dict],
    config: Config,
) -> pd.DataFrame:
    config.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(summary_rows)
    frame.to_csv(config.output, index=False)
    pd.DataFrame(puzzle_rows).to_csv(
        config.output.with_name(config.output.stem + "_puzzles.csv"), index=False
    )
    return frame


def parse_config() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    type_hints = get_type_hints(Config)
    for field in dataclasses.fields(Config):
        if type_hints[field.name] is Path:
            parser.add_argument(f"--{field.name}", type=Path, required=True)
        elif field.name == "protocol":
            parser.add_argument("--protocol",
                                choices=[p.value for p in Protocol],
                                default=field.default)
        elif field.name == "limit_per_bin":
            parser.add_argument("--limit_per_bin", type=int, default=None)
        else:
            parser.add_argument(f"--{field.name}",
                                type=type_hints[field.name],
                                default=field.default)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    config = parse_config()
    started = time.time()
    frame = run_config(config)
    print(frame.to_string(index=False))
    print(f"Saved to {config.output} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
