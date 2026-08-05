"""Loss coordinates of a checkpoint on the puzzle panel (experiment 1).

Total cross-entropy mixes calibration with decision competence; the program's
claim 2 holds that the statistic transferring across training histories is
loss at genuine decision points, weighted by how much choice the position
offers. Chess makes that exact: this evaluator scores each puzzle's reference
line teacher-forced and records, per solver decision, the negative
log-likelihood of the correct move's tokens and the board's legal-move count
at that state. From one forward pass per puzzle it reports

- ``total_ce``: mean cross-entropy over all scored tokens (the coordinate the
  paper's law uses);
- ``decision_ce``: mean per-decision NLL over decisions with more than one
  legal move;
- ``entropy_weighted_ce``: decision NLL weighted by ``log2(branching)``, so
  forced moves carry zero weight and wide choices dominate;
- ``line_nll``: the sum of NLL over every solver-move span, forced moves
  included. ``exp(-line_nll)`` is the probability of producing the full
  solver line move-by-move with opponent replies injected. On the pretrain
  vocabulary this is the policy's exact task success probability ``q``. On
  the SFT/RL vocabulary the rollout grammar generates a latent think block
  before each move, so ``exp(-line_nll)`` is the *reflex* success
  probability — the direct-move distribution without thinking — and its
  fidelity to sampled rollout success is a measured question
  (``qhat_validation``), not an identity.

Contrast mass, the third coordinate, needs rollouts and already exists:
``evaluate_checkpoint``'s per-puzzle CSV yields ``M(q̂, K)`` directly.

Pretraining checkpoints (81-token vocabulary) score the prompt without its
``<T>`` marker; SFT/RL checkpoints score it verbatim. Continuation moves are
rendered from the board, so their words are vocabulary tokens by construction
and token spans align exactly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from math import log2
from pathlib import Path
from typing import get_type_hints

import chess
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import Qwen3ForCausalLM

from .lan_tokenizer import THINK, LanTokenizer, render_move
from .puzzle_data import MultiTurnPuzzle, load_multi_turn_puzzles


@dataclass(frozen=True)
class Config:
    checkpoint: Path
    eval_dir: Path
    output: Path
    device: str = "auto"
    batch_size: int = 32
    limit_per_bin: int | None = None

    def __post_init__(self) -> None:
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


def tokenizer_for_checkpoint(checkpoint: Path) -> LanTokenizer:
    """The vendored layout matching the checkpoint's shipped vocabulary."""
    shipped = json.loads((checkpoint / "vocab.json").read_text())
    for tokenizer in (LanTokenizer.pretrain(), LanTokenizer.sft()):
        if tokenizer.vocab == shipped:
            return tokenizer
    raise ValueError(f"checkpoint {checkpoint} ships an unknown vocabulary")


@dataclass(frozen=True)
class ScoredSequence:
    """One puzzle's token ids with its solver-decision spans."""

    ids: list[int]
    # Per solver decision: (first token index, token count, legal-move count).
    decision_spans: tuple[tuple[int, int, int], ...]


def build_sequence(item: MultiTurnPuzzle, tokenizer: LanTokenizer) -> ScoredSequence:
    """Tokenize prompt plus reference continuation, marking decision spans.

    The prompt already ends at the trigger move; the continuation interleaves
    solver and opponent moves down the line, rendered at the board.
    """
    prompt = item.prompt
    if not tokenizer.extra_tokens:
        prompt = prompt[: prompt.rfind(THINK)].rstrip()
    ids = list(tokenizer.encode_prompt(prompt))

    puzzle = item.puzzle
    board = chess.Board(puzzle.initial_fen)
    board.push(chess.Move.from_uci(puzzle.trigger_move))
    spans = []
    for depth, solver_move in enumerate(puzzle.solver_moves):
        branching = board.legal_moves.count()
        words = render_move(board, chess.Move.from_uci(solver_move))
        spans.append((len(ids), len(words), branching))
        ids.extend(tokenizer.vocab[word] for word in words)
        if depth < len(puzzle.opponent_moves):
            words = render_move(board, chess.Move.from_uci(puzzle.opponent_moves[depth]))
            ids.extend(tokenizer.vocab[word] for word in words)
    return ScoredSequence(ids=ids, decision_spans=tuple(spans))


@torch.no_grad()
def score_batch(
    model: Qwen3ForCausalLM,
    sequences: list[ScoredSequence],
    pad_id: int,
    device: torch.device,
) -> list[dict]:
    """Per-puzzle loss coordinates from one right-padded forward pass."""
    width = max(len(s.ids) for s in sequences)
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for i, sequence in enumerate(sequences):
        input_ids[i, : len(sequence.ids)] = torch.tensor(sequence.ids)
        attention_mask[i, : len(sequence.ids)] = 1
    logits = model(
        input_ids.to(device), attention_mask=attention_mask.to(device)
    ).logits.float()
    log_probs = F.log_softmax(logits, dim=-1)

    rows = []
    for i, sequence in enumerate(sequences):
        n = len(sequence.ids)
        targets = torch.tensor(sequence.ids[1:], device=device)
        token_nll = -log_probs[i, : n - 1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        decisions = []
        for start, count, branching in sequence.decision_spans:
            # Position j is predicted by logits at j - 1.
            nll = token_nll[start - 1 : start - 1 + count].sum().item()
            decisions.append((nll, branching))
        uncertain = [(nll, log2(b)) for nll, b in decisions if b > 1]
        weight_sum = sum(w for _, w in uncertain)
        rows.append({
            "total_ce": (token_nll.sum() / (n - 1)).item(),
            "num_tokens": n - 1,
            "line_nll": sum(nll for nll, _ in decisions),
            "decision_ce": (
                sum(nll for nll, _ in uncertain) / len(uncertain) if uncertain else 0.0
            ),
            "entropy_weighted_ce": (
                sum(nll * w for nll, w in uncertain) / weight_sum if weight_sum else 0.0
            ),
            "num_decisions": len(decisions),
            "num_uncertain_decisions": len(uncertain),
        })
    return rows


def run_config(config: Config) -> pd.DataFrame:
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    tokenizer = tokenizer_for_checkpoint(config.checkpoint)
    model = Qwen3ForCausalLM.from_pretrained(config.checkpoint, dtype=torch.float32)
    model = model.to(device).eval()

    parquets = sorted(config.eval_dir.glob("test_B*_multi_turn.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no test_B*_multi_turn.parquet under {config.eval_dir}")
    puzzle_rows = []
    for parquet in parquets:
        puzzles = load_multi_turn_puzzles(parquet)
        if config.limit_per_bin is not None:
            puzzles = puzzles[: config.limit_per_bin]
        sequences = [build_sequence(item, tokenizer) for item in puzzles]
        for start in range(0, len(sequences), config.batch_size):
            batch = sequences[start:start + config.batch_size]
            scored = score_batch(model, batch, tokenizer.bos_id, device)
            for item, row in zip(puzzles[start:start + config.batch_size], scored):
                row.update({
                    "data_source": item.data_source,
                    "fen": item.puzzle.initial_fen,
                    "horizon": len(item.puzzle.solver_moves),
                    "rating": item.rating,
                    "checkpoint": str(config.checkpoint),
                })
                puzzle_rows.append(row)
        print(f"{parquet.stem}: {len(puzzles)} puzzles scored", flush=True)

    frame = pd.DataFrame(puzzle_rows)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.output, index=False)
    reflex_q = np.exp(-frame.line_nll)
    summary = {
        "checkpoint": str(config.checkpoint),
        "total_ce": (frame.total_ce * frame.num_tokens).sum() / frame.num_tokens.sum(),
        "decision_ce": frame.decision_ce.mean(),
        "entropy_weighted_ce": frame.entropy_weighted_ce.mean(),
        "reflex_success_probability": reflex_q.mean(),
        "reflex_contrast_mass_k8": (1 - reflex_q**8 - (1 - reflex_q) ** 8).mean(),
        "num_puzzles": len(frame),
    }
    print(summary)
    return frame


def parse_config() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    type_hints = get_type_hints(Config)
    for field in dataclasses.fields(Config):
        if type_hints[field.name] is Path:
            parser.add_argument(f"--{field.name}", type=Path, required=True)
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
    run_config(config)
    print(f"Saved {config.output} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
