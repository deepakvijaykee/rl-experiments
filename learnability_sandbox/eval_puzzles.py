"""Reconstruct the withheld B1-B5 test puzzles from released eval logs.

The released code evaluates on ``test_B{1..5}_multi_turn.parquet`` files that
were not published. Their content is recoverable from the released eval
generation logs (chess-rl-eval): each row carries the puzzle prompt, its
difficulty bin, and the ground-truth solver line, and — because the released
worker injects the recorded opponent reply after every ``<call_env>``
regardless of the submitted move — the reply for turn ``k`` appears verbatim
after the ``k``-th ``<call_env>`` in every rollout that reached that turn.

Reconstruction replays the prompt from the starting position and takes the
full line from the open Lichess puzzle database (matched on the pre-trigger
FEN plus the trigger and solver moves, which identify a puzzle immutably even
as its rating drifts). Replies observed in the logs — checked unanimous across
legality-validated candidates — must agree with the database line; a handful
of deep turns that no logged rollout reached (the 2560-token budget runs out
first) exist only in the database. Every recovered line passes the
ChessPuzzle contract. Output parquets use the training parquet's schema, so
one loader (``puzzle_data``) serves both.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chess
import pandas as pd

from .chess_env import ChessPuzzle, split_release_line
from .lan_tokenizer import CALL_ENV, THINK, lan_to_uci
from .puzzle_data import MultiTurnPuzzle, to_release_row


def replay_prompt(input_text: str) -> tuple[str, str]:
    """Replay a LAN prompt from the starting position.

    Returns (initial_fen, trigger_uci): the position before the prompt's last
    move, and that move — the released puzzle contract's reset state.
    """
    words = input_text.split()
    if words[-1] == THINK:
        words = words[:-1]
    board = chess.Board()
    moves = []
    for word in words:
        move = chess.Move.from_uci(lan_to_uci(word, board.turn))
        if move not in board.legal_moves:
            raise ValueError(f"prompt move {word!r} is illegal during replay")
        moves.append(move)
        board.push(move)
    board.pop()
    return board.fen(), moves[-1].uci()


def _observed_reply_prefix(
    board: chess.Board,
    solver_moves: tuple[str, ...],
    outputs: list[str],
) -> list[str]:
    """The consecutively observed opponent replies (UCIs), from turn 0.

    Observation is prefix-shaped: recovering turn ``k`` requires walking the
    board through reply ``k - 1``, so the first unobservable turn ends
    recovery. ``board`` is positioned after the trigger move and is walked
    down the line. Teacher forcing guarantees a unanimous reply at every reached
    turn; disagreement among legality-validated candidates raises. For turn ``k``
    the candidate reply is the first word after the ``k``-th ``<call_env>``.
    """
    segmented = [output.split(CALL_ENV) for output in outputs]
    replies: list[str] = []
    for k, solver in enumerate(solver_moves[:-1]):
        board.push(chess.Move.from_uci(solver))
        candidates: set[str] = set()
        for segments in segmented:
            if len(segments) <= k + 1:
                continue
            segment_words = segments[k + 1].split()
            if not segment_words:
                continue
            try:
                uci = lan_to_uci(segment_words[0], board.turn)
            except ValueError:
                continue
            if chess.Move.from_uci(uci) in board.legal_moves:
                candidates.add(uci)
        if len(candidates) > 1:
            raise ValueError(
                f"teacher forcing guarantees a unanimous reply at turn {k}, "
                f"got {sorted(candidates)}"
            )
        if not candidates:
            break
        (uci,) = candidates
        board.push(chess.Move.from_uci(uci))
        replies.append(uci)
    return replies


def _position_key(fen: str) -> str:
    """Piece placement, side, castling, and en passant — no move counters.

    The released prompt truncates the source game at the *first* occurrence
    of the puzzle's piece placement, so a position that repeated in the game
    replays with smaller counters than the database FEN carries.
    """
    return " ".join(fen.split()[:4])


def _load_lichess_lines(
    csv_path: Path,
    position_keys: set[str],
) -> dict[str, list[dict[str, str]]]:
    """Stream the Lichess puzzle database, keeping rows for needed positions."""
    lines: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = _position_key(row["FEN"])
            if key in position_keys:
                lines[key].append(row)
    return lines


@dataclass(frozen=True)
class ResolvedLine:
    """The adopted full move line for one test puzzle, with its provenance."""

    moves: str
    source: Literal["lichess", "logs"]
    puzzle_id: str | None
    rating: int | None
    themes: str | None


def _resolve_line(
    candidates: list[dict[str, str]],
    trigger: str,
    solver_moves: tuple[str, ...],
    observed: list[str],
) -> ResolvedLine:
    """Adopt one puzzle's full line from the database or the logs.

    Identity is position + trigger + solver line + observed-reply prefix:
    trivial endgame positions recur across source games with the same
    solution but different recorded replies, and the observed replies (exact
    under teacher forcing) pick the crawled game's line among them. A puzzle
    deleted from the database is adoptable from the logs alone when its
    complete reply line was observed.
    """
    matches = []
    for row in candidates:
        row_trigger, row_solver, row_opponent = split_release_line(row["Moves"])
        if (row_trigger == trigger
                and row_solver == solver_moves
                and list(row_opponent[:len(observed)]) == observed):
            matches.append(row)
    if matches and all(r["Moves"] == matches[0]["Moves"] for r in matches):
        match = min(matches, key=lambda r: r["PuzzleId"])
        return ResolvedLine(
            moves=match["Moves"],
            source="lichess",
            puzzle_id=match["PuzzleId"],
            rating=int(match["Rating"]),
            themes=match["Themes"],
        )
    if not matches and len(observed) == len(solver_moves) - 1:
        line = [trigger]
        for k, solver in enumerate(solver_moves):
            line.append(solver)
            if k < len(observed):
                line.append(observed[k])
        return ResolvedLine(
            moves=" ".join(line),
            source="logs",
            puzzle_id=None,
            rating=None,
            themes=None,
        )
    raise ValueError(
        f"{len(matches)} distinct Lichess lines match puzzle line "
        f"{[trigger, *solver_moves]} with observed replies {observed}"
    )


def reconstruct_test_sets(
    generations_dir: Path,
    lichess_csv: Path,
    output_dir: Path,
) -> None:
    rows_by_puzzle: dict[tuple[str, str], dict] = {}
    for path in sorted(generations_dir.rglob("generations/*.jsonl")):
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                key = (row["data_source"], row["input"])
                entry = rows_by_puzzle.setdefault(
                    key, {"ground_truth": row["ground_truth"], "outputs": []}
                )
                if entry["ground_truth"] != row["ground_truth"]:
                    raise ValueError(f"conflicting ground truth for {key[1][:60]!r}")
                entry["outputs"].append(row["output"])

    replayed = {
        key: replay_prompt(key[1]) for key in rows_by_puzzle
    }
    lichess = _load_lichess_lines(
        lichess_csv, {_position_key(fen) for fen, _ in replayed.values()}
    )

    records = defaultdict(list)
    observed_turns = log_only = 0
    for (data_source, input_text), entry in sorted(rows_by_puzzle.items()):
        solver_moves = tuple(ast.literal_eval(entry["ground_truth"]))
        initial_fen, trigger = replayed[(data_source, input_text)]

        observed_board = chess.Board(initial_fen)
        observed_board.push(chess.Move.from_uci(trigger))
        observed = _observed_reply_prefix(
            observed_board, solver_moves, entry["outputs"]
        )
        observed_turns += len(observed)

        resolved = _resolve_line(
            lichess.get(_position_key(initial_fen), []),
            trigger,
            solver_moves,
            observed,
        )
        log_only += resolved.source == "logs"

        # ChessPuzzle validates the adopted line; the reply texts are
        # derived from it by MultiTurnPuzzle, never assembled here.
        reconstructed = MultiTurnPuzzle(
            puzzle=ChessPuzzle.from_release(initial_fen, resolved.moves),
            prompt=input_text,
            data_source=data_source,
            rating=resolved.rating,
        )
        records[data_source].append(to_release_row(
            reconstructed,
            moves=resolved.moves,
            puzzle_id=resolved.puzzle_id,
            themes=resolved.themes,
        ))

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"reply turns observed in logs (unanimity checked, prefix-matched "
          f"against adopted lines): {observed_turns}; "
          f"puzzles absent from the database (log-only lines): {log_only}")
    for data_source, rows in sorted(records.items()):
        path = output_dir / f"{data_source}_multi_turn.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        print(f"{path}: {len(rows)} puzzles")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations_dir", type=Path, required=True)
    parser.add_argument("--lichess_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.time()
    reconstruct_test_sets(
        arguments.generations_dir, arguments.lichess_csv, arguments.output_dir
    )
    print(f"done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
