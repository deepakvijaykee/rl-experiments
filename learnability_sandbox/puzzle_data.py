"""The token-level puzzle data contract shared by evaluation and training.

A :class:`MultiTurnPuzzle` couples a validated :class:`ChessPuzzle` line with
the prompt text the model sees. The environment-reply texts are *derived*
from the line — rendered at the board via :func:`render_move`, the same
rendering the released pipeline used for its ``env_replies`` — never stored
alongside it: a puzzle whose replies disagree with its line is
unrepresentable. Parquets do persist the reply texts for schema parity with
the released training data, so the loader checks the stored texts against
the derivation once, at that boundary; ``tests/verify_released_parity.py``
re-derives every reply of the released training parquet the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd

from .chess_env import ChessPuzzle
from .lan_tokenizer import render_move


def rendered_replies(puzzle: ChessPuzzle) -> tuple[str, ...]:
    """The LAN reply texts an environment injects, rendered in line order
    from the puzzle's own walk — the single derivation of reply texts."""
    return tuple(
        "".join(render_move(board, move))
        for board, move in puzzle.opponent_turns()
    )


@dataclass(frozen=True)
class MultiTurnPuzzle:
    """One puzzle in the token-level interaction format."""

    puzzle: ChessPuzzle
    prompt: str
    data_source: str
    rating: int | None

    @cached_property
    def env_replies(self) -> tuple[str, ...]:
        return rendered_replies(self.puzzle)

    @property
    def target_moves(self) -> tuple[str, ...]:
        """The reward function's view of the line: the solver moves."""
        return self.puzzle.solver_moves


def to_release_row(
    item: MultiTurnPuzzle,
    moves: str,
    puzzle_id: str | None,
    themes: str | None,
) -> dict:
    """Serialize one puzzle to the released parquet row schema.

    Paired with :func:`load_multi_turn_puzzles` below — writer and reader of
    the schema live at this single site.
    """
    return {
        "data_source": item.data_source,
        "prompt": item.prompt,
        "ability": "chess",
        "reward_model": {
            "ground_truth": str(list(item.target_moves)),
            "style": "rule",
        },
        "extra_info": {
            "FEN": item.puzzle.initial_fen,
            "Moves": moves,
            "env_replies": list(item.env_replies),
            "PuzzleId": puzzle_id,
            "Rating": item.rating,
            "Themes": themes,
        },
        "difficulty": item.rating,
    }


def load_multi_turn_puzzles(path: Path) -> list[MultiTurnPuzzle]:
    """Load a released-schema puzzle parquet for token-level interaction."""
    frame = pd.read_parquet(path)
    puzzles = []
    for row in frame.itertuples(index=False):
        info = row.extra_info
        rating = info["Rating"]
        item = MultiTurnPuzzle(
            puzzle=ChessPuzzle.from_release(info["FEN"], info["Moves"]),
            prompt=row.prompt,
            data_source=row.data_source,
            rating=int(rating) if rating is not None else None,
        )
        if tuple(info["env_replies"]) != item.env_replies:
            raise ValueError(
                f"stored env replies disagree with the puzzle line in {path}"
            )
        puzzles.append(item)
    return puzzles
