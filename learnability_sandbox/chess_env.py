"""Chess-puzzle state transitions for the released paper data."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal

import chess


FailureReason = Literal["illegal_move", "wrong_move"]


class Protocol(str, enum.Enum):
    """The two released transition contracts: the paper's stated protocol
    and the released worker's actual behaviour."""

    STRICT_TERMINATION = "strict_termination"
    TEACHER_FORCED_REPLY_REPLAY = "teacher_forced_reply_replay"


def split_release_line(moves: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Split a released ``Moves`` string into (trigger, solver, opponent).

    The released layout interleaves one trigger move, then alternating
    solver and opponent moves; this is the single definition of that layout.
    """
    line = tuple(moves.split())
    return line[0], line[1::2], line[2::2]


@dataclass(frozen=True)
class ChessPuzzle:
    """A validated alternating solution line from the released dataset."""

    initial_fen: str
    trigger_move: str
    solver_moves: tuple[str, ...]
    opponent_moves: tuple[str, ...]

    def __post_init__(self) -> None:
        board = chess.Board(self.initial_fen)
        for move_index, move_uci in enumerate(self.reference_line):
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                raise ValueError(
                    f"reference move {move_index} is illegal: {move_uci!r}"
                )
            board.push(move)

    @classmethod
    def from_release(cls, initial_fen: str, moves: str) -> ChessPuzzle:
        if len(moves.split()) < 2 or len(moves.split()) % 2 != 0:
            raise ValueError("released puzzle lines must contain 2H moves")
        trigger, solver, opponent = split_release_line(moves.lower())
        return cls(
            initial_fen=initial_fen,
            trigger_move=trigger,
            solver_moves=solver,
            opponent_moves=opponent,
        )

    def _line_pairs(self):
        """(solver, opponent) pairs preceding the final solver move — the
        one pairing of the alternating line."""
        return zip(self.solver_moves[:-1], self.opponent_moves, strict=True)

    @property
    def reference_line(self) -> tuple[str, ...]:
        moves = [self.trigger_move]
        for solver_move, opponent_move in self._line_pairs():
            moves.extend((solver_move, opponent_move))
        moves.append(self.solver_moves[-1])
        return tuple(moves)

    def opponent_turns(self):
        """Walk the reference line, yielding each opponent reply in position.

        Yields ``(board, opponent_move)`` per reply, with the board an
        independent copy positioned after the preceding solver move — the
        state an environment renders the reply from. The puzzle owns its
        line; consumers outside this module must not re-derive this walk.
        """
        board = chess.Board(self.initial_fen)
        board.push(chess.Move.from_uci(self.trigger_move))
        for solver_move, opponent_move in self._line_pairs():
            board.push(chess.Move.from_uci(solver_move))
            move = chess.Move.from_uci(opponent_move)
            yield board.copy(stack=False), move
            board.push(move)


@dataclass(frozen=True)
class ChessTransition:
    opponent_move: str | None
    reward: float
    terminated: bool
    failure_reason: FailureReason | None


class _ReferenceLineEnvironment:
    """Own board state shared by the two released puzzle protocols."""

    def __init__(self, puzzle: ChessPuzzle) -> None:
        self.puzzle = puzzle
        self._initial_board = chess.Board(puzzle.initial_fen)
        self._initial_board.push(chess.Move.from_uci(puzzle.trigger_move))
        self._board: chess.Board
        self._solver_move_index: int
        self._terminated: bool
        self._reset_reference_state()

    @property
    def solver_move_index(self) -> int:
        return self._solver_move_index

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def legal_actions(self) -> tuple[str, ...]:
        return tuple(move.uci() for move in self._board.legal_moves)

    @property
    def turn(self) -> chess.Color:
        return self._board.turn

    def reset(self) -> None:
        self._reset_reference_state()

    def _reset_reference_state(self) -> None:
        self._board = self._initial_board.copy(stack=False)
        self._solver_move_index = 0
        self._terminated = False

    def _advance_reference_line(self) -> str | None:
        move_index = self._solver_move_index
        solver_move = self.puzzle.solver_moves[move_index]
        self._board.push(chess.Move.from_uci(solver_move))
        self._solver_move_index += 1

        if self._solver_move_index == len(self.puzzle.solver_moves):
            self._terminated = True
            return None

        opponent_move = self.puzzle.opponent_moves[move_index]
        self._board.push(chess.Move.from_uci(opponent_move))
        return opponent_move


class StrictChessPuzzleEnvironment(_ReferenceLineEnvironment):
    """Apply the paper's immediate-termination puzzle protocol."""

    def step(self, action: str) -> ChessTransition:
        if self._terminated:
            raise RuntimeError("cannot step a terminated puzzle")

        try:
            move = chess.Move.from_uci(action)
        except ValueError:
            return self._terminate("illegal_move")

        if move not in self._board.legal_moves:
            return self._terminate("illegal_move")

        expected_action = self.puzzle.solver_moves[self._solver_move_index]
        if action != expected_action:
            return self._terminate("wrong_move")

        opponent_move = self._advance_reference_line()
        if opponent_move is None:
            return ChessTransition(
                opponent_move=None,
                reward=1.0,
                terminated=True,
                failure_reason=None,
            )

        return ChessTransition(
            opponent_move=opponent_move,
            reward=0.0,
            terminated=False,
            failure_reason=None,
        )

    def _terminate(self, reason: FailureReason) -> ChessTransition:
        self._terminated = True
        return ChessTransition(
            opponent_move=None,
            reward=0.0,
            terminated=True,
            failure_reason=reason,
        )


class TeacherForcedReplyEnvironment(_ReferenceLineEnvironment):
    """Replay recorded replies and score the full solver sequence at the end."""

    def __init__(self, puzzle: ChessPuzzle) -> None:
        super().__init__(puzzle)
        self._all_actions_correct = True

    def reset(self) -> None:
        super().reset()
        self._all_actions_correct = True

    def step(self, action: str) -> ChessTransition:
        if self._terminated:
            raise RuntimeError("cannot step a terminated puzzle")

        expected_action = self.puzzle.solver_moves[self._solver_move_index]
        self._all_actions_correct &= action == expected_action
        opponent_move = self._advance_reference_line()

        if opponent_move is not None:
            return ChessTransition(
                opponent_move=opponent_move,
                reward=0.0,
                terminated=False,
                failure_reason=None,
            )

        reward = float(self._all_actions_correct)
        return ChessTransition(
            opponent_move=None,
            reward=reward,
            terminated=True,
            failure_reason=None if reward else "wrong_move",
        )
