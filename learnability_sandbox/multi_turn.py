"""Token-level multi-turn puzzle rollout and scoring from the released RL worker.

Ports, per sample, the interaction contract of ``generate_multi_turn_sequences``
(``fsdp_workers.py``) and the trajectory scoring of
``reward_function_multiturn.py`` from pre2post-chess @ 256e8b64 (parity
fixtures in tests/fixtures). The contract:

- The rolling context starts at the encoded prompt (ending in ``<T>``). Each
  round the model generates until ``<call_env>``, EOS, or its token budget.
- On ``<call_env>``, the environment reply is injected into the context with
  ``policy_mask`` 0; model tokens carry ``policy_mask`` 1. The released worker
  injects the next *recorded* reply without checking the submitted move
  (teacher-forced replay); the paper's stated protocol instead terminates on a
  wrong move. Both are reply protocols here.

Two rewards exist deliberately and must not be conflated. The released
scorer (:func:`score_solver_sequence`) decodes the response, splits on
``<call_env>``, extracts one move per segment (first complete move after
``</T>``, else the last complete move in the segment), and converts to UCI
with castling always mapped to White's squares — defects preserved because
its purpose is parity with released logs and published numbers. The
sandbox's canonical reward (:func:`board_verdict`) replays the submitted
moves — extracted per turn at generation time by the same released grammar —
through the strict board environment, which owns the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import chess

from .chess_env import ChessPuzzle, StrictChessPuzzleEnvironment
from .lan_tokenizer import (
    CALL_ENV,
    THINK_END,
    LanTokenizer,
    is_complete_move,
    lan_to_uci,
)
from .puzzle_data import rendered_replies


# ---------------------------------------------------------------------------
# Trajectory scoring (reward_function_multiturn.py port)
# ---------------------------------------------------------------------------

def _complete_move_in(words: Iterable[str]) -> str | None:
    """First complete LAN move in ``words``."""
    for word in words:
        if is_complete_move(word):
            return word
    return None


def extract_turn_move(segment: str) -> str | None:
    """The solver move submitted in one ``<call_env>`` segment.

    With a closed thinking block, the answer is the first complete move after
    ``</T>``; otherwise the last complete move anywhere in the segment.
    """
    if THINK_END in segment:
        after = segment[segment.find(THINK_END) + len(THINK_END):]
        move = _complete_move_in(after.split())
        if move is not None:
            return move
    return _complete_move_in(reversed(segment.split()))


def extract_solver_moves(response_text: str) -> list[str | None]:
    """One submitted move per environment call, in order."""
    segments = response_text.split(CALL_ENV)
    return [extract_turn_move(segment) for segment in segments[:-1]]


@dataclass(frozen=True)
class ScoreResult:
    score: float
    extracted_ucis: tuple[str, ...]


def score_solver_sequence(
    response_text: str,
    target_moves: Sequence[str],
) -> ScoreResult:
    """The RELEASED reward, defects preserved — a parity instrument only.

    This re-parses decoded text because the released worker has no
    structural record of what was submitted, and it converts castling as
    White regardless of the side to move. Use it to compare against released
    logs and published numbers. The sandbox's own reward is
    :func:`board_verdict`.
    """
    extracted = extract_solver_moves(response_text)
    if not extracted:
        return ScoreResult(0.0, ())
    all_match = len(extracted) == len(target_moves)
    ucis: list[str] = []
    for i, move in enumerate(extracted):
        if move is None:
            ucis.append("")
            all_match = False
            continue
        try:
            # The released reward converts with side_to_move defaulted to
            # White, so Black castling maps to e1g1/e1c1 and scores wrong.
            uci = lan_to_uci(move, chess.WHITE)
        except ValueError:
            uci = move
        ucis.append(uci)
        if i >= len(target_moves) or uci != target_moves[i]:
            all_match = False
    return ScoreResult(1.0 if all_match else 0.0, tuple(ucis))


def _step_submitted(
    environment: StrictChessPuzzleEnvironment,
    submitted_lan: str | None,
):
    """Step the strict environment with a submitted move.

    A turn without an extractable move, or with an unparseable one, passes
    an invalid action string through: ``chess.Move.from_uci`` raises inside
    the environment, which records the turn as an illegal move — the same
    verdict the released reward gives it.
    """
    if submitted_lan is None:
        uci = ""
    else:
        try:
            uci = lan_to_uci(submitted_lan, environment.turn)
        except ValueError:
            uci = submitted_lan
    return environment.step(uci)


def board_verdict(
    submitted_moves: Sequence[str | None],
    puzzle: ChessPuzzle,
) -> float:
    """The sandbox's canonical trajectory reward: the board decides.

    Replays the per-turn submitted moves through the strict environment —
    correct side-to-move conversion, no text re-parsing. Under the strict
    protocol this reproduces the live environment's verdict; under
    teacher-forced replay it scores what the model actually submitted.
    """
    environment = StrictChessPuzzleEnvironment(puzzle)
    for submitted in submitted_moves:
        transition = _step_submitted(environment, submitted)
        if transition.terminated:
            return transition.reward
    return 0.0


# ---------------------------------------------------------------------------
# Reply protocols
# ---------------------------------------------------------------------------

Termination = Literal[
    "eos",
    "no_call_env",
    "budget_exhausted",
    "replies_exhausted",
    "line_completed",
    "wrong_move",
    "illegal_move",
    "max_env_calls",
]


@dataclass(frozen=True)
class ReplyOutcome:
    """One environment call's result: a reply to inject, or a termination."""

    reply: str | None
    termination: Termination | None


class TeacherForcedReplies:
    """The released worker's protocol: pop the next recorded reply on every
    environment call, regardless of what the model submitted."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._queue = list(replies)

    def next_reply(self, submitted_lan: str | None) -> ReplyOutcome:
        if self._queue:
            return ReplyOutcome(self._queue.pop(0), None)
        return ReplyOutcome(None, "replies_exhausted")


class StrictReplies:
    """The paper's stated protocol: the submitted move steps a strict board
    environment; a malformed, illegal, or wrong move terminates, and replies
    derive from the puzzle's own line."""

    def __init__(self, puzzle: ChessPuzzle) -> None:
        self._env = StrictChessPuzzleEnvironment(puzzle)
        self._rendered = rendered_replies(puzzle)

    def next_reply(self, submitted_lan: str | None) -> ReplyOutcome:
        transition = _step_submitted(self._env, submitted_lan)
        if transition.terminated:
            termination = (
                "line_completed" if transition.reward else transition.failure_reason
            )
            return ReplyOutcome(None, termination)
        return ReplyOutcome(self._rendered[self._env.solver_move_index - 1], None)


# ---------------------------------------------------------------------------
# Per-sample rollout state machine
# ---------------------------------------------------------------------------

def _clean_row(row: Sequence[int], eos_id: int) -> list[int]:
    """The released worker's response extraction: drop id-0 tokens (``<bos>``
    and padding share id 0 in every released vocabulary) and stop inclusively
    at the first EOS."""
    tokens: list[int] = []
    for token in row:
        if token == 0:
            continue
        tokens.append(token)
        if token == eos_id:
            break
    return tokens


class MultiTurnRollout:
    """One sample's rolling context under the multi-turn contract.

    The driver owns generation: it feeds ``context`` to the model with
    ``<call_env>`` and EOS as stop tokens and calls :meth:`submit` with the
    raw generated row until ``finished``.
    """

    def __init__(
        self,
        prompt_ids: Sequence[int],
        tokenizer: LanTokenizer,
        replies: TeacherForcedReplies | StrictReplies,
        max_env_calls: int,
        max_model_tokens: int,
    ) -> None:
        self._tokenizer = tokenizer
        self._replies = replies
        self._max_env_calls = max_env_calls
        self._max_model_tokens = max_model_tokens
        self._call_env_id = tokenizer.vocab[CALL_ENV]
        self._eos_id = tokenizer.eos_id
        self._context = list(prompt_ids)
        # The current <call_env> segment: the previous turn's injected reply
        # followed by this turn's model tokens — the single owner of the
        # segment boundary the scorer's text splitting also expresses.
        self._segment_ids: list[int] = []
        self.response_tokens: list[int] = []
        self.policy_mask: list[int] = []
        self.submitted_moves: list[str | None] = []
        self.n_model_tokens = 0
        self.n_env_calls = 0
        self.termination: Termination | None = None

    @property
    def context(self) -> list[int]:
        return self._context

    @property
    def finished(self) -> bool:
        return self.termination is not None

    @property
    def remaining_budget(self) -> int:
        return self._max_model_tokens - self.n_model_tokens

    @property
    def response_text(self) -> str:
        return self._tokenizer.decode(self.response_tokens)

    def submit(self, row: Sequence[int]) -> None:
        """Consume one round's raw generated row."""
        if self.finished:
            raise RuntimeError("cannot submit to a finished rollout")
        tokens = _clean_row(row, self._eos_id)[: self.remaining_budget]
        if not tokens:
            self.termination = "budget_exhausted"
            return

        if self._call_env_id not in tokens:
            self._append_model(tokens)
            self.termination = "eos" if tokens[-1] == self._eos_id else "no_call_env"
            return

        cut = tokens.index(self._call_env_id) + 1
        model_part = tokens[:cut]
        self._append_model(model_part)
        self.n_env_calls += 1

        self._segment_ids.extend(model_part[:-1])
        submitted = extract_turn_move(self._tokenizer.decode(self._segment_ids))
        self.submitted_moves.append(submitted)

        outcome = self._replies.next_reply(submitted)
        if outcome.reply is None:
            self.termination = outcome.termination
            return
        reply_ids = self._tokenizer.encode_moves(outcome.reply)
        self._context.extend(reply_ids)
        self.response_tokens.extend(reply_ids)
        self.policy_mask.extend([0] * len(reply_ids))
        self._segment_ids = list(reply_ids)

        # Parity with the released worker's env-call cap; at the released
        # defaults (6 calls, horizon <= 6) the reply queue or the strict
        # environment terminates first, so this binds only when configured
        # below the puzzle horizon.
        if self.n_env_calls >= self._max_env_calls:
            self.termination = "max_env_calls"

    def _append_model(self, tokens: list[int]) -> None:
        self._context.extend(tokens)
        self.response_tokens.extend(tokens)
        self.policy_mask.extend([1] * len(tokens))
        self.n_model_tokens += len(tokens)
