"""Vendored LAN chess tokenizer from the released pre2post-chess pipeline.

Reproduces, token-for-token, the WordLevel tokenizer that ships inside the
released checkpoints (source frozen at pavelslab-nyu/pre2post-chess@256e8b64;
golden parity fixtures in tests/fixtures/lan_tokenizer_golden.json). A move is
a whitespace word in long algebraic notation and tokenizes into its parts:
``Pe2e4`` -> ``P e2 e4``, ``Nd4xe6+`` -> ``N d4 x e6 +``, ``Pe7e8=Q`` ->
``P e7 e8 = Q``, castling stays one token.

Two released vocabulary layouts exist:

- :meth:`LanTokenizer.pretrain`: 81 tokens, shipped with pretraining
  checkpoints.
- :meth:`LanTokenizer.sft`: 85 tokens, appending ``<T> </T> <sep>
  <call_env>``, shipped with SFT and RL checkpoints (RL reuses the SFT
  embedding table, so both stages share this layout).

The released configuration is fixed here by construction: no move-number
tokens, no game-result tokens, no reward tokens. Encode's input domain is the
pipeline's real one — SAN or LAN game prefixes (optionally ending in ``<T>``),
bare move words such as environment replies, and previously decoded full
sequences. Text containing special tokens but no ``<T>`` is outside that
domain and raises, exactly as the frozen source does.
"""

from __future__ import annotations

import contextlib
import io
import os
import re

import chess
import chess.pgn

BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
THINK = "<T>"
THINK_END = "</T>"
SEP = "<sep>"
CALL_ENV = "<call_env>"
SFT_TOKENS = (THINK, THINK_END, SEP, CALL_ENV)

_PIECES = "KQRBNP"
_FILES = "abcdefgh"
_RANKS = "12345678"
_SQUARES = tuple(f + r for f in _FILES for r in _RANKS)
_OPERATORS = ("x", "=", "+", "#", "O-O", "O-O-O", ".", "...")

_LAN_MOVE = re.compile(r"^([PNBRQK])([a-h][1-8])(x)?([a-h][1-8])(=([QRBN]))?$")


def is_complete_move(word: str) -> bool:
    """True when ``word`` is a complete LAN move (the released reward check)."""
    if not word:
        return False
    move = word.rstrip("+#")
    if move in ("O-O", "O-O-O"):
        return True
    return _LAN_MOVE.match(move) is not None


def lan_to_uci(lan: str, side_to_move: chess.Color) -> str:
    """Convert one LAN move word to UCI, as the released reward function does.

    Castling needs the side to move because UCI spells it as a king move.
    Raises ValueError on words that are not complete LAN moves.
    """
    move = lan.rstrip("+#")
    if move == "O-O":
        return "e1g1" if side_to_move == chess.WHITE else "e8g8"
    if move == "O-O-O":
        return "e1c1" if side_to_move == chess.WHITE else "e8c8"
    match = _LAN_MOVE.match(move)
    if match is None:
        raise ValueError(f"not a complete LAN move: {lan!r}")
    _, from_square, _, to_square, _, promotion = match.groups()
    return from_square + to_square + (promotion.lower() if promotion else "")


def render_move(board: chess.Board, move: chess.Move) -> list[str]:
    """LAN token words for a legal ``move``, derived from the board (capture
    and check/mate suffixes included), pushing ``move`` onto ``board``.

    ``"".join(render_move(...))`` is the rendered move word — the exact form
    of the parquet's pre-rendered environment replies."""
    if board.is_castling(move):
        words = ["O-O" if chess.square_file(move.to_square) == 6 else "O-O-O"]
    else:
        words = [
            board.piece_at(move.from_square).symbol().upper(),
            chess.square_name(move.from_square),
        ]
        if board.is_capture(move):
            words.append("x")
        words.append(chess.square_name(move.to_square))
        if move.promotion:
            words += ["=", chess.piece_symbol(move.promotion).upper()]
    board.push(move)
    if board.is_checkmate():
        words.append("#")
    elif board.is_check():
        words.append("+")
    return words


def _word_split_move(word: str) -> list[str]:
    """Split one textual LAN move word into tokens (text-derived path).

    Mirrors the frozen source: malformed words pass through unchanged and map
    to ``<unk>`` at lookup time.
    """
    if word.rstrip("+#") in ("O-O", "O-O-O"):
        base = word.rstrip("+#")
        suffix = word[len(base):]
        return [base] + ([suffix] if suffix else [])
    if word[0] not in _PIECES:
        return [word]
    out = [word[0]]
    i, n = 1, len(word)
    if i + 1 < n and word[i] in _FILES and word[i + 1] in _RANKS:
        out.append(word[i:i + 2])
        i += 2
    if i < n and word[i] == "x":
        out.append("x")
        i += 1
    if i + 1 < n and word[i] in _FILES and word[i + 1] in _RANKS:
        out.append(word[i:i + 2])
        i += 2
    if i < n and word[i] == "=":
        out.append("=")
        i += 1
        if i < n and word[i] in "QRBN":
            out.append(word[i])
            i += 1
    if i < n and word[i] in "+#":
        out.append(word[i])
    return out


class LanTokenizer:
    """WordLevel tokenizer over LAN move words and special tokens."""

    def __init__(self, extra_tokens: tuple[str, ...]) -> None:
        tokens = [BOS, EOS, UNK, *_PIECES, *_SQUARES, *_OPERATORS, *extra_tokens]
        self.extra_tokens = extra_tokens
        self.vocab: dict[str, int] = {tok: i for i, tok in enumerate(tokens)}
        self._id_to_token = tokens
        self.bos_id = self.vocab[BOS]
        self.eos_id = self.vocab[EOS]
        self.unk_id = self.vocab[UNK]

    @classmethod
    def pretrain(cls) -> "LanTokenizer":
        """The 81-token layout shipped with pretraining checkpoints."""
        return cls(())

    @classmethod
    def sft(cls) -> "LanTokenizer":
        """The 85-token layout shipped with SFT and RL checkpoints."""
        return cls(SFT_TOKENS)

    # ---- encode ----

    def encode(self, text: str) -> list[int]:
        """Token ids as ``[<bos>] + words + [<eos>]``, unknown words -> <unk>."""
        words = [BOS, *self._text_words(text), EOS]
        return [self.vocab.get(word, self.unk_id) for word in words]

    def encode_moves(self, text: str) -> list[int]:
        """Token ids without the ``<bos>``/``<eos>`` sentinels.

        This is how the released worker tokenizes environment replies before
        injecting them into the rolling context.
        """
        return self.encode(text)[1:-1]

    def encode_prompt(self, text: str) -> list[int]:
        """Token ids with ``<bos>`` kept and the trailing ``<eos>`` stripped.

        This is the checkpoint-shipped HF wrapper's ``add_special_tokens``
        path — the exact prompt form the released rollout engine feeds the
        model.
        """
        return self.encode(text)[:-1]

    def _text_words(self, text: str) -> list[str]:
        if self.extra_tokens and any(tok in text for tok in self.extra_tokens):
            think_at = text.index(THINK)
            prompt_part = text[:think_at].strip()
            prompt_words = self._game_words(prompt_part) if prompt_part else None
            if prompt_words is None:
                prompt_words = self._free_words(prompt_part) if prompt_part else []
            return prompt_words + self._free_words(text[think_at:])
        game_words = self._game_words(text)
        if game_words:
            return game_words
        return [w for word in text.split() for w in _word_split_move(word)]

    def _game_words(self, text: str) -> list[str] | None:
        """Parse ``text`` as a game (SAN or LAN movetext) into board-derived
        LAN words; None when it is not a game at all. Illegal continuations
        truncate the mainline, matching the frozen source's lenient parse."""
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            game = chess.pgn.read_game(io.StringIO(text))
        if game is None:
            return None
        board = game.board()
        words: list[str] = []
        for move in game.mainline_moves():
            words += render_move(board, move)
        return words

    def _free_words(self, text: str) -> list[str]:
        """Tokenize CoT/interaction text: special tokens and results pass
        through, digit runs split per digit, everything else is a move word."""
        out: list[str] = []
        for word in text.split():
            if word in self.extra_tokens:
                out.append(word)
            elif word[0].isdigit() and "." in word:
                number = word.rstrip(".")
                dots = word[len(number):]
                out += list(number)
                if dots:
                    out.append("..." if len(dots) > 1 else ".")
            elif all(c.isdigit() for c in word):
                out += list(word)
            else:
                out += _word_split_move(word)
        return out

    # ---- decode ----

    def decode(self, ids: list[int]) -> str:
        """Reassemble LAN move words from a token stream.

        Greedy grouping identical to the frozen source: a piece letter
        consumes the following tokens as from-square/capture/to-square/
        promotion/suffix without validation, because model-generated streams
        are arbitrary and the released reward parsing sees exactly this text.
        """
        words = [
            self._id_to_token[i]
            for i in ids
            if i not in (self.bos_id, self.eos_id)
        ]
        out: list[str] = []
        i, n = 0, len(words)
        while i < n:
            word = words[i]
            if word in self.extra_tokens:
                out.append(word)
                i += 1
            elif word in ("O-O", "O-O-O"):
                j = i + 1
                if j < n and words[j] in ("+", "#"):
                    out.append(word + words[j])
                    j += 1
                else:
                    out.append(word)
                i = j
            elif word in _PIECES:
                j = i + 1
                from_square = words[j] if j < n else ""
                j += 1
                capture = ""
                if j < n and words[j] == "x":
                    capture = "x"
                    j += 1
                to_square = words[j] if j < n else ""
                j += 1
                promotion = ""
                if j + 1 <= n - 1 and words[j] == "=" and words[j + 1] in "QRBN":
                    promotion = "=" + words[j + 1]
                    j += 2
                suffix = ""
                if j < n and words[j] in ("+", "#"):
                    suffix = words[j]
                    j += 1
                out.append(f"{word}{from_square}{capture}{to_square}{promotion}{suffix}")
                i = j
            else:
                out.append(word)
                i += 1
        return " ".join(out)
