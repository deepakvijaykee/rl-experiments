"""Parsing, judge rewards, and char-span diagnostics for the RLM flow."""

from __future__ import annotations

import ast
import json
import os
import re
import string
import textwrap
import time
from dataclasses import dataclass
from typing import Any


_FINAL_BLOCK_RE = re.compile(
    r"<final>(.*?)</final>|FINAL\s*[:=]\s*(\[.*\]|\{.*\})",
    flags=re.IGNORECASE | re.DOTALL,
)
_JSON_BLOCK_RE = re.compile(r"```(?:json|python)?\s*(.*?)```", re.DOTALL)
_QUOTED_RE = re.compile(r'"([^"\n]{8,})"|\'([^\'\n]{8,})\'')


@dataclass(frozen=True)
class EvidenceReward:
    score: float
    f1: float
    copy_rate: float
    count_penalty: float
    num_predictions: int
    judge_precision: float | None = None
    judge_recall: float | None = None


def normalize_text(text: str) -> str:
    table = str.maketrans({c: " " for c in string.punctuation})
    return " ".join(text.lower().translate(table).split())


def _flatten_strings(value: Any) -> list[str]:
    if type(value) is str:
        return [value]
    if type(value) is dict:
        out: list[str] = []
        for key in ("text", "evidence", "selection", "selections", "passages"):
            if key in value:
                out.extend(_flatten_strings(value[key]))
        if out:
            return out
        for child in value.values():
            out.extend(_flatten_strings(child))
        return out
    if type(value) in (list, tuple):
        out = []
        for child in value:
            out.extend(_flatten_strings(child))
        return out
    return []


def _parse_literal(text: str) -> list[str]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        return _flatten_strings(value)
    try:
        return _flatten_strings(ast.literal_eval(text))
    except (SyntaxError, ValueError, TypeError):
        return []


def extract_evidence_strings(completion: str) -> list[str]:
    """Extract predicted evidence strings from common final-answer formats."""
    candidates: list[str] = []
    for match in _FINAL_BLOCK_RE.finditer(completion):
        block = next(group for group in match.groups() if group)
        candidates.extend(_parse_literal(block.strip()))

    if not candidates:
        for match in _JSON_BLOCK_RE.finditer(completion):
            candidates.extend(_parse_literal(match.group(1).strip()))

    if not candidates:
        candidates.extend(_parse_literal(completion.strip()))

    if not candidates:
        for match in _QUOTED_RE.finditer(completion):
            candidates.append(match.group(1) or match.group(2))

    if not candidates:
        for line in completion.splitlines():
            cleaned = line.strip(" -*\t\r\n")
            if len(cleaned) >= 16:
                candidates.append(cleaned)

    deduped = []
    seen = set()
    for text in candidates:
        text = " ".join(str(text).split())
        key = normalize_text(text)
        if key and key not in seen:
            seen.add(key)
            deduped.append(text)
    return deduped


def gold_evidence_strings(reward_spec: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for entry in reward_spec["evidence"]:
        for selection in entry["selections"]:
            text = selection["text"]
            if text and text.strip():
                out.append(text)
    return out


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def _union_size(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge_intervals(intervals))


def _intersection_size(
        left: list[tuple[int, int]],
        right: list[tuple[int, int]]) -> int:
    left = _merge_intervals(left)
    right = _merge_intervals(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        lo = max(left[i][0], right[j][0])
        hi = min(left[i][1], right[j][1])
        if lo < hi:
            total += hi - lo
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _find_intervals(strings: list[str], context_text: str) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for text in strings:
        start = context_text.find(text)
        if start >= 0:
            intervals.append((start, start + len(text)))
    return intervals


def char_span_f1(predictions: list[str], targets: list[str],
                 context_text: str) -> float:
    retrieved = _find_intervals(predictions, context_text)
    evidence = _find_intervals(targets, context_text)
    if not retrieved or not evidence:
        return 0.0
    covered = _intersection_size(retrieved, evidence)
    total_retrieved = _union_size(retrieved)
    total_evidence = _union_size(evidence)
    precision = covered / total_retrieved if total_retrieved else 0.0
    recall = covered / total_evidence if total_evidence else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def copied_from_context(predictions: list[str], context_text: str) -> float:
    if not predictions:
        return 0.0
    normalized_context = normalize_text(context_text)
    copied = 0
    for pred in predictions:
        if normalize_text(pred) in normalized_context:
            copied += 1
    return copied / len(predictions)


def evidence_reward(
        completion: str,
        reward_spec: dict[str, Any],
        context_text: str,
        max_predictions: int = 12) -> EvidenceReward:
    """Score a final RLM answer against verifiable evidence spans.

    The main signal is character-span F1 between copied predicted substrings and
    gold evidence substrings located in the provided paper context.
    """
    predictions = extract_evidence_strings(completion)
    targets = gold_evidence_strings(reward_spec)
    f1 = char_span_f1(predictions, targets, context_text)
    copy_rate = copied_from_context(predictions, context_text)
    count_penalty = max(0, len(predictions) - max_predictions) / max(max_predictions, 1)
    score = max(0.0, min(1.0, f1 - 0.05 * count_penalty))
    return EvidenceReward(
        score=score,
        f1=f1,
        copy_rate=copy_rate,
        count_penalty=count_penalty,
        num_predictions=len(predictions),
    )


_RUBRIC_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "RubricScore",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "precision_score": {"type": "integer"},
                "recall_score": {"type": "integer"},
                "reasoning": {"type": "string"},
            },
            "required": ["precision_score", "recall_score", "reasoning"],
            "additionalProperties": False,
        },
    },
}


def _judge_prompt(question: str, targets: list[str], predictions: list[str]) -> str:
    gt_block = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(targets)) or "(none)"
    pred_block = (
        "\n\n".join(f"[{i}] {text}" for i, text in enumerate(predictions))
        or "(none)"
    )
    return textwrap.dedent(
        f"""\
        You are evaluating predicted evidence extractions against ground truth
        evidence for a question. Treat the ground truth evidence as the
        complete reference.

        Question: {question}

        Ground truth evidence:
        {gt_block}

        Predicted evidence:
        {pred_block}

        Score the predicted evidence on two integer 1-10 dimensions:

        Precision: spans are tight, accurate, and free of off-topic padding.
        Recall: spans collectively cover the facts needed to answer the
        question.

        Return only the requested JSON schema. The reward will be the mean of
        precision and recall after scaling each score to 0-1.
        """
    )


def judge_evidence_reward(
        completion: str,
        question: str,
        reward_spec: dict[str, Any],
        model: str,
        base_url: str,
        api_key_env: str,
        timeout_seconds: float,
        max_retries: int) -> EvidenceReward:
    """Score final evidence with an OpenAI-compatible rubric judge."""
    import httpx

    if api_key_env not in os.environ:
        raise RuntimeError(
            f"{api_key_env} must be set when reward_mode='judge'"
        )
    api_key = os.environ[api_key_env]
    if not model:
        raise ValueError("judge_model must be set when reward_mode='judge'")
    if max_retries < 1:
        raise ValueError("judge_max_retries must be >= 1")

    predictions = extract_evidence_strings(completion)
    targets = gold_evidence_strings(reward_spec)
    prompt = _judge_prompt(question, targets, predictions)
    base_url = base_url.rstrip("/")
    last_error: Exception | None = None

    for attempt in range(max_retries):
        if attempt:
            time.sleep(min(2 ** attempt, 30))
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "response_format": _RUBRIC_RESPONSE_FORMAT,
                    },
                )
                response.raise_for_status()
                payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            result = json.loads(content)
            precision = max(0.0, min(1.0, float(result["precision_score"]) / 10.0))
            recall = max(0.0, min(1.0, float(result["recall_score"]) / 10.0))
            score = (precision + recall) / 2.0
            return EvidenceReward(
                score=score,
                f1=score,
                copy_rate=0.0,
                count_penalty=0.0,
                num_predictions=len(predictions),
                judge_precision=precision,
                judge_recall=recall,
            )
        except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc

    raise RuntimeError("judge reward failed after retries") from last_error
