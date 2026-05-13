"""Dataset helpers for real multi-paper RLM training."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    abstract: str
    text: str

    def context(self, max_chars: int | None = None) -> str:
        if self.text.startswith("### PAPER:"):
            body = self.text
        else:
            abstract = f"<abstract>\n{self.abstract}\n</abstract>\n" if self.abstract else ""
            body = f"### PAPER: {self.title}\n{abstract}{self.text}"
        if max_chars is not None and len(body) > max_chars:
            return body[:max_chars] + "\n[TRUNCATED]"
        return body


@dataclass(frozen=True)
class RLMMultiPaperExample:
    question: str
    papers: list[Paper]
    reward_spec: dict[str, Any]

    def paper_by_id(self) -> dict[str, Paper]:
        return {paper.paper_id: paper for paper in self.papers}

    def full_context(self) -> str:
        return "\n\n".join(paper.context() for paper in self.papers)


def _loads_maybe_json(value: Any, default: Any, field_name: str) -> Any:
    if value is None:
        return default
    if type(value) is str:
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} is not valid JSON") from exc
    return value


def _as_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _as_list(value: Any, field_name: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{field_name} must be a list")
    return value


def _required_string(record: dict[str, Any], key: str, field_name: str) -> str:
    if key not in record or record[key] is None:
        raise ValueError(f"{field_name}.{key} is required")
    value = record[key]
    if type(value) is not str:
        raise ValueError(f"{field_name}.{key} must be a string")
    return value


def _optional_string(record: dict[str, Any], key: str, field_name: str) -> str:
    if key not in record or record[key] is None:
        return ""
    value = record[key]
    if type(value) is not str:
        raise ValueError(f"{field_name}.{key} must be a string")
    return value


def _first_present(record: dict[str, Any], keys: tuple[str, ...],
                   field_name: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    raise ValueError(f"{field_name} must contain one of {keys}")


def _title_from_text(text: str) -> str:
    first_line = text.splitlines()[0] if text.splitlines() else ""
    return first_line.replace("### PAPER: ", "")


def _abstract_from_text(text: str) -> str:
    match = re.search(r"<abstract>\s*(.*?)\s*</abstract>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _paper_from_record(value: Any, index: int) -> Paper:
    record = _as_mapping(value, f"papers[{index}]")
    raw_id = _first_present(record, ("paperId", "paper_id"), f"papers[{index}]")
    if type(raw_id) is not str:
        raise ValueError(f"papers[{index}].paperId must be a string")
    text = _required_string(record, "text", f"papers[{index}]")
    return Paper(
        paper_id=raw_id,
        title=_optional_string(record, "title", f"papers[{index}]"),
        abstract=_optional_string(record, "abstract", f"papers[{index}]"),
        text=text,
    )


def _papers_from_records(records: Any) -> list[Paper]:
    papers_raw = _as_list(records, "papers")
    return [
        _paper_from_record(paper, idx)
        for idx, paper in enumerate(papers_raw)
    ]


def _papers_from_context_payload(payload: Any) -> list[Paper]:
    if type(payload) is str:
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("context_text must be a JSON object when provided as a string") from exc
    context = _as_mapping(payload, "context_text")
    papers: list[Paper] = []
    for paper_id, text in context.items():
        if type(paper_id) is not str:
            raise ValueError("context_text keys must be paper ID strings")
        if type(text) is not str:
            raise ValueError(f"context_text[{paper_id!r}] must be a paper string")
        papers.append(Paper(
            paper_id=paper_id,
            title=_title_from_text(text),
            abstract=_abstract_from_text(text),
            text=text,
        ))
    return papers


def _question_from_prompt(prompt: Any) -> str:
    if type(prompt) is str:
        return prompt
    messages = _as_list(prompt, "prompt")
    user_messages = []
    for index, value in enumerate(messages):
        message = _as_mapping(value, f"prompt[{index}]")
        role = _required_string(message, "role", f"prompt[{index}]")
        if role == "user":
            user_messages.append(_required_string(message, "content", f"prompt[{index}]"))
    if not user_messages:
        raise ValueError("prompt must contain a user message")
    return user_messages[-1]


def _reward_spec_from_row(
        row: dict[str, Any],
        extra_info: dict[str, Any]) -> dict[str, Any]:
    if "reward_spec" in row and row["reward_spec"] is not None:
        reward_spec = _loads_maybe_json(row["reward_spec"], None, "reward_spec")
        return dict(_as_mapping(reward_spec, "reward_spec"))
    if "reward_spec" in extra_info and extra_info["reward_spec"] is not None:
        reward_spec = _loads_maybe_json(
            extra_info["reward_spec"], None, "extra_info.reward_spec")
        return dict(_as_mapping(reward_spec, "extra_info.reward_spec"))
    evidence = _loads_maybe_json(row["evidence"], [], "evidence") if "evidence" in row else []
    return {"ground_truth": row["ground_truth"] if "ground_truth" in row else None,
            "evidence": evidence}


def example_from_row(row: dict[str, Any]) -> RLMMultiPaperExample:
    row_map = _as_mapping(row, "row")
    extra_info = _loads_maybe_json(
        row_map["extra_info"] if "extra_info" in row_map else {},
        {},
        "extra_info",
    )
    extra_info = _as_mapping(extra_info, "extra_info")
    reward_spec = _reward_spec_from_row(row_map, extra_info)

    if "papers" in row_map and row_map["papers"] is not None:
        papers = _papers_from_records(
            _loads_maybe_json(row_map["papers"], [], "papers")
        )
    else:
        if "context_text" in row_map and row_map["context_text"] is not None:
            context_payload = row_map["context_text"]
        elif "context" in row_map and row_map["context"] is not None:
            context_payload = row_map["context"]
        elif "context_text" in extra_info and extra_info["context_text"] is not None:
            context_payload = extra_info["context_text"]
        else:
            raise ValueError("RLM row has no papers/context_text")
        papers = _papers_from_context_payload(context_payload)

    if "question" in row_map and row_map["question"] is not None:
        question = _required_string(row_map, "question", "row")
    elif "prompt" in row_map and row_map["prompt"] is not None:
        question = _question_from_prompt(row_map["prompt"])
    else:
        question = ""
    if not question.strip():
        raise ValueError("RLM row is missing question/prompt")
    if not papers:
        raise ValueError("RLM row has no papers")

    return RLMMultiPaperExample(
        question=str(question),
        papers=papers,
        reward_spec=reward_spec,
    )
