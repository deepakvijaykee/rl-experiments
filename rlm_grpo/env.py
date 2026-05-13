"""Local RLM REPL environment used by the standalone trainer."""

from __future__ import annotations

import io
import re
import signal
import threading
import textwrap
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any, Callable, Protocol


_REPL_BLOCK_RE = re.compile(r"```repl\s*\n(.*?)\n```", re.DOTALL)
_ABSTRACT_RE = re.compile(r"<abstract>\s*(.*?)\s*</abstract>", re.DOTALL | re.IGNORECASE)
_FINAL_RE = re.compile(r"^\s*FINAL\((.*)\)\s*$", re.MULTILINE | re.DOTALL)


ROOT_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an evidence extraction coordinator. You need verbatim text passages
    from a multi-paper context that answer the user's query.

    You interact only by writing one Python ```repl code block per response.
    The environment executes the first block and returns observations.

    Available variables and tools:
    - context: dict mapping paper IDs to paper text.
    - SHOW_VARS(): show variables currently stored in the REPL.
    - list_papers(context): list paper IDs with previews.
    - search(text, keyword, window=300): search one paper string or all papers.
    - get_paper_abstract(context, paper_id): get title and abstract.
    - rlm_query(prompt, context=None): dispatch one child agent.
    - rlm_query_batched(prompts, context_list): dispatch child agents. Each
      child receives one prompt and one paper string. The return value is a
      list of child final answers.
    - FINAL(answer): finish with a direct answer.
    - FINAL_VAR(variable_name): finish by returning the named variable.

    Strategy:
    1. Inspect the context with list_papers and search.
    2. Select relevant papers and dispatch child agents with rlm_query_batched.
    3. Flatten child evidence into a final list and return it with FINAL_VAR.

    For multi-paper questions, use child agents rather than extracting
    everything yourself. Final evidence must be a list of exact substrings
    copied from the provided paper texts.

    Use at most four papers in one rlm_query_batched call. If more papers look
    relevant, split them across turns and merge all child results before
    calling FINAL_VAR.
    """
)


CHILD_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a precise evidence extraction worker. You have one paper in the
    variable context and must find verbatim passages that answer the query.

    You interact only by writing one Python ```repl code block per response.
    The environment executes the first block and returns observations.

    Available variables and tools:
    - context: the assigned paper text.
    - SHOW_VARS(): show variables currently stored in the REPL.
    - search(text, keyword, window=300): search the paper.
    - extract_section(snippet, start_phrase, end_phrase): extract a tighter
      exact substring from a search snippet.
    - FINAL(answer): finish with a direct answer.
    - FINAL_VAR(variable_name): finish by returning the named variable.

    Strategy:
    1. Search with multiple keyword sets.
    2. Expand promising snippets with larger windows when needed.
    3. Extract tight spans with extract_section.
    4. Return a list of exact substrings copied from context, or an empty list
       if the paper has no relevant evidence.
    """
)


class FinalAnswer(Exception):
    def __init__(self, value: Any):
        self.value = value


class GeneratedTurn(Protocol):
    text: str


@dataclass
class EnvRunResult:
    final_answer: str
    segments: list[GeneratedTurn]
    child_count: int
    stopped_by_turn_limit: bool


SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "repr": repr,
    "range": range,
    "enumerate": enumerate,
    "map": map,
    "filter": filter,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "sorted": sorted,
    "min": min,
    "max": max,
    "sum": sum,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "type": type,
    "any": any,
    "all": all,
    "abs": abs,
    "round": round,
    "pow": pow,
    "divmod": divmod,
    "zip": zip,
    "reversed": reversed,
    "slice": slice,
    "callable": callable,
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "AssertionError": AssertionError,
    "StopIteration": StopIteration,
    "__import__": None,
    "open": None,
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}

REPL_OBSERVED_ERRORS = (
    AssertionError,
    ArithmeticError,
    AttributeError,
    IndexError,
    KeyError,
    LookupError,
    NameError,
    RuntimeError,
    SyntaxError,
    TypeError,
    ValueError,
    ZeroDivisionError,
)

RESERVED_TOOL_NAMES = {
    "FINAL",
    "FINAL_VAR",
    "SHOW_VARS",
    "context",
    "rlm_query",
    "rlm_query_batched",
    "list_papers",
    "search",
    "extract_section",
    "get_paper_abstract",
}


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[TRUNCATED]"


def list_papers(context: dict[str, str]) -> list[str]:
    titles = []
    print(f"Found {len(context)} papers:")
    for paper_id, text in context.items():
        lines = text.splitlines()
        title = lines[0].replace("### PAPER: ", "") if lines else "Unknown Title"
        match = _ABSTRACT_RE.search(text)
        abstract = match.group(1).strip() if match else ""
        print(f"\nPaper ID: {paper_id}")
        print(f"Title: {title}")
        if abstract:
            print(f"Abstract: {abstract}")
        print("-" * 80)
        titles.append(title)
    return titles


def _snippets(
        text: str,
        keyword: str,
        window: int,
        bidirectional: bool = True) -> list[str]:
    snippets = []
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    for match in pattern.finditer(text):
        if bidirectional:
            left = max(0, match.start() - window // 2)
            right = min(len(text), match.end() + window // 2)
        else:
            left = match.start()
            right = min(len(text), match.start() + window)

        while left > 0 and text[left - 1] not in ".!?\n":
            left -= 1
            if match.start() - left > window:
                break
        while right < len(text) and text[right] not in ".!?\n":
            right += 1
            if right - match.end() > window:
                break
        if right < len(text) and text[right] in ".!?\n":
            right += 1

        snippet = text[left:right].strip()
        print(f"--- snippet {len(snippets)} ---")
        print(snippet)
        snippets.append(snippet)
    return snippets


def search(text: str | dict[str, str], keyword: str,
           window: int = 300, bidirectional: bool = True) -> list[str]:
    if type(text) is dict:
        out = []
        for paper_id, paper_text in text.items():
            title = paper_text.splitlines()[0].replace("### PAPER: ", "")
            paper_results = _snippets(paper_text, keyword, window, bidirectional)
            if paper_results:
                print(f"\n=== Paper: {paper_id} - {title} ===")
            for snippet in paper_results:
                out.append(f"PAPER_ID={paper_id}\n{snippet}")
        if not out:
            print(f"(no hits for {keyword!r} in any paper)")
        return out
    if type(text) is not str:
        raise ValueError("search expects a paper string or context dict")
    results = _snippets(text, keyword, window, bidirectional)
    if not results:
        print(f"(no hits for {keyword!r})")
    return results


def extract_section(snippet: str, start_phrase: str, end_phrase: str) -> str:
    lowered = snippet.lower()
    start = lowered.find(start_phrase.lower())
    if start < 0:
        start = 0
    end = lowered.find(end_phrase.lower(), start)
    if end < 0:
        result = snippet[start:]
    else:
        result = snippet[start:end + len(end_phrase)]
    print(result)
    return result


def get_paper_abstract(context: dict[str, str], paper_id: str) -> str:
    if paper_id not in context:
        raise KeyError(f"unknown paper_id: {paper_id}")
    text = context[paper_id]
    first_line = text.splitlines()[0].replace("### PAPER: ", "") if text.splitlines() else ""
    match = _ABSTRACT_RE.search(text)
    abstract = match.group(1).strip() if match else ""
    return f"Paper ID: {paper_id}\nTitle: {first_line}\nAbstract: {abstract}"


def extract_repl_code(text: str) -> str | None:
    matches = _REPL_BLOCK_RE.findall(text)
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError("multiple ```repl blocks found; submit exactly one")
    return matches[0].strip()


def format_value(value: Any, max_chars: int) -> str:
    text = repr(value)
    return truncate_text(text, max_chars)


class RLMReplEnv:
    """Multi-turn Python-REPL environment for one root or child rollout."""

    def __init__(
            self,
            system_prompt: str,
            user_prompt: str,
            initial_state: dict[str, Any],
            generate_fn: Callable[[list[dict[str, str]], int], GeneratedTurn],
            max_new_tokens: int,
            max_turns: int,
            max_observation_chars: int = 6000,
            repl_timeout: int = 30,
            rlm_query_fn: Callable[..., Any] | None = None,
            rlm_query_batched_fn: Callable[..., Any] | None = None):
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._context_metadata(initial_state["context"])},
        ]
        self.root_prompt = user_prompt
        self.state = dict(initial_state)
        self.generate_fn = generate_fn
        self.max_new_tokens = max_new_tokens
        self.max_turns = max_turns
        self.max_observation_chars = max_observation_chars
        self.repl_timeout = repl_timeout
        self.rlm_query_fn = rlm_query_fn
        self.rlm_query_batched_fn = rlm_query_batched_fn
        self.segments: list[GeneratedTurn] = []
        self.child_count = 0
        self._exec_state: dict[str, Any] | None = None
        self._install_tools()

    def _install_tools(self):
        self.state.update({
            "list_papers": list_papers,
            "search": search,
            "extract_section": extract_section,
            "get_paper_abstract": get_paper_abstract,
            "FINAL": self._final,
            "FINAL_VAR": self._final_var,
            "SHOW_VARS": self._show_vars,
        })
        if self.rlm_query_fn is not None:
            self.state["rlm_query"] = self.rlm_query_fn
        if self.rlm_query_batched_fn is not None:
            self.state["rlm_query_batched"] = self.rlm_query_batched_fn

    def _final(self, value: Any):
        raise FinalAnswer(value)

    def _final_var(self, variable_name: Any):
        if type(variable_name) is not str:
            raise FinalAnswer(variable_name)
        variable_name = variable_name.strip().strip("\"'")
        lookup = self._exec_state if self._exec_state is not None else self.state
        if variable_name in lookup:
            raise FinalAnswer(lookup[variable_name])
        visible = [
            key for key in lookup
            if not key.startswith("_") and key not in RESERVED_TOOL_NAMES
        ]
        if visible:
            print(
                f"Error: Variable {variable_name!r} not found. "
                f"Available variables: {visible}. "
                "Create the variable before calling FINAL_VAR."
            )
        else:
            print(
                f"Error: Variable {variable_name!r} not found. "
                "No variables have been created yet."
            )
        return None

    def _show_vars(self) -> str:
        lookup = self._exec_state if self._exec_state is not None else self.state
        visible = {
            key: type(value).__name__
            for key, value in lookup.items()
            if not key.startswith("_") and key not in RESERVED_TOOL_NAMES
        }
        if not visible:
            return "No variables created yet."
        return f"Available variables: {visible}"

    def run(self) -> EnvRunResult:
        for turn_index in range(self.max_turns):
            prompt = self._turn_prompt(turn_index)
            segment = self.generate_fn(self.messages + [prompt], self.max_new_tokens)
            self.segments.append(segment)
            self.messages.append({"role": "assistant", "content": segment.text})
            observation, final = self._execute_response(segment.text)
            if final is not None:
                return EnvRunResult(
                    final_answer=self._stringify_final(final),
                    segments=self.segments,
                    child_count=self.child_count,
                    stopped_by_turn_limit=False,
                )
            self.messages.append({"role": "user", "content": observation})

        return EnvRunResult(
            final_answer="",
            segments=self.segments,
            child_count=self.child_count,
            stopped_by_turn_limit=True,
        )

    def _execute_response(self, response: str) -> tuple[str, Any | None]:
        final = self._find_final_answer(response)
        if final is not None:
            return "", final
        try:
            code = extract_repl_code(response)
        except ValueError as exc:
            return str(exc), None
        if code is None:
            return "No executable ```repl block found. Submit exactly one repl block.", None

        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            self._execute_code(code, stdout, stderr)
        except FinalAnswer as final:
            return "", final.value
        finally:
            self._restore_tools()

        output = stdout.getvalue()
        err_output = stderr.getvalue()
        var_names = self._visible_var_names()
        parts = [f"Code executed:\n```python\n{code}\n```"]
        if output:
            parts.append(f"REPL stdout:\n{output}")
        if err_output:
            parts.append(f"REPL stderr:\n{err_output}")
        if var_names:
            parts.append(f"REPL variables: {var_names}")
        if len(parts) == 1:
            parts.append("REPL output:\nNo output")
        return truncate_text("\n\n".join(parts), self.max_observation_chars), None

    def _execute_code(self, code: str, stdout: io.StringIO, stderr: io.StringIO):
        if self._can_use_sigalrm():
            def _timeout(*_):
                raise TimeoutError("Code execution timed out")

            old_alarm = signal.signal(signal.SIGALRM, _timeout)
            try:
                signal.alarm(self.repl_timeout)
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    self._execute_once(code)
            except FinalAnswer:
                raise
            except TimeoutError:
                print(f"Timeout after {self.repl_timeout} seconds", file=stderr)
            except REPL_OBSERVED_ERRORS:
                print(traceback.format_exc(), file=stderr)
            finally:
                self._exec_state = None
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_alarm)
            return

        result: dict[str, Any] = {"error": None, "final": None}

        def _run():
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    self._execute_once(code)
            except FinalAnswer as final:
                result["final"] = final
            except REPL_OBSERVED_ERRORS:
                result["error"] = traceback.format_exc()
            finally:
                self._exec_state = None

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.repl_timeout)
        if thread.is_alive():
            print(f"Timeout after {self.repl_timeout} seconds", file=stderr)
        elif result["final"] is not None:
            raise result["final"]
        elif result["error"]:
            print(result["error"], file=stderr)

    def _execute_once(self, code: str):
        combined = {
            "__builtins__": SAFE_BUILTINS.copy(),
            "__name__": "__main__",
            **self.state,
        }
        self._exec_state = combined
        try:
            exec(code, combined, combined)
        except FinalAnswer:
            self._save_exec_state(combined)
            raise
        else:
            self._save_exec_state(combined)

    def _save_exec_state(self, combined: dict[str, Any]):
        for key, value in combined.items():
            if key in {"__builtins__", "__name__"} or key.startswith("__"):
                continue
            self.state[key] = value

    @staticmethod
    def _can_use_sigalrm() -> bool:
        return (
            _signal_alarm_name() is not None
            and threading.current_thread() is threading.main_thread()
        )

    def _restore_tools(self):
        self._install_tools()

    def _find_final_answer(self, text: str) -> Any | None:
        matches = _FINAL_RE.findall(text)
        if matches:
            return matches[-1].strip()
        return None

    def _visible_var_names(self) -> list[str]:
        return [
            key for key in self.state
            if not key.startswith("_") and key not in RESERVED_TOOL_NAMES
        ]

    @staticmethod
    def _stringify_final(value: Any) -> str:
        if type(value) is str:
            return value
        return repr(value)

    def _turn_prompt(self, turn_index: int) -> dict[str, str]:
        if turn_index == 0:
            prefix = (
                "You have not interacted with the REPL yet. Inspect the context "
                "before finalizing.\n\n"
            )
        else:
            prefix = "Continue from the REPL history above.\n\n"
        return {
            "role": "user",
            "content": (
                prefix
                + "Use exactly one ```repl block for your next action. "
                + f"Original prompt:\n{self.root_prompt}"
            ),
        }

    @staticmethod
    def _context_metadata(context: Any) -> str:
        if type(context) is dict:
            for value in context.values():
                if type(value) is not str:
                    raise ValueError("RLM context dict values must be paper strings")
            lengths = [
                len(value)
                for value in context.values()
            ]
            return (
                f"Your context is a dict with {sum(lengths)} total characters, "
                f"broken into chunks of char lengths: {lengths}."
            )
        if type(context) is str:
            return f"Your context is a str with {len(context)} characters."
        raise ValueError("RLM context must be a paper string or context dict")


def _signal_alarm_name() -> int | None:
    try:
        return signal.SIGALRM
    except AttributeError:
        return None
