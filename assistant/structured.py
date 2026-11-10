"""Getting validated, typed data out of a language model.

Models wrap JSON in prose or ``` fences, leave trailing commas, or miss a field. This module
1. pulls the JSON out of the text (`extract_json`),
2. validates it with a pydantic model,
3. if it fails, sends the model its own output plus the exact validation errors and asks for a
   corrected version (up to `repairs` times) - the repair loop shown in the app.
"""
from __future__ import annotations

import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import BadModelOutput
from .llm import LLMClient, Params

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(text: str):
    """Return the parsed JSON value found in `text`, or raise ValueError."""
    if "<answer>" in text:                                   # chain-of-thought answers
        text = text.split("<answer>", 1)[1].split("</answer>", 1)[0]
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON object or array in the answer")
    start = min(starts)
    closer = "}" if text[start] == "{" else "]"
    end = text.rfind(closer)
    if end <= start:
        raise ValueError("the JSON is not closed (answer cut off?)")
    chunk = text[start:end + 1]
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", chunk)          # trailing commas
        cleaned = re.sub(r"//[^\n]*", "", cleaned)               # // comments
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e.msg} at line {e.lineno} column {e.colno}") from e


def ask_structured(client: LLMClient, *, task: str, technique: str, system: str, user: str,
                   schema: Type[T], params: Params, repairs: int = 2) -> T:
    """Call the model and return an instance of `schema`, repairing invalid output."""
    text = client.complete(task, technique, system, user, params)
    for attempt in range(repairs + 1):
        try:
            return schema.model_validate(extract_json(text))
        except (ValueError, ValidationError) as e:
            problem = _describe(e)
            if attempt == repairs:
                raise BadModelOutput(f"Step '{task}' returned invalid data after {repairs} repair attempts: {problem}") from e
            text = client.complete(
                f"{task}:repair", "self-correction (validation feedback)", system,
                REPAIR_PROMPT.format(output=text[:6000], problem=problem,
                                     schema=json.dumps(schema.model_json_schema())[:3000]),
                Params(temperature=0.0, max_tokens=params.max_tokens, top_p=1.0))
    raise AssertionError("unreachable")  # pragma: no cover


def _describe(e: Exception) -> str:
    if isinstance(e, ValidationError):
        parts = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:8]]
        return "; ".join(parts)
    return str(e)


REPAIR_PROMPT = """Your previous answer could not be used.

Previous answer:
{output}

Problem found by the validator:
{problem}

Required JSON schema:
{schema}

Return ONLY the corrected JSON, with no explanation and no code fences."""
