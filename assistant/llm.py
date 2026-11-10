"""One small client in front of every provider.

Real providers go through LangChain chat models (Groq, OpenAI, Ollama, as in the course);
`demo` replays recorded answers for the sample files, so the app and the tests run without
a key. Every call names its `task` (classify_pages, extract_registers, ...) and its
parameters, so the app can show exactly which prompt and which settings produced a result.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .errors import (AuthError, ConfigError, DemoModeError, LLMTimeout, ProviderUnavailable,
                     RateLimitError, TruncatedOutput)

PROVIDERS = {
    # name: (default model, environment variable for the key)
    "groq": ("llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "openai": ("gpt-4o-mini", "OPENAI_API_KEY"),
    "ollama": ("llama3.2", None),
    "demo": ("recorded-answers", None),
}


@dataclass
class Params:
    """Generation parameters for one call (the 'parameter tuning' part of the project)."""
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError(f"temperature must be between 0 and 2, got {self.temperature}.")
        if not 16 <= self.max_tokens <= 32768:
            raise ConfigError(f"max_tokens must be between 16 and 32768, got {self.max_tokens}.")
        if not 0.0 < self.top_p <= 1.0:
            raise ConfigError(f"top_p must be in (0, 1], got {self.top_p}.")


@dataclass
class CallRecord:
    """What the app shows in 'Prompts and settings used'."""
    task: str
    technique: str
    params: Params
    system: str
    user: str
    output: str
    seconds: float
    attempts: int


@dataclass
class LLMClient:
    provider: str = "demo"
    model: str = ""
    api_key: str = ""
    timeout_s: float = 60.0
    max_attempts: int = 4            # 1 call + 3 retries on rate limits / transient errors
    backoff_s: float = 2.0           # 2 s, 4 s, 8 s
    demo_answers: Optional[dict] = None
    log: list = field(default_factory=list)
    _sleep: Callable[[float], None] = time.sleep
    _model_factory: Optional[Callable] = None   # tests inject a fake chat model here
    _demo_counters: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ConfigError(f"Unknown provider '{self.provider}'. Choose one of: {', '.join(PROVIDERS)}.")
        self.model = self.model or PROVIDERS[self.provider][0]
        env = PROVIDERS[self.provider][1]
        if env and not self.api_key:
            self.api_key = os.getenv(env, "")
        if env and not self.api_key and self._model_factory is None:
            raise ConfigError(f"No API key for {self.provider}: paste it in the sidebar or set {env} in .env.")

    # ------------------------------------------------------------------ public
    def complete(self, task: str, technique: str, system: str, user: str, params: Params) -> str:
        start = time.monotonic()
        if self.provider == "demo":
            text, attempts = self._demo(task, user), 1
        else:
            text, attempts = self._call_with_retries(system, user, params)
        self.log.append(CallRecord(task, technique, params, system, user, text,
                                   round(time.monotonic() - start, 2), attempts))
        return text

    # ------------------------------------------------------------------ real providers
    def _chat_model(self, params: Params):
        if self._model_factory is not None:
            return self._model_factory(params)
        try:
            if self.provider == "groq":
                from langchain_groq import ChatGroq
                return ChatGroq(model=self.model, api_key=self.api_key, temperature=params.temperature,
                                max_tokens=params.max_tokens, request_timeout=self.timeout_s, max_retries=0,
                                model_kwargs={"top_p": params.top_p})
            if self.provider == "openai":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model=self.model, api_key=self.api_key, temperature=params.temperature,
                                  max_tokens=params.max_tokens, top_p=params.top_p,
                                  request_timeout=self.timeout_s, max_retries=0)
            if self.provider == "ollama":
                from langchain_ollama import ChatOllama
                return ChatOllama(model=self.model, temperature=params.temperature,
                                  num_predict=params.max_tokens, top_p=params.top_p,
                                  base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        except ImportError as e:
            raise ConfigError(f"The package for {self.provider} is not installed ({e.name}). Run: pip install -r requirements.txt") from e
        raise ConfigError(f"Provider {self.provider} has no chat model.")

    def _call_with_retries(self, system: str, user: str, params: Params):
        from langchain_core.messages import HumanMessage, SystemMessage
        chat = self._chat_model(params)
        last = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                msg = chat.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            except Exception as e:                           # noqa: BLE001 - mapped below
                err = classify_exception(e)
                retryable = isinstance(err, (RateLimitError, LLMTimeout, ProviderUnavailable))
                if not retryable or attempt == self.max_attempts:
                    raise err from e
                last = err
                self._sleep(self.backoff_s * 2 ** (attempt - 1))
                continue
            meta = getattr(msg, "response_metadata", {}) or {}
            reason = meta.get("finish_reason") or meta.get("done_reason") or ""
            if reason == "length":
                raise TruncatedOutput(f"The model stopped at max_tokens={params.max_tokens} before finishing.")
            return _text(msg), attempt
        raise last  # pragma: no cover - loop always returns or raises

    # ------------------------------------------------------------------ demo
    def _demo(self, task: str, user: str = "") -> str:
        if self.demo_answers is None:
            raise DemoModeError("Demo mode has no recorded answers loaded for this file.")
        answers = self.demo_answers.get(task)
        if not answers:
            raise DemoModeError(f"Demo mode has no recorded answer for step '{task}'.")
        if isinstance(answers, dict):                 # answers keyed by question text
            for key, text in answers.items():
                if key.lower() in user.lower():
                    return text
            raise DemoModeError("Demo mode can only answer the suggested sample questions.")
        i = self._demo_counters.get(task, 0)
        self._demo_counters[task] = i + 1
        return answers[min(i, len(answers) - 1)]


def _text(msg) -> str:
    content = msg.content
    if isinstance(content, list):                    # some providers return content blocks
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def classify_exception(e: Exception):
    """Map SDK exceptions (openai, groq, httpx, ollama) to the app's errors by name,
    so the app does not need to import every SDK."""
    names = " ".join(c.__name__ for c in type(e).__mro__)
    text = str(e)
    if "Authentication" in names or "PermissionDenied" in names or "401" in text:
        return AuthError("The provider rejected the API key.")
    if "RateLimit" in names or "429" in text:
        return RateLimitError("The provider's rate limit was reached.")
    if "Timeout" in names:
        return LLMTimeout("The request timed out.")
    if "Connect" in names or "ConnectionError" in names:
        return ProviderUnavailable("Could not connect to the provider.")
    if "NotFound" in names or "model_not_found" in text:
        return ConfigError("The provider does not know this model name.")
    if "BadRequest" in names and ("context" in text.lower() or "too long" in text.lower()):
        return ConfigError("The input is too long for this model's context window.")
    return ProviderUnavailable(f"The provider returned an error: {type(e).__name__}: {text[:200]}")


def load_demo_answers(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
