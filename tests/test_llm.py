import pytest

from assistant.errors import (AuthError, ConfigError, LLMTimeout, ProviderUnavailable, RateLimitError,
                              TruncatedOutput)
from assistant.llm import LLMClient, Params, classify_exception


class AuthenticationError(Exception): ...
class RateLimitErrorSDK(Exception): ...
class APITimeoutError(Exception): ...
class APIConnectionError(Exception): ...
RateLimitErrorSDK.__name__ = "RateLimitError"


class Msg:
    def __init__(self, content, reason="stop"):
        self.content, self.response_metadata = content, {"finish_reason": reason}


class FakeChat:
    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def invoke(self, messages):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(script, **kw):
    chat = FakeChat(script)
    sleeps = []
    c = LLMClient("groq", api_key="test", _model_factory=lambda p: chat, _sleep=sleeps.append, **kw)
    return c, chat, sleeps


def test_params_are_checked():
    with pytest.raises(ConfigError):
        Params(temperature=3)
    with pytest.raises(ConfigError):
        Params(max_tokens=5)
    with pytest.raises(ConfigError):
        Params(top_p=0)


def test_unknown_provider_and_missing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        LLMClient("nope")
    with pytest.raises(ConfigError):
        LLMClient("groq")


@pytest.mark.parametrize("exc,cls", [(AuthenticationError("401"), AuthError), (RateLimitErrorSDK("slow down"), RateLimitError),
                                     (APITimeoutError("t"), LLMTimeout), (APIConnectionError("c"), ProviderUnavailable)])
def test_exception_mapping(exc, cls):
    assert isinstance(classify_exception(exc), cls)


def test_rate_limit_is_retried_with_backoff():
    c, chat, sleeps = client([RateLimitErrorSDK("429"), RateLimitErrorSDK("429"), Msg("ok")])
    assert c.complete("t", "zero-shot", "s", "u", Params()) == "ok"
    assert sleeps == [2.0, 4.0] and chat.calls == 3 and c.log[0].attempts == 3


def test_rate_limit_gives_up():
    c, chat, sleeps = client([RateLimitErrorSDK("429")] * 4)
    with pytest.raises(RateLimitError):
        c.complete("t", "x", "s", "u", Params())
    assert chat.calls == 4


def test_auth_error_is_not_retried():
    c, chat, _ = client([AuthenticationError("bad key")])
    with pytest.raises(AuthError):
        c.complete("t", "x", "s", "u", Params())
    assert chat.calls == 1


def test_truncated_answer_is_reported():
    c, _, _ = client([Msg('{"a": ', reason="length")])
    with pytest.raises(TruncatedOutput):
        c.complete("t", "x", "s", "u", Params(max_tokens=32))


def test_log_records_technique_and_parameters():
    c, _, _ = client([Msg("hi")])
    c.complete("brief", "persona", "sys", "usr", Params(temperature=0.7, max_tokens=500))
    rec = c.log[0]
    assert (rec.task, rec.technique, rec.params.temperature, rec.params.max_tokens) == ("brief", "persona", 0.7, 500)


@pytest.mark.parametrize("provider", ["groq", "openai", "ollama"])
def test_real_providers_build_their_langchain_model(provider):
    """No network: only checks that the parameters reach the LangChain chat model."""
    c = LLMClient(provider, api_key="test-key")
    chat = c._chat_model(Params(temperature=0.3, max_tokens=123, top_p=0.9))
    assert chat.temperature == 0.3
    assert getattr(chat, "max_tokens", None) == 123 or getattr(chat, "num_predict", None) == 123
