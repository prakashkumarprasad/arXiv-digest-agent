"""Rate-limit (HTTP 429) handling in the LLM client: wait, then re-send the same request."""

from types import SimpleNamespace

import httpx
import pytest
from openai import RateLimitError

from agent.services import llm as llm_module
from agent.services.llm import LLMClient, MAX_RATE_LIMIT_RETRIES, _parse_wait

GROQ_429 = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` "
    "on tokens per minute (TPM): Limit 8000, Used 4155, Requested 4284. "
    "Please try again in 3.2925s. Need more tokens?'}}"
)


def _rate_limit_error(message=GROQ_429, retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "http://test"))
    return RateLimitError(message, response=response, body=None)


class _FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))])


def _client_with(outcomes, monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: sleeps.append(s))
    completions = _FakeCompletions(outcomes)
    client = LLMClient("ollama")
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions, sleeps


def test_parse_wait_understands_provider_formats():
    assert _parse_wait("Please try again in 3.2925s. Need more") == pytest.approx(3.2925)
    assert _parse_wait("try again in 2m30.5s") == pytest.approx(150.5)
    assert _parse_wait("try again in 250ms") == pytest.approx(0.25)
    assert _parse_wait("no hint here") is None


def test_429_waits_then_resends_the_same_request(monkeypatch):
    client, completions, sleeps = _client_with([_rate_limit_error(), "ok"], monkeypatch)

    assert client.complete_text("sys", "user question") == "ok"

    assert sleeps == [pytest.approx(3.2925 + 0.5)]
    assert len(completions.calls) == 2
    assert completions.calls[0]["messages"] == completions.calls[1]["messages"]
    assert "Rate limit" not in str(completions.calls[1]["messages"])  # error text is not fed back


def test_retry_after_header_wins_over_message_text(monkeypatch):
    client, _, sleeps = _client_with([_rate_limit_error(retry_after=2), "ok"], monkeypatch)
    client.complete_text("sys", "user")
    assert sleeps == [pytest.approx(2.5)]


def test_gives_up_after_max_retries(monkeypatch):
    errors = [_rate_limit_error()] * (MAX_RATE_LIMIT_RETRIES + 1)
    client, completions, sleeps = _client_with(errors, monkeypatch)
    with pytest.raises(RateLimitError):
        client.complete_text("sys", "user")
    assert len(completions.calls) == MAX_RATE_LIMIT_RETRIES + 1
    assert len(sleeps) == MAX_RATE_LIMIT_RETRIES


def test_long_quota_reset_is_not_waited_out(monkeypatch):
    error = _rate_limit_error("Rate limit reached. Please try again in 12m30s.")
    client, completions, sleeps = _client_with([error], monkeypatch)
    with pytest.raises(RateLimitError):
        client.complete_text("sys", "user")
    assert sleeps == []
    assert len(completions.calls) == 1


def test_other_errors_are_not_retried(monkeypatch):
    client, completions, sleeps = _client_with([ValueError("boom")], monkeypatch)
    with pytest.raises(ValueError):
        client.complete_text("sys", "user")
    assert sleeps == []
    assert len(completions.calls) == 1