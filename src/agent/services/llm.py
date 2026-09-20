"""LLM provider abstraction with JSON repair loop."""

import json
import logging
import os
import re
import time
from typing import Any, Type

from pydantic import BaseModel, ValidationError

from agent.config import get_settings

logger = logging.getLogger(__name__)

# Rate-limit handling: extra attempts after a 429, and the longest wait we will sit
# through. A longer wait means a quota reset (e.g. a daily limit), so we fail fast.
MAX_RATE_LIMIT_RETRIES = 4
MAX_RATE_LIMIT_WAIT_S = 60.0

_WAIT_RE = re.compile(r"try again in\s+((?:\d+(?:\.\d+)?(?:ms|m|s|h))+)", re.IGNORECASE)
_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)", re.IGNORECASE)
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def _parse_wait(text: str) -> float | None:
    """Parse 'try again in 3.29s' / '2m30.5s' / '250ms' from a provider error message."""
    match = _WAIT_RE.search(text)
    if not match:
        return None
    return sum(float(n) * _UNIT_SECONDS[u.lower()] for n, u in _UNIT_RE.findall(match.group(1)))


def _retry_after_seconds(exc: Exception) -> float | None:
    """Seconds the provider asks us to wait: the Retry-After header first, else the message text."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        value = headers.get("retry-after")
        if value:
            try:
                return float(value)
            except ValueError:
                pass
    return _parse_wait(str(exc))



# Provider configurations
PROVIDER_CONFIGS = {
    "groq": {
        "model": "openai/gpt-oss-120b",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
    },
    "gemini": {
        "model": "gemini-1.5-flash",
        "base_url": None,  # Uses Google AI SDK
        "env_key": "GEMINI_API_KEY",
    },
    "ollama": {
        "model": "qwen2.5:7b-instruct",
        "base_url": "http://localhost:11434/v1",
        "env_key": None,
    },
}


class LLMClient:
    """Abstract LLM client supporting multiple providers."""

    def __init__(self, provider: str | None = None):
        self.settings = get_settings()
        self.provider = provider or self.settings.llm_provider
        self.config = PROVIDER_CONFIGS.get(self.provider, PROVIDER_CONFIGS["ollama"])
        self._client = None

    def _get_client(self):
        """Lazy initialization of the provider client."""
        if self._client is not None:
            return self._client

        if self.provider == "groq":
            from openai import OpenAI
            api_key = os.getenv(self.config["env_key"])
            if not api_key:
                raise ValueError(f"Groq API key not found. Set {self.config['env_key']} in environment.")
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.config["base_url"],
            )
        elif self.provider == "gemini":
            import google.generativeai as genai
            api_key = os.getenv(self.config["env_key"])
            if not api_key:
                raise ValueError(f"Gemini API key not found. Set {self.config['env_key']} in environment.")
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.config["model"])
        elif self.provider == "ollama":
            from openai import OpenAI
            self._client = OpenAI(
                api_key="ollama",  # Dummy key for Ollama
                base_url=self.config["base_url"],
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._client

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        """Get a plain text completion."""
        client = self._get_client()

        if self.provider == "gemini":
            # Gemini uses a different API
            response = client.generate_content(
                f"{system_prompt}\n\n{user_prompt}",
                generation_config={"temperature": 0.1, "max_output_tokens": 4000},
            )
            return response.text
        else:
            # OpenAI-compatible (Groq, Ollama)
            response = self._create_with_backoff(
                model=self.config["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=4000,
            )
            return response.choices[0].message.content

    def _create_with_backoff(self, **kwargs):
        """Call chat.completions.create, waiting out provider rate limits (HTTP 429).

        The original request is re-sent unchanged. A wait longer than
        MAX_RATE_LIMIT_WAIT_S, or running out of attempts, re-raises the original error.
        """
        from openai import RateLimitError

        client = self._get_client()
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return client.chat.completions.create(**kwargs)
            except RateLimitError as e:
                wait = _retry_after_seconds(e)
                if attempt >= MAX_RATE_LIMIT_RETRIES or (wait is not None and wait > MAX_RATE_LIMIT_WAIT_S):
                    raise
                delay = (wait if wait is not None else 2.0 * (attempt + 1)) + 0.5
                logger.warning(
                    f"Rate limited by {self.provider}; waiting {delay:.1f}s "
                    f"(retry {attempt + 1}/{MAX_RATE_LIMIT_RETRIES})"
                )
                time.sleep(delay)

    def complete_json(self, schema: Type[BaseModel], system_prompt: str, user_prompt: str) -> BaseModel:
        """Get a JSON completion validated against a Pydantic schema with one retry."""
        # First attempt
        text = self.complete_text(system_prompt, user_prompt)

        try:
            data = self._extract_json(text)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"First JSON attempt failed: {e}. Retrying with error feedback.")
            # Retry with error feedback
            retry_prompt = (
                f"{user_prompt}\n\n"
                f"Your previous response failed validation:\n{e}\n\n"
                f"Please return ONLY valid JSON matching the schema. No extra text."
            )
            text = self.complete_text(system_prompt, retry_prompt)

            try:
                data = self._extract_json(text)
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as e2:
                logger.error(f"Second JSON attempt also failed: {e2}")
                raise

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from response, handling markdown code fences."""
        # Try to find JSON in markdown code fences
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        # Try to find JSON object boundaries
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)
        elif text.startswith("[") and text.endswith("]"):
            return json.loads(text)
        else:
            # Try to extract first {...} or [...] block
            for start_char, end_char in [("{", "}"), ("[", "]")]:
                start = text.find(start_char)
                if start >= 0:
                    # Find matching closing bracket
                    depth = 0
                    for i, ch in enumerate(text[start:], start):
                        if ch == start_char:
                            depth += 1
                        elif ch == end_char:
                            depth -= 1
                            if depth == 0:
                                return json.loads(text[start:i+1])
            raise json.JSONDecodeError("No valid JSON found", text, 0)


def complete_json(
    schema: Type[BaseModel],
    system_prompt: str,
    user_prompt: str,
    provider: str | None = None,
) -> BaseModel:
    """Calls the LLM, validates against `schema`, retries once with the ValidationError text on failure.

    Raises only after the retry also fails.
    """
    client = LLMClient(provider)
    return client.complete_json(schema, system_prompt, user_prompt)


def complete_text(
    system_prompt: str,
    user_prompt: str,
    provider: str | None = None,
) -> str:
    """Get a plain text completion from the LLM."""
    client = LLMClient(provider)
    return client.complete_text(system_prompt, user_prompt)


def get_llm_client(provider: str | None = None) -> LLMClient:
    """Get an LLM client instance for the specified provider."""
    return LLMClient(provider)