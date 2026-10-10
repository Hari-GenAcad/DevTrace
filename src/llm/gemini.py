"""
Production Gemini LLM client.

Wraps google-generativeai behind the LLMClient interface.
Real API calls are never made during the test suite — tests use FakeLLMClient.
"""

from __future__ import annotations

import re
from typing import Any

from src.config import settings
from src.errors import ConfigurationError, LLMError, SchemaValidationError
from src.llm.base import LLMClient, LLMResponse


class GeminiResponse(LLMResponse):
    """Wraps a Gemini SDK response object."""

    def __init__(self, raw_response: Any) -> None:
        self._raw = raw_response

    @property
    def text(self) -> str:
        try:
            return self._raw.text  # type: ignore[attr-defined]
        except AttributeError:
            return ""

    @property
    def success(self) -> bool:
        return bool(self.text)

    @property
    def raw(self) -> Any:
        return self._raw


class GeminiClient(LLMClient):
    """
    Production LLM client backed by the Gemini API.

    Construction raises ConfigurationError immediately if GEMINI_API_KEY is
    absent, so misconfigured deployments fail fast rather than silently.

    Temperature and model are read from settings but can be overridden per-call
    via kwargs (useful for future structured-output prompts in M4/M5).
    """

    supports_batch_verification = True

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
    ) -> None:
        if not settings.gemini_api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example to .env and provide your API key."
            )

        # Lazy import so that tests that never construct GeminiClient do not
        # pay the import cost or require the SDK to be importable.
        try:
            import google.generativeai as genai  # noqa: PLC0415
        except ImportError as exc:
            raise ConfigurationError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=settings.gemini_api_key)
        self._genai = genai
        self._model_name = model_name or settings.gemini_model
        self._default_temperature = (
            settings.gemini_temperature if temperature is None else temperature
        )

    def generate(self, prompt: str, **kwargs: Any) -> GeminiResponse:
        """
        Send a prompt to Gemini and return a GeminiResponse.

        Args:
            prompt: The prompt string.
            **kwargs: Optional overrides:
                - temperature (float)
                - generation_config (dict)

        Raises:
            LLMError: On API failure (network, quota, 5xx).
            SchemaValidationError: If a structured schema is requested and
                                   the response cannot be validated.
        """
        temperature = kwargs.pop("temperature", self._default_temperature)

        generation_config = kwargs.pop(
            "generation_config",
            {"temperature": temperature},
        )

        # Do not inherit the SDK's long retry window. M6 owns workflow retry;
        # provider retries here can otherwise leave the UI spinning for minutes.
        request_options = dict(kwargs.pop("request_options", {}) or {})
        request_options.setdefault("timeout", settings.gemini_timeout_seconds)
        request_options.setdefault("retry", None)
        kwargs["request_options"] = request_options

        try:
            model = self._genai.GenerativeModel(
                model_name=self._model_name,
                generation_config=generation_config,
            )
            raw = model.generate_content(prompt, **kwargs)
            return GeminiResponse(raw_response=raw)
        except Exception as exc:
            # Translate all provider errors into our typed LLMError so that
            # upper layers never need to import Gemini-specific exceptions.
            error_text = str(exc)
            if type(exc).__name__ == "ResourceExhausted" or "429" in error_text:
                retry_match = re.search(
                    r"(?:retry in|seconds:\s*)\s*(\d+(?:\.\d+)?)",
                    error_text,
                    flags=re.IGNORECASE,
                )
                retry_hint = (
                    f" Retry after approximately {round(float(retry_match.group(1)))} seconds."
                    if retry_match
                    else " Try again after the provider quota window resets."
                )
                raise LLMError(
                    "Gemini rate limit reached (HTTP 429)." + retry_hint
                ) from exc
            if type(exc).__name__ in {"DeadlineExceeded", "TimeoutError", "RetryError"}:
                raise LLMError(
                    f"Gemini request timed out after {settings.gemini_timeout_seconds:g} seconds."
                ) from exc
            raise LLMError(f"Gemini API error: {type(exc).__name__}: {error_text}") from exc
