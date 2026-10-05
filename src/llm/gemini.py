"""
Production Gemini LLM client.

Wraps google-generativeai behind the LLMClient interface.
Real API calls are never made during the test suite — tests use FakeLLMClient.
"""

from __future__ import annotations

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

    def __init__(self) -> None:
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
        self._model_name = settings.gemini_model
        self._default_temperature = settings.gemini_temperature

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
            raise LLMError(
                f"Gemini API error: {type(exc).__name__}: {exc}"
            ) from exc
