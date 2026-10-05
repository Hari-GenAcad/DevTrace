"""
LLM abstraction layer for DevTrace.

The rest of the system depends on LLMClient, never on Gemini-specific code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMResponse(ABC):
    """
    Base class for all LLM responses.

    Concrete implementations (GeminiResponse, FakeLLMResponse) wrap the
    provider-specific response objects and expose a uniform interface.
    """

    @property
    @abstractmethod
    def text(self) -> str:
        """The raw text content of the response."""

    @property
    @abstractmethod
    def success(self) -> bool:
        """True if the response contains usable content."""

    @property
    @abstractmethod
    def raw(self) -> Any:
        """The underlying provider response object (for debugging / logging)."""


class LLMClient(ABC):
    """
    Abstract LLM client interface.

    All modules that need language model access depend on this interface.
    The production implementation is GeminiClient; tests use FakeLLMClient.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Send a prompt and return an LLMResponse.

        Args:
            prompt: The full prompt string to send.
            **kwargs: Provider-specific overrides (e.g. temperature, schema).

        Returns:
            LLMResponse with the model output.

        Raises:
            LLMError: On API failure, timeout, or quota exhaustion.
            SchemaValidationError: If structured output fails schema validation.
        """
