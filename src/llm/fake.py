"""
FakeLLMClient — deterministic, offline LLM client for testing.

Design goals:
  1. Zero network access.
  2. Fully deterministic.
  3. Easy to configure inside tests.
  4. Can simulate malformed responses.
  5. Can simulate exceptions / provider failures.
  6. Knows nothing about DevTrace business logic.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from src.errors import LLMError
from src.llm.base import LLMClient, LLMResponse


class FakeLLMResponse(LLMResponse):
    """A concrete LLMResponse backed by a plain string."""

    def __init__(self, text: str, *, success: bool = True) -> None:
        self._text = text
        self._success = success

    @property
    def text(self) -> str:
        return self._text

    @property
    def success(self) -> bool:
        return self._success

    @property
    def raw(self) -> Any:
        return {"fake_text": self._text}


class FakeLLMClient(LLMClient):
    """
    Controllable fake LLM client for deterministic offline testing.

    Usage modes:

    1. Single fixed response:
       >>> client = FakeLLMClient(response="Hello")
       >>> client.generate("anything").text
       'Hello'

    2. Sequence of responses (consumed in order):
       >>> client = FakeLLMClient(responses=["first", "second"])
       >>> client.generate("q1").text
       'first'
       >>> client.generate("q2").text
       'second'

    3. Raise an exception on call:
       >>> client = FakeLLMClient(exception=LLMError("timeout"))
       >>> client.generate("q")  # raises LLMError

    4. Sequence with an exception at a specific position:
       >>> client = FakeLLMClient(responses=["ok", LLMError("fail")])
       >>> client.generate("q1").text  # "ok"
       >>> client.generate("q2")       # raises LLMError("fail")

    When the response queue is exhausted, subsequent calls raise LLMError
    unless a fixed `response` was provided (in which case it repeats).
    """

    def __init__(
        self,
        *,
        response: str | None = None,
        responses: list[str | Exception] | None = None,
        exception: Exception | None = None,
    ) -> None:
        if exception is not None and (response is not None or responses is not None):
            raise ValueError(
                "Specify either 'exception' or 'response'/'responses', not both."
            )

        self._fixed_response = response
        self._exception = exception

        # Build a deque from a mixed list of strings and exceptions
        if responses is not None:
            self._queue: deque[str | Exception] = deque(responses)
        else:
            self._queue = deque()

        self._call_count = 0

    def generate(self, prompt: str, **kwargs: Any) -> FakeLLMResponse:
        """Return the next predetermined response or raise a predetermined error."""
        self._call_count += 1

        # Global exception mode
        if self._exception is not None:
            raise self._exception

        # Sequence mode
        if self._queue:
            item = self._queue.popleft()
            if isinstance(item, Exception):
                raise item
            return FakeLLMResponse(text=item)

        # Fixed single-response mode (repeating)
        if self._fixed_response is not None:
            return FakeLLMResponse(text=self._fixed_response)

        # Queue exhausted with no fixed fallback
        raise LLMError(
            f"FakeLLMClient: response queue exhausted after {self._call_count} call(s)."
        )

    @property
    def call_count(self) -> int:
        """How many times generate() has been called."""
        return self._call_count
