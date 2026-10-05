"""
DevTrace project-level error types.

Keeps the exception hierarchy small and meaningful.
Later modules map these to system outcomes (e.g. DEGRADED).
"""

from __future__ import annotations


class DevTraceError(Exception):
    """Base class for all DevTrace exceptions."""


class ConfigurationError(DevTraceError):
    """Raised when required configuration is absent or invalid.

    Example: missing GEMINI_API_KEY when the real client is requested.
    """


class LLMError(DevTraceError):
    """Raised when the LLM provider returns an error or is unreachable.

    This maps to the DEGRADED system outcome when the failure is
    unrecoverable (e.g. API timeout, HTTP 5xx, quota exceeded).
    """


class SchemaValidationError(DevTraceError):
    """Raised when a model response cannot be parsed into the expected schema.

    Distinct from pydantic.ValidationError so callers can catch it without
    importing pydantic directly.
    """
