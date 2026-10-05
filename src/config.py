"""
DevTrace configuration management.

Loads settings from environment variables / .env file.
API keys must never be hardcoded here.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DevTraceConfig(BaseSettings):
    """Central configuration for DevTrace.

    All values can be overridden via environment variables or a .env file.
    The GEMINI_API_KEY must be set in the environment before creating the
    production Gemini client; it is optional here so that tests that never
    touch the real client can still import and use the config object.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Runtime environment ---
    devtrace_env: Literal["development", "test", "production"] = Field(
        default="development",
        description="Runtime environment.",
    )

    # --- Gemini / LLM settings ---
    gemini_api_key: str | None = Field(
        default=None,
        description="Gemini API key. Must be set for production use.",
    )
    gemini_model: str = Field(
        default="gemini-1.5-pro",
        description="Pinned Gemini model name.",
    )
    gemini_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature. Default 0 for deterministic outputs.",
    )

    @field_validator("gemini_temperature", mode="before")
    @classmethod
    def _coerce_temperature(cls, v: object) -> float:
        return float(v)


# Module-level singleton — callers import `settings` directly.
settings = DevTraceConfig()
