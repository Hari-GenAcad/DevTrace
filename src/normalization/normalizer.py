"""
Incident normalization layer.

Converts raw input (structured dict, natural-language string, or a mix)
into a validated TroubleshootingIncident with extracted signals.

Design rules:
  - Structured fields take precedence over natural-language extraction.
  - Signal extraction is deterministic (no LLM).
  - The normalizer never invents information that is not in the input.
  - If current_version cannot be reliably determined, it remains None.
    The NEEDS_INFO path (M4) handles that downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.contracts import TroubleshootingIncident
from src.normalization.signals import ExtractedSignals, extract_signals


@dataclass
class NormalizedIncident:
    """
    The output of incident normalization.

    Bundles the structured incident with the extracted deterministic signals
    so downstream modules don't need to re-run extraction.
    """

    incident: TroubleshootingIncident
    signals: ExtractedSignals


def normalize_incident(
    description: str,
    *,
    current_version: str | None = None,
    previous_version: str | None = None,
    error_codes: list[str] | None = None,
    product: str | None = None,
    context: dict[str, Any] | None = None,
) -> NormalizedIncident:
    """
    Normalize a troubleshooting incident from structured and/or natural-language input.

    Structured keyword arguments always take precedence over values that could
    be inferred from the description text.

    Args:
        description: Free-text incident description. Required.
        current_version: Explicit current version (overrides text extraction).
        previous_version: Explicit previous version (overrides text extraction).
        error_codes: Explicit error codes (merged with text-extracted codes,
                     structured codes appear first).
        product: Product or component name.
        context: Arbitrary structured context dict.

    Returns:
        NormalizedIncident containing the validated TroubleshootingIncident
        and the ExtractedSignals.

    Raises:
        pydantic.ValidationError: If the incident description is empty.
    """
    # Run signal extraction. Structured fields are passed in so they take
    # precedence inside extract_signals.
    signals = extract_signals(
        description,
        structured_error_codes=error_codes or [],
        structured_current_version=current_version,
        structured_previous_version=previous_version,
    )

    # Build the incident. Prefer explicitly provided structured values;
    # fall back to signal-extracted values.
    incident = TroubleshootingIncident(
        description=description,
        current_version=current_version if current_version is not None else signals.current_version,
        previous_version=previous_version if previous_version is not None else signals.previous_version,
        error_codes=signals.error_codes,  # already merged inside extract_signals
        product=product,
        context=context or {},
    )

    return NormalizedIncident(incident=incident, signals=signals)


def normalize_incident_from_dict(raw: dict[str, Any]) -> NormalizedIncident:
    """
    Normalize from a raw dictionary (e.g. from a JSON payload or eval dataset).

    The dict may contain any combination of the fields accepted by
    normalize_incident(). At minimum it must have 'description'.

    Args:
        raw: Dictionary with incident fields.

    Returns:
        NormalizedIncident.
    """
    return normalize_incident(
        description=raw["description"],
        current_version=raw.get("current_version"),
        previous_version=raw.get("previous_version"),
        error_codes=raw.get("error_codes"),
        product=raw.get("product"),
        context=raw.get("context"),
    )
