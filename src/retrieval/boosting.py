"""
DevTrace — Module 3: Signal-based score boosting.

Responsibilities:
  - Accept a retrieval candidate (chunk content + metadata) and the incident
    signals extracted by M2.
  - Compute an additive boost that reflects how well the chunk matches the
    most important incident signals.
  - Return the individual boost components for transparency (the UI and trace
    can show exactly why a chunk received extra weight).

Design rules:
  - This is a RANKING boost, NOT an eligibility filter.
  - A chunk with score 0 on every boost is still in the result list.
  - Version matching produces a BOOST, not a rejection.  M4 owns applicability.
  - The total boost is capped at RetrievalBoosts.max_total.

Boost breakdown (see RetrievalBoosts in config.py for the numeric values):
  error_code_match    — incident error code appears in chunk content/metadata.
  technical_term_match — a known incident technical term appears in the chunk.
  version_match       — chunk's applies_to range mentions the incident version.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.normalization.signals import ExtractedSignals
from src.retrieval.config import RetrievalBoosts


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class BoostResult:
    """
    Detailed breakdown of the signal boost applied to one retrieval candidate.

    Transparent by design — every component is visible in the trace so we
    can explain "why did this chunk rank highly?".
    """

    error_code_boost: float = 0.0
    technical_term_boost: float = 0.0
    version_boost: float = 0.0
    total: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_error_code_match(
    content_lower: str,
    metadata_error_codes: list[str],
    incident_error_codes: list[str],
) -> bool:
    """
    Return True if any incident error code appears in the chunk content or
    its structured metadata error_codes list.
    """
    if not incident_error_codes:
        return False
    meta_set = {c.upper() for c in metadata_error_codes}
    for code in incident_error_codes:
        code_upper = code.upper()
        if code_upper in meta_set:
            return True
        if code_upper.lower() in content_lower:
            return True
    return False


def _check_technical_term_match(
    content_lower: str,
    incident_terms: list[str],
) -> bool:
    """
    Return True if any incident technical term appears in the chunk content.
    """
    if not incident_terms:
        return False
    for term in incident_terms:
        if term.lower() in content_lower:
            return True
    return False


def _check_version_match(
    applies_to: str,
    current_version: str | None,
    previous_version: str | None,
) -> bool:
    """
    Return True if the chunk's applies_to range mentions the incident version.

    This is a lightweight textual heuristic — full PEP 440 range evaluation
    is M4's job.  Here we only check for major version hints to provide a
    modest retrieval signal.

    Examples:
        applies_to=">=3.0,<4.0", current_version="3.1"  → True  (3.x match)
        applies_to=">=2.0,<3.0", current_version="3.1"  → False
        applies_to="*"                                    → False (agnostic; no boost needed)
    """
    if applies_to == "*" or not applies_to:
        return False

    # Extract major version from incident string (e.g. "3.1" → "3").
    def _major(v: str | None) -> str | None:
        if v is None:
            return None
        parts = v.split(".")
        return parts[0] if parts else None

    curr_major = _major(current_version)
    prev_major = _major(previous_version)

    # Textual scan: does the applies_to string contain the major version digit?
    for major in [curr_major, prev_major]:
        if major and f">={major}." in applies_to:
            return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_boost(
    *,
    content: str,
    applies_to: str,
    metadata_error_codes: list[str],
    signals: ExtractedSignals,
    boosts: RetrievalBoosts,
) -> BoostResult:
    """
    Compute the additive signal boost for a single retrieval candidate.

    Args:
        content:              The chunk's text content.
        applies_to:           The chunk's version range string.
        metadata_error_codes: Error codes from the chunk's structured metadata.
        signals:              M2 signals extracted from the incident.
        boosts:               Boost weights from RetrievalConfig.

    Returns:
        BoostResult with per-component breakdown and capped total.
    """
    content_lower = content.lower()
    result = BoostResult()

    # --- Error code match ---
    if _check_error_code_match(
        content_lower, metadata_error_codes, signals.error_codes
    ):
        result.error_code_boost = boosts.error_code_match

    # --- Technical term match ---
    if _check_technical_term_match(content_lower, signals.technical_terms):
        result.technical_term_boost = boosts.technical_term_match

    # --- Version match (boost only; NOT a filter) ---
    if _check_version_match(applies_to, signals.current_version, signals.previous_version):
        result.version_boost = boosts.version_match

    raw_total = (
        result.error_code_boost
        + result.technical_term_boost
        + result.version_boost
    )
    result.total = min(raw_total, boosts.max_total)
    return result
