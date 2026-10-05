"""
Deterministic signal extraction from troubleshooting incidents.

Extracts:
  - Error codes (e.g. AUTH_401, RATE_429, ERR_TIMEOUT)
  - Version numbers (e.g. 3.1, v2.8, SDK 3.0)
  - Meaningful technical terms

No LLM is used here. All extraction is regex-based and deterministic.

Structured fields on TroubleshootingIncident always take precedence over
what is extracted from free text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Error codes: uppercase letters/digits separated by underscores, starting
# with a letter group. Examples: AUTH_401, RATE_429, ERR_TIMEOUT, NOT_FOUND.
_ERROR_CODE_PATTERN = re.compile(
    r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b"
)

# Version numbers: an optional leading "v" or "version"/"SDK" prefix,
# followed by digits.digits (e.g. 3.1, v2.8, SDK 3.0, version 2.5).
# Captures just the numeric version (e.g. "3.1", "2.8").
_VERSION_PATTERN = re.compile(
    r"(?:SDK\s+|API\s+|v(?:ersion\s+)?)?(\d+\.\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)

# Upgrade pattern: "from X to Y" or "X to Y" where both look like versions.
# Captures (from_version, to_version).
_UPGRADE_PATTERN = re.compile(
    r"(?:from\s+)?v?(\d+\.\d+(?:\.\d+)?)\s+to\s+v?(\d+\.\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Past-tense version references: "used to run X", "was on X", "ran X",
# "previously on X" etc.  These suggest a historical version, not current.
_HISTORICAL_PATTERN = re.compile(
    r"(?:used\s+to(?:\s+run)?|was\s+on|previously\s+(?:on|running|using)|ran\s+(?:on\s+)?)\s+v?(\d+\.\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# HTTP status code words that map to error codes.
_HTTP_STATUS_PATTERN = re.compile(
    r"\bHTTP[_\s]?(4\d{2}|5\d{2})\b",
    re.IGNORECASE,
)

# Technical term heuristics: known important tokens for DevCore troubleshooting.
_TECHNICAL_TERMS: frozenset[str] = frozenset({
    "oauth", "bearer", "token", "api key", "apikey", "webhook", "signature",
    "hmac", "sha256", "sha1", "pagination", "cursor", "offset", "rate limit",
    "timeout", "proxy", "tls", "ssl", "certificate", "environment variable",
    "client id", "client secret", "migration", "upgrade",
})


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------

@dataclass
class ExtractedSignals:
    """
    Deterministic signals extracted from an incident.

    All extraction is from the incident description and structured fields.
    Structured fields (error_codes, current_version, etc.) always win
    over extracted values if both are present.
    """

    error_codes: list[str] = field(default_factory=list)
    """Normalised error code strings, e.g. ['AUTH_401', 'RATE_429']."""

    current_version: str | None = None
    """Inferred current version, e.g. '3.1'."""

    previous_version: str | None = None
    """Inferred previous/from version, e.g. '2.8'."""

    technical_terms: list[str] = field(default_factory=list)
    """Relevant technical keywords found in the description."""

    raw_versions_found: list[str] = field(default_factory=list)
    """All version strings found in text, for traceability."""

    def to_dict(self) -> dict:
        return {
            "error_codes": self.error_codes,
            "current_version": self.current_version,
            "previous_version": self.previous_version,
            "technical_terms": self.technical_terms,
            "raw_versions_found": self.raw_versions_found,
        }


# ---------------------------------------------------------------------------
# Internal extraction helpers
# ---------------------------------------------------------------------------

def _extract_error_codes(text: str) -> list[str]:
    """Extract all error codes from text, deduplicated, original order preserved."""
    found = _ERROR_CODE_PATTERN.findall(text)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    result: list[str] = []
    for code in found:
        if code not in seen:
            seen.add(code)
            result.append(code)
    return result


def _extract_versions_from_text(text: str) -> tuple[str | None, str | None, list[str]]:
    """
    Extract current and previous version from free text.

    Strategy:
    1. Look for explicit upgrade pattern: "from X to Y" → previous=X, current=Y.
    2. Look for historical markers: "used to run X" → historical (not current).
    3. Remaining version mentions become current_version candidates.
    4. If only one version mentioned (and no historical context), it's current.
    5. If ambiguous (multiple non-upgrade versions), leave current_version=None.

    Returns:
        (current_version, previous_version, all_raw_versions)
    """
    text_lower = text.lower()

    # Step 1: Look for upgrade/migration pattern.
    upgrade_match = _UPGRADE_PATTERN.search(text)
    if upgrade_match:
        prev_v = upgrade_match.group(1)
        curr_v = upgrade_match.group(2)
        all_raw = _VERSION_PATTERN.findall(text)
        return curr_v, prev_v, list(dict.fromkeys(all_raw))

    # Step 2: Find historical version references (these are NOT current).
    historical_matches = {m.group(1) for m in _HISTORICAL_PATTERN.finditer(text)}

    # Step 3: All version mentions.
    all_raw = list(dict.fromkeys(_VERSION_PATTERN.findall(text)))

    # Step 4: Non-historical versions are current candidates.
    candidate_current = [v for v in all_raw if v not in historical_matches]

    if len(candidate_current) == 1:
        current_version = candidate_current[0]
        previous_version = next(iter(historical_matches), None)
        return current_version, previous_version, all_raw

    # Step 5: Multiple non-historical versions or none — ambiguous.
    # Do NOT infer current version from vague context.
    return None, None, all_raw


def _extract_technical_terms(text: str) -> list[str]:
    """Return known technical terms present in the text."""
    text_lower = text.lower()
    found = [term for term in _TECHNICAL_TERMS if term in text_lower]
    return sorted(found)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_signals(
    description: str,
    *,
    structured_error_codes: list[str] | None = None,
    structured_current_version: str | None = None,
    structured_previous_version: str | None = None,
) -> ExtractedSignals:
    """
    Extract deterministic signals from an incident description.

    Structured fields (from TroubleshootingIncident) always take precedence
    over text-extracted values when both are present.

    Args:
        description: The incident's free-text description.
        structured_error_codes: Error codes already present on the incident
                                 (takes precedence over text extraction).
        structured_current_version: Current version already on the incident.
        structured_previous_version: Previous version already on the incident.

    Returns:
        ExtractedSignals with all found signals.
    """
    # --- Error codes ---
    text_error_codes = _extract_error_codes(description)
    if structured_error_codes:
        # Merge: structured first, then add any text-found that aren't already there.
        merged_codes = list(structured_error_codes)
        seen = set(structured_error_codes)
        for code in text_error_codes:
            if code not in seen:
                merged_codes.append(code)
                seen.add(code)
        final_error_codes = merged_codes
    else:
        final_error_codes = text_error_codes

    # --- Versions ---
    text_current, text_previous, raw_versions = _extract_versions_from_text(description)

    # Structured always wins.
    final_current = structured_current_version if structured_current_version is not None else text_current
    final_previous = structured_previous_version if structured_previous_version is not None else text_previous

    # --- Technical terms ---
    tech_terms = _extract_technical_terms(description)

    return ExtractedSignals(
        error_codes=final_error_codes,
        current_version=final_current,
        previous_version=final_previous,
        technical_terms=tech_terms,
        raw_versions_found=raw_versions,
    )
