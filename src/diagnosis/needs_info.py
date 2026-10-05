"""
DevTrace — Module 4: Deterministic NEEDS_INFO gate.

Responsibility
--------------
Before invoking the diagnosis generator, determine whether the incident contains
enough technical signal to proceed meaningfully.

This is NOT an LLM decision.
This is fully deterministic and rule-based.

The question is: "Do we have enough information to search for and apply evidence?"

NEEDS_INFO Triggers
--------------------
Case A — Missing current_version + competing version evidence
    If the retrieved candidates span multiple major versions (2.x AND 3.x
    both represented among top candidates), and the incident has no
    current_version, we cannot safely disambiguate.

    We do NOT trigger NEEDS_INFO just because a version is missing.
    If all retrieved candidates are version-agnostic ("*"), we can proceed.
    If candidates span multiple major versions but the incident has a clear
    version, we do NOT trigger NEEDS_INFO here (applicability handles it).

Case B — No meaningful technical signal
    If the incident has no error codes, no version, no technical terms, and
    a vague description that provides no distinguishable diagnostic signal.

    Important: we do NOT use a word-count threshold.
    We check for presence of *meaningful* technical content.

Sufficient incidents (should NOT trigger NEEDS_INFO):
    - Has error code(s), even without version
    - Has version AND non-trivial description
    - Has technical terms that narrow down the topic
    - Has structured fields (product + error_code, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models.contracts import RetrievalResult
from src.models.enums import SystemOutcome
from src.normalization.normalizer import NormalizedIncident


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NeedsInfoResult:
    """
    Result of the NEEDS_INFO gate.

    Attributes:
        triggered:     True if the system cannot proceed without more info.
        outcome:       Always NEEDS_INFO when triggered; None otherwise.
        reason:        Human-readable explanation of why more info is needed.
        missing_fields: List of field names that are absent/insufficient.
        unblock_hint:  Concrete suggestion for what the caller should provide.
    """

    triggered: bool
    outcome: SystemOutcome | None
    reason: str
    missing_fields: list[str]
    unblock_hint: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_meaningful_signal(normalized: NormalizedIncident) -> bool:
    """
    Return True if the incident has enough technical signal to support retrieval.

    Signal is considered meaningful when ANY of the following is present:
      - One or more error codes  (most specific signal — AUTH_401, RATE_429, etc.)
      - A current or previous version string  (scopes evidence to the right major version)
      - Recognized technical terms (oauth, webhook, bearer, token, signature, etc.)

    Deliberately excluded:
      - Product name alone — too generic on its own
      - Long descriptions that lack any of the above signals
      - Word count — "AUTH_401" (1 word) beats a 50-word vague incident

    The question is not "is the description long enough?" but
    "does the system know what to search for?"
    """
    signals = normalized.signals

    # Error codes present?  (strongest, most specific signal)
    if signals.error_codes:
        return True

    # Version present?  (scopes the evidence space deterministically)
    if signals.current_version or signals.previous_version:
        return True

    # Recognized technical terms from the M2 signal extraction vocabulary?
    # (oauth, webhook, bearer, token, hmac, rate limit, timeout, migration, etc.)
    if signals.technical_terms:
        return True

    # No actionable technical signal found.
    return False


def _spans_multiple_major_versions(
    retrieval_results: list[RetrievalResult],
) -> bool:
    """
    Return True if retrieved candidates span more than one major version
    (i.e., both 2.x-specific AND 3.x-specific docs are present).

    Version-agnostic docs ("*") are excluded from this count — they apply
    to all versions and do not create ambiguity.
    """
    import re as _re

    major_versions: set[int] = set()

    for rr in retrieval_results:
        applies_to: str = rr.metadata.get("applies_to", "*")
        if applies_to.strip() == "*":
            continue  # agnostic — no version conflict

        # Extract the major version from the lower bound.
        m = _re.search(r">=\s*(\d+)\.", applies_to)
        if m:
            major_versions.add(int(m.group(1)))

    return len(major_versions) > 1


# ---------------------------------------------------------------------------
# Public gate function
# ---------------------------------------------------------------------------

def needs_info_check(
    normalized: NormalizedIncident,
    retrieval_results: list[RetrievalResult],
) -> NeedsInfoResult:
    """
    Determine whether the incident has enough information to proceed to diagnosis.

    Args:
        normalized:        Output of M2 normalize_incident().
        retrieval_results: Top-ranked M3 retrieval results (pre-applicability).

    Returns:
        NeedsInfoResult with triggered=True if more information is required,
        or triggered=False if the pipeline may proceed.

    This function is called AFTER M3 retrieval so we can check whether the
    retrieved evidence spans conflicting version ranges.
    """
    incident = normalized.incident
    signals = normalized.signals

    # -------------------------------------------------------------------
    # Case B — No meaningful technical signal (checked first, independently
    # of retrieval results, because retrieval itself may be poor quality)
    # -------------------------------------------------------------------
    if not _has_meaningful_signal(normalized):
        return NeedsInfoResult(
            triggered=True,
            outcome=SystemOutcome.NEEDS_INFO,
            reason=(
                "The incident description does not contain sufficient technical signal "
                "(error code, version, technical term, or specific symptom) to form a "
                "meaningful evidence query."
            ),
            missing_fields=["error_codes", "current_version", "technical_description"],
            unblock_hint=(
                "Please provide at least one of: the specific error code (e.g. AUTH_401, "
                "RATE_429), the SDK/API version in use, the affected component or endpoint, "
                "or a more detailed description of the observed behaviour."
            ),
        )

    # -------------------------------------------------------------------
    # Case A — Missing current_version + competing version evidence
    # -------------------------------------------------------------------
    if incident.current_version is None and signals.current_version is None:
        # Only trigger NEEDS_INFO if retrieved evidence spans multiple major versions.
        # If all evidence is version-agnostic, we can still proceed.
        if retrieval_results and _spans_multiple_major_versions(retrieval_results):
            return NeedsInfoResult(
                triggered=True,
                outcome=SystemOutcome.NEEDS_INFO,
                reason=(
                    "The retrieved evidence spans multiple major SDK versions (2.x and 3.x), "
                    "and the incident does not specify a current version. The root cause and "
                    "fix differ significantly between versions — proceeding without version "
                    "information risks providing incorrect guidance."
                ),
                missing_fields=["current_version"],
                unblock_hint=(
                    "Please provide the current SDK/API version in use "
                    "(e.g. 'current_version: 3.1'). "
                    "This is required to select the applicable documentation."
                ),
            )

    # -------------------------------------------------------------------
    # All clear — sufficient information to proceed
    # -------------------------------------------------------------------
    return NeedsInfoResult(
        triggered=False,
        outcome=None,
        reason="",
        missing_fields=[],
        unblock_hint="",
    )
