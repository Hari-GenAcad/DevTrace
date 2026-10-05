"""
DevTrace — Module 4: Deterministic applicability checker.

Responsibility
--------------
Given a retrieved evidence chunk and a normalized incident, decide deterministically
whether that evidence applies to the incident's current version.

This is NOT retrieval.  M3 already retrieved candidates.
This is NOT an LLM decision.  Version matching is rule-based.

Applicability decisions
-----------------------
APPLICABLE     — chunk definitely applies to the incident's current version
NOT_APPLICABLE — chunk definitely does NOT apply (wrong major version)
UNKNOWN        — cannot determine applicability (e.g. missing current_version
                 and document is version-specific)

Version range format (from M2 corpus)
--------------------------------------
The corpus uses PEP 440-style specifiers stored in the ``applies_to`` field:

    ">=2.0,<3.0"   → SDK 2.x  (major version 2)
    ">=3.0,<4.0"   → SDK 3.x  (major version 3)
    "*"            → all versions (version-agnostic)

We evaluate a simple rule set rather than importing a full PEP 440 parser,
keeping the logic fully transparent and easy to audit.

Rules (in order)
-----------------
Rule 1 — Version-agnostic document
    If applies_to == "*" or applies_to lists both 2.x and 3.x ranges covering
    the incident version: APPLICABLE.

Rule 2 — Exact major-version compatibility
    Extract the major version from the incident's current_version (e.g. 3.1 → 3).
    If the document's version range covers that major version: APPLICABLE.

Rule 3 — Wrong major version
    If the document's version range covers a different major version: NOT_APPLICABLE.

Rule 4 — Missing current_version with version-specific document
    If current_version is None and the document is version-specific: UNKNOWN.

Rule 5 — Previous version is context only
    previous_version must never make a wrong-version document APPLICABLE.
    Only current_version drives the decision.

Traceability
-----------
Every decision is wrapped in M1's ApplicabilityResult so it can be stored in
the Trace and audited later.
"""

from __future__ import annotations

import re
from enum import Enum

from src.models.contracts import ApplicabilityResult, RetrievalResult


# ---------------------------------------------------------------------------
# Decision enum
# ---------------------------------------------------------------------------

class ApplicabilityDecision(str, Enum):
    """Three-way applicability decision."""

    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Version-range parsing helpers
# ---------------------------------------------------------------------------

# Matches individual specifiers like ">=2.0", "<3.0", ">=3.0", "<4.0"
_SPECIFIER_RE = re.compile(
    r"(>=|>|<=|<|==|!=)\s*(\d+)\.(\d+)(?:\.\d+)?"
)


def _parse_major_range(applies_to: str) -> set[int]:
    """
    Extract the set of major version integers that the applies_to range covers.

    Strategy: parse all specifiers and intersect implied major versions.

    Examples:
        ">=2.0,<3.0"   → {2}
        ">=3.0,<4.0"   → {3}
        ">=2.0,<4.0"   → {2, 3}
        "*"            → set()  (special: means ALL versions)

    Returns an empty set for "*" (caller must handle).
    Raises ValueError if the string cannot be parsed at all.
    """
    if applies_to.strip() == "*":
        return set()

    specifiers = _SPECIFIER_RE.findall(applies_to)
    if not specifiers:
        # Not parseable — treat as unknown range
        return set()

    # Collect lower and upper major bounds.
    lower_majors: set[int] = set()
    upper_exclusive: set[int] = set()
    upper_inclusive: set[int] = set()

    for op, major_str, _minor_str in specifiers:
        major = int(major_str)
        if op in (">=", ">"):
            lower_majors.add(major)
        elif op == "<":
            upper_exclusive.add(major)
        elif op in ("<=",):
            upper_inclusive.add(major)
        elif op == "==":
            lower_majors.add(major)
            upper_inclusive.add(major)

    # Determine the effective major range.
    # Typical patterns: ">=2.0,<3.0" → covers major 2 only.
    #                   ">=3.0,<4.0" → covers major 3 only.
    #                   ">=2.0,<4.0" → covers majors 2, 3.

    min_major = min(lower_majors) if lower_majors else 0

    # Upper bound: exclusive takes precedence over inclusive for overlapping.
    max_exclusive = min(upper_exclusive) if upper_exclusive else None
    max_inclusive = max(upper_inclusive) if upper_inclusive else None

    if max_exclusive is not None:
        # "< N" means major ≤ N-1
        effective_max_major = max_exclusive - 1
    elif max_inclusive is not None:
        effective_max_major = max_inclusive
    else:
        # No upper bound expressed — assume single major from lower bound.
        effective_max_major = min_major

    return set(range(min_major, effective_max_major + 1))


def _incident_major(version: str | None) -> int | None:
    """
    Extract the major version integer from an incident version string.

    Examples:
        "3.1"  → 3
        "2.8"  → 2
        "3.0"  → 3
        "4.0"  → 4
        None   → None
    """
    if version is None:
        return None
    m = re.match(r"(\d+)\.", version.strip())
    if m:
        return int(m.group(1))
    # Try bare integer (e.g. "3")
    m2 = re.match(r"^(\d+)$", version.strip())
    if m2:
        return int(m2.group(1))
    return None


# ---------------------------------------------------------------------------
# Core applicability logic
# ---------------------------------------------------------------------------

def _decide(
    current_version: str | None,
    applies_to: str,
) -> tuple[ApplicabilityDecision, str]:
    """
    Pure applicability logic — no Pydantic, no I/O.

    Returns (decision, reason_string).
    """
    at = applies_to.strip()

    # --- Rule 1: Version-agnostic document ---
    if at == "*":
        return (
            ApplicabilityDecision.APPLICABLE,
            "Document is version-agnostic (applies_to='*') — applicable to all versions.",
        )

    # Parse the major range the document covers.
    doc_majors = _parse_major_range(at)

    # Handle un-parseable range conservatively.
    if not doc_majors and at != "*":
        # We couldn't determine what versions the document covers.
        if current_version is None:
            return (
                ApplicabilityDecision.UNKNOWN,
                f"Cannot parse applies_to={at!r} and incident has no current_version.",
            )
        # If we can't parse the range, mark UNKNOWN — don't guess.
        return (
            ApplicabilityDecision.UNKNOWN,
            f"Cannot parse applies_to={at!r}; applicability cannot be determined.",
        )

    # --- Rule 4: Missing current_version with version-specific document ---
    if current_version is None:
        return (
            ApplicabilityDecision.UNKNOWN,
            (
                f"Incident has no current_version; cannot determine whether document "
                f"(applies_to={at!r}, covering major versions {sorted(doc_majors)}) applies."
            ),
        )

    incident_major = _incident_major(current_version)
    if incident_major is None:
        return (
            ApplicabilityDecision.UNKNOWN,
            f"Cannot parse major version from incident current_version={current_version!r}.",
        )

    # --- Rule 2: Exact major-version compatibility ---
    if incident_major in doc_majors:
        return (
            ApplicabilityDecision.APPLICABLE,
            (
                f"Incident current version {current_version} (major {incident_major}) "
                f"is within the document's version range {at!r}."
            ),
        )

    # --- Rule 3: Wrong major version ---
    return (
        ApplicabilityDecision.NOT_APPLICABLE,
        (
            f"Incident current version {current_version} (major {incident_major}) "
            f"is incompatible with document version range {at!r} "
            f"(covers major versions {sorted(doc_majors)})."
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_applicability(
    retrieval_result: RetrievalResult,
    current_version: str | None,
) -> ApplicabilityResult:
    """
    Determine whether a single retrieved evidence chunk applies to the incident.

    Args:
        retrieval_result: A single M3 RetrievalResult with metadata.applies_to.
        current_version:  The incident's current_version (may be None).

    Returns:
        ApplicabilityResult with a traceable decision and reason.

    Note:
        previous_version is intentionally NOT an argument here.
        Only current_version drives applicability decisions (Rule 5).
    """
    # Extract applies_to from the RetrievalResult metadata (set by M3).
    # Fall back to "*" if somehow absent — conservative (applicable).
    applies_to: str = retrieval_result.metadata.get("applies_to", "*")

    decision, reason = _decide(current_version, applies_to)

    return ApplicabilityResult(
        chunk_id=retrieval_result.chunk_id,
        doc_id=retrieval_result.doc_id,
        applicable=(decision == ApplicabilityDecision.APPLICABLE),
        reason=reason,
        incident_version=current_version,
        document_range=applies_to,
    )


def filter_applicable(
    retrieval_results: list[RetrievalResult],
    current_version: str | None,
) -> tuple[list[RetrievalResult], list[ApplicabilityResult]]:
    """
    Filter a batch of retrieval results into applicable and non-applicable sets.

    UNKNOWN decisions are treated as NOT applicable for the evidence bundle
    (we do not pass uncertain evidence to the diagnosis generator).

    Args:
        retrieval_results: Ranked M3 retrieval results.
        current_version:   Incident's current version (may be None).

    Returns:
        (applicable_results, all_decisions) where:
            - applicable_results: RetrievalResult items that passed applicability
            - all_decisions:      ApplicabilityResult for every input item
                                  (for traceability, regardless of decision)
    """
    all_decisions: list[ApplicabilityResult] = []
    applicable_results: list[RetrievalResult] = []

    for rr in retrieval_results:
        decision = check_applicability(rr, current_version)
        all_decisions.append(decision)
        if decision.applicable:
            applicable_results.append(rr)

    return applicable_results, all_decisions
