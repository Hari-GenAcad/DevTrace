"""
Shared enums used across the DevTrace architecture.

Centralised here so no module duplicates string literals.
"""

from __future__ import annotations

from enum import Enum


# ---------------------------------------------------------------------------
# System-level outcomes
# ---------------------------------------------------------------------------

class SystemOutcome(str, Enum):
    """Top-level result produced by the DevTrace pipeline."""

    ANSWERED = "ANSWERED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NEEDS_INFO = "NEEDS_INFO"
    DEGRADED = "DEGRADED"


class AnswerCompleteness(str, Enum):
    """Sub-state of a ANSWERED outcome."""

    FULL = "FULL"
    PARTIAL = "PARTIAL"


# ---------------------------------------------------------------------------
# Diagnosis / claim roles
# ---------------------------------------------------------------------------

class ClaimRole(str, Enum):
    """Semantic role of a single diagnosis claim."""

    ROOT_CAUSE = "root_cause"
    FIX = "fix"
    EXPLANATION = "explanation"


# ---------------------------------------------------------------------------
# Retrieval source
# ---------------------------------------------------------------------------

class RetrievalSource(str, Enum):
    """How a candidate evidence chunk was surfaced."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Verification verdict
# ---------------------------------------------------------------------------

class VerificationStatus(str, Enum):
    """Outcome of verifying a single claim against applicable evidence."""

    PASS = "PASS"
    FAIL = "FAIL"


class SupportDecision(str, Enum):
    """Does the cited evidence actually support the claim?"""

    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    INSUFFICIENT = "INSUFFICIENT"


class ContradictionDecision(str, Enum):
    """Is there contradictory applicable evidence?"""

    NONE = "NONE"
    CONTRADICTED = "CONTRADICTED"
    UNRESOLVED = "UNRESOLVED"
