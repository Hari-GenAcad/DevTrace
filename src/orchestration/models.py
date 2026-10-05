"""
DevTrace — Module 6: Orchestration models.

Defines the M6 result contract that carries the full end-to-end
pipeline record for a single troubleshooting run.

Design goals
------------
- Rich enough for M7 evaluation (all intermediate stages preserved).
- Rich enough for M8 UI (final answer and outcome always top-level).
- Keeps separate the initial diagnosis/verification vs retry diagnosis/verification.
- Clearly exposes retry_attempted and retry_reason for observability.
- Uses only enums and Pydantic models consistent with M1-M5.

Outcome semantics
-----------------
ANSWERED_FULL
    root_cause, fix, and explanation all VERIFIED.

ANSWERED_PARTIAL
    root_cause VERIFIED, but fix or explanation is REJECTED/missing.
    Also: root_cause + fix VERIFIED, but no explanation claim.

INSUFFICIENT_EVIDENCE
    root_cause is NOT verified after both initial diagnosis AND retry.
    This is a claim-quality outcome, not a system failure.

NEEDS_INFO
    M4 returned NEEDS_INFO early. No diagnosis or verification was run.

DEGRADED
    A system/API/schema failure prevented reliable completion.
    Do NOT confuse with INSUFFICIENT_EVIDENCE: the problem is infrastructure,
    not diagnosis quality.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.models.contracts import DiagnosisClaim, DiagnosisResult, RetrievalResult
from src.models.enums import SystemOutcome
from src.verification.models import DiagnosisVerification


# ---------------------------------------------------------------------------
# M6 outcome enum
# ---------------------------------------------------------------------------

class FinalOutcome(str, Enum):
    """
    M6 top-level outcome classification.

    Distinct from M1's SystemOutcome because M6 separates ANSWERED into
    ANSWERED_FULL / ANSWERED_PARTIAL, and adds the specific failure modes.
    """

    ANSWERED_FULL = "ANSWERED_FULL"
    """All key claims (root_cause, fix, explanation) are VERIFIED."""

    ANSWERED_PARTIAL = "ANSWERED_PARTIAL"
    """root_cause VERIFIED but fix or explanation rejected/absent."""

    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    """root_cause could not be verified even after one retry."""

    NEEDS_INFO = "NEEDS_INFO"
    """M4 determined the incident lacks sufficient information."""

    DEGRADED = "DEGRADED"
    """A system failure (LLM/schema error) prevented reliable completion."""


# ---------------------------------------------------------------------------
# Final answer type (M6 output)
# ---------------------------------------------------------------------------

class VerifiedAnswer(BaseModel):
    """
    The final user-facing answer assembled from VERIFIED claims only.

    Rejected claims are never included here. If root_cause is not verified,
    answer_text will be None and verified_claims will be empty.
    """

    outcome: FinalOutcome = Field(..., description="Top-level M6 outcome.")
    answer_text: str | None = Field(
        default=None,
        description=(
            "Human-readable answer assembled from verified claims. "
            "None when outcome is INSUFFICIENT_EVIDENCE, NEEDS_INFO, or DEGRADED."
        ),
    )
    verified_claims: list[DiagnosisClaim] = Field(
        default_factory=list,
        description="Claims that passed verification (in role order).",
    )
    needs_info_reason: str | None = Field(
        default=None,
        description="Set when outcome is NEEDS_INFO.",
    )
    error_detail: str | None = Field(
        default=None,
        description="Set when outcome is DEGRADED.",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Complete M6 result record
# ---------------------------------------------------------------------------

class TroubleshootingResult(BaseModel):
    """
    The complete structured result of one DevTrace troubleshooting run (M6).

    All intermediate pipeline stages are preserved here so M7 can evaluate
    retrieval behaviour, applicability decisions, verification quality,
    and retry behaviour without needing to re-run the pipeline.

    Fields
    ------
    incident_description:
        The raw incident description text.
    current_version:
        The incident's current_version used in the run.

    retrieval_results:
        All M3 retrieval candidates (pre-applicability).
    applicability_decisions:
        Per-chunk M4 applicability results.
    applicable_results:
        Chunks that passed M4 applicability filtering.

    initial_diagnosis:
        The M4 diagnosis (before verification). None if M4 degraded/needs_info.
    initial_verification:
        The M5 verification of initial_diagnosis. None if skipped.

    retry_attempted:
        True if the one-shot retry was triggered.
    retry_reason:
        Why the retry was triggered (for observability). None if no retry.
    retry_diagnosis:
        The retry diagnosis. None if no retry was performed.
    retry_verification:
        The M5 verification of retry_diagnosis. None if no retry.

    final_answer:
        The assembled verified answer from surviving claims.
    final_outcome:
        Top-level M6 outcome classification.

    system_errors:
        Non-fatal error messages collected during the run (for diagnostics).
    """

    # Incident identification
    incident_description: str = Field(..., description="Raw incident description text.")
    current_version: str | None = Field(
        default=None,
        description="Incident current_version (may be None if not provided).",
    )

    # M3 + M4 intermediate results
    retrieval_results: list[RetrievalResult] = Field(
        default_factory=list,
        description="All M3 retrieval candidates (pre-applicability).",
    )
    applicability_decisions: list[Any] = Field(
        default_factory=list,
        description="Per-chunk M4 applicability decisions (ApplicabilityResult list).",
    )
    applicable_results: list[RetrievalResult] = Field(
        default_factory=list,
        description="Chunks that passed M4 applicability filtering.",
    )

    # Initial diagnosis + verification
    initial_diagnosis: DiagnosisResult | None = Field(
        default=None,
        description="Structured diagnosis from M4 (pre-verification).",
    )
    initial_verification: DiagnosisVerification | None = Field(
        default=None,
        description="M5 verification of initial_diagnosis.",
    )

    # Retry
    retry_attempted: bool = Field(
        default=False,
        description="True if the targeted one-shot retry was triggered.",
    )
    retry_reason: str | None = Field(
        default=None,
        description="Why retry was triggered (root_cause verification failure reason).",
    )
    retry_diagnosis: DiagnosisResult | None = Field(
        default=None,
        description="Diagnosis produced by the retry attempt.",
    )
    retry_verification: DiagnosisVerification | None = Field(
        default=None,
        description="M5 verification of retry_diagnosis.",
    )

    # Final output
    final_answer: VerifiedAnswer = Field(
        ...,
        description="The assembled verified answer (or INSUFFICIENT_EVIDENCE/DEGRADED record).",
    )
    final_outcome: FinalOutcome = Field(
        ...,
        description="Top-level M6 outcome classification.",
    )

    # Diagnostics
    system_errors: list[str] = Field(
        default_factory=list,
        description="Non-fatal error messages collected during the run.",
    )

    model_config = {"frozen": True}
