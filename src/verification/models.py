"""
DevTrace — Module 5: Verification data models.

These are M5-specific types. They are separate from the M1 VerificationVerdict
contract (which is kept there for the Trace) so M5 can carry richer per-claim
detail without bloating the shared contracts file.

Key distinctions maintained here:
    citation_valid    — deterministic: are cited IDs actually in the applicable bundle?
    citation_correct  — semantic: does the cited evidence actually support the claim?
    sufficient        — semantic: is the evidence strong enough to establish the claim?
    contradicted      — semantic: does any applicable evidence conflict with the claim?
    verdict           — computed from all four: VERIFIED iff all pass.

A claim can have citation_valid=True and citation_correct=False — the ID exists
in the bundle but the content doesn't support the claim. This distinction is
important for debugging and evaluation.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VerificationVerdict(str, Enum):
    """Binary outcome for a single claim's verification."""

    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"


class CitationValidity(str, Enum):
    """
    Deterministic citation check result.

    VALID   — all evidence_ids in the claim exist in the applicable bundle.
    INVALID — at least one evidence_id is missing/unknown/not applicable.
    EMPTY   — the claim has no evidence_ids at all.
    """

    VALID = "VALID"
    INVALID = "INVALID"
    EMPTY = "EMPTY"


# ---------------------------------------------------------------------------
# Per-claim verification result
# ---------------------------------------------------------------------------

class ClaimVerification(BaseModel):
    """
    Full verification record for a single DiagnosisClaim.

    Fields are kept separate so callers can distinguish the failure mode:

        citation_valid=True, citation_correct=False
            → the ID exists but content doesn't support the claim
        citation_valid=False
            → the ID was hallucinated or references a non-applicable chunk
        sufficient=False
            → the evidence is related but too weak to justify the claim
        contradicted=True
            → applicable evidence actively conflicts with the claim

    verdict is VERIFIED only when:
        citation_valid IN {VALID} AND citation_correct AND sufficient AND NOT contradicted.
    """

    claim_id: str = Field(..., description="ID of the DiagnosisClaim being evaluated.")
    verdict: VerificationVerdict = Field(
        ...,
        description="VERIFIED or REJECTED.",
    )
    citation_validity: CitationValidity = Field(
        ...,
        description=(
            "Deterministic result: are the cited IDs present in the applicable bundle?"
        ),
    )
    citation_correct: bool = Field(
        ...,
        description="Semantic: does the cited evidence actually support the claim?",
    )
    sufficient: bool = Field(
        ...,
        description=(
            "Semantic: is the available evidence strong enough to establish the claim?"
        ),
    )
    contradicted: bool = Field(
        ...,
        description="Semantic: does any applicable evidence conflict with the claim?",
    )
    reason: str = Field(
        default="",
        description="Human-readable explanation of the verdict.",
    )
    supporting_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence IDs the verifier found to support the claim.",
    )
    contradicting_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence IDs the verifier found to contradict the claim.",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Batch verification result
# ---------------------------------------------------------------------------

class DiagnosisVerification(BaseModel):
    """
    Complete M5 verification result for a full DiagnosisResult.

    claim_verifications: One ClaimVerification per DiagnosisClaim.
    all_verified:        True only when every claim is VERIFIED.
    """

    claim_verifications: list[ClaimVerification] = Field(
        default_factory=list,
        description="Per-claim verification results, in diagnosis claim order.",
    )

    @property
    def all_verified(self) -> bool:
        """True if every claim is VERIFIED."""
        return all(
            cv.verdict == VerificationVerdict.VERIFIED
            for cv in self.claim_verifications
        )

    @property
    def verified_claim_ids(self) -> list[str]:
        """IDs of claims that passed verification."""
        return [
            cv.claim_id
            for cv in self.claim_verifications
            if cv.verdict == VerificationVerdict.VERIFIED
        ]

    @property
    def rejected_claim_ids(self) -> list[str]:
        """IDs of claims that failed verification."""
        return [
            cv.claim_id
            for cv in self.claim_verifications
            if cv.verdict == VerificationVerdict.REJECTED
        ]

    model_config = {"frozen": True}
