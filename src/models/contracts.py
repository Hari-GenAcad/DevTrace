"""
Core Pydantic contracts for DevTrace — M1.

These types are the stable foundation that M2–M8 build on.
No business logic lives here; only data shape and basic structural validation.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from src.models.enums import (
    AnswerCompleteness,
    ClaimRole,
    ContradictionDecision,
    RetrievalSource,
    SupportDecision,
    SystemOutcome,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# Incident
# ---------------------------------------------------------------------------

class TroubleshootingIncident(BaseModel):
    """
    Structured representation of a developer troubleshooting incident.

    Most fields are optional — the only hard requirement is that the incident
    carries enough information for M2 signal extraction to form a meaningful
    retrieval query (enforced there, not here).

    Version strings are kept as plain strings in M1; normalization and
    comparison logic is added in M2/M4.
    """

    description: str = Field(
        ...,
        min_length=1,
        description="Free-text description of the incident.",
    )
    current_version: str | None = Field(
        default=None,
        description="Current software/SDK/API version in use.",
    )
    previous_version: str | None = Field(
        default=None,
        description="Version in use before the change that triggered the incident.",
    )
    error_codes: list[str] = Field(
        default_factory=list,
        description="Observed error codes (e.g. AUTH_401, HTTP_429).",
    )
    product: str | None = Field(
        default=None,
        description="Product or component name (e.g. 'DevCore SDK', 'Payments API').",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary structured technical context "
            "(e.g. runtime, platform, config keys). "
            "Kept as a dict so later modules can add typed overlays."
        ),
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Corpus / Evidence
# ---------------------------------------------------------------------------

class Document(BaseModel):
    """
    A single document in the DevCore corpus.

    version_range uses the same string format as Python packaging specifiers
    (e.g. '>=3.0,<4.0') or '*' for version-agnostic documents.
    Applicability evaluation against this range is M4's responsibility.
    """

    doc_id: str = Field(..., description="Stable unique document identifier.")
    title: str = Field(..., description="Human-readable document title.")
    content: str = Field(..., description="Full document text.")
    applies_to: str = Field(
        default="*",
        description=(
            "Version specifier string (PEP 440 style) or '*' for all versions."
        ),
    )
    topic: str | None = Field(
        default=None,
        description="High-level topic tag (e.g. 'authentication', 'webhooks').",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra metadata (e.g. product, last_updated).",
    )

    model_config = {"frozen": True}


class EvidenceChunk(BaseModel):
    """
    A retrievable unit derived from a Document.

    In M2 a document may be split into multiple chunks; each chunk carries
    enough metadata to trace back to the source document.
    Chunking logic is M2's responsibility — M1 only establishes the type.
    """

    chunk_id: str = Field(..., description="Stable unique chunk identifier.")
    doc_id: str = Field(..., description="Source document ID.")
    content: str = Field(..., description="Text content of this chunk.")
    applies_to: str = Field(
        default="*",
        description="Inherited from the parent document.",
    )
    topic: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    """
    Output record from the hybrid retrieval step (M3).

    score semantics: higher = more relevant (normalised to [0, 1] by M3).
    source indicates which retrieval path surfaced this candidate.
    """

    chunk_id: str = Field(..., description="Retrieved chunk identifier.")
    doc_id: str = Field(..., description="Source document identifier.")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised relevance score in [0, 1].",
    )
    source: RetrievalSource = Field(
        ...,
        description="Which retrieval mechanism produced this result.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------

class ApplicabilityResult(BaseModel):
    """
    Deterministic applicability decision for a single evidence chunk (M4).

    applicable=True  → chunk may be used as evidence for this incident.
    applicable=False → chunk is excluded from the evidence bundle but
                       remains visible in the trace.
    """

    chunk_id: str = Field(..., description="Evaluated chunk identifier.")
    doc_id: str = Field(..., description="Source document identifier.")
    applicable: bool = Field(
        ...,
        description="True if the chunk applies to the incident's current_version.",
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation of the applicability decision.",
    )
    incident_version: str | None = Field(
        default=None,
        description="The incident version used in evaluation (for traceability).",
    )
    document_range: str | None = Field(
        default=None,
        description="The document version_range evaluated (for traceability).",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

class DiagnosisClaim(BaseModel):
    """
    A single structured claim produced by the diagnosis generator (M4).

    evidence_ids must reference chunks in the applicable evidence bundle;
    citation validity checking (cited_ids ⊆ applicable_ids) is M5's job.
    """

    claim_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique claim identifier.",
    )
    role: ClaimRole = Field(..., description="Semantic role of this claim.")
    text: str = Field(..., min_length=1, description="Claim text.")
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs cited as evidence for this claim.",
    )

    model_config = {"frozen": True}


class DiagnosisResult(BaseModel):
    """
    The complete structured diagnosis produced by the generator (M4).

    Structural invariant: exactly one claim with role ROOT_CAUSE is required.
    This is enforced here as it is a fundamental contract constraint.
    """

    claims: list[DiagnosisClaim] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _require_exactly_one_root_cause(self) -> DiagnosisResult:
        root_causes = [c for c in self.claims if c.role == ClaimRole.ROOT_CAUSE]
        if len(root_causes) != 1:
            raise ValueError(
                f"DiagnosisResult must contain exactly one root_cause claim; "
                f"found {len(root_causes)}."
            )
        return self

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class VerificationVerdict(BaseModel):
    """
    The verification outcome for a single DiagnosisClaim (M5).
    """

    claim_id: str = Field(..., description="Claim being evaluated.")
    status: VerificationStatus = Field(
        ...,
        description="PASS or FAIL for this claim.",
    )
    support: SupportDecision = Field(
        ...,
        description="Whether cited evidence supports the claim.",
    )
    contradiction: ContradictionDecision = Field(
        ...,
        description="Whether contradictory applicable evidence exists.",
    )
    reasoning: str = Field(
        default="",
        description="LLM-generated or rule-generated reasoning for the verdict.",
    )
    rejection_reason: str | None = Field(
        default=None,
        description="If FAIL, a short human-readable rejection reason.",
    )
    evidence_ids_considered: list[str] = Field(
        default_factory=list,
        description="Chunk IDs the verifier consulted.",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Final answer
# ---------------------------------------------------------------------------

class FinalAnswer(BaseModel):
    """
    The user-facing answer assembled from verified claims (M6).

    Raw LLM text is never surfaced directly; this structure is always the
    result of assembling verified claims.
    """

    outcome: SystemOutcome = Field(..., description="Top-level system outcome.")
    completeness: AnswerCompleteness | None = Field(
        default=None,
        description="Set only when outcome is ANSWERED.",
    )
    answer_text: str | None = Field(
        default=None,
        description="Human-readable answer (assembled from verified claims).",
    )
    verified_claims: list[DiagnosisClaim] = Field(
        default_factory=list,
        description="Claims that passed verification.",
    )
    needs_info_reason: str | None = Field(
        default=None,
        description="Set when outcome is NEEDS_INFO.",
    )

    @model_validator(mode="after")
    def _completeness_requires_answered(self) -> FinalAnswer:
        if self.completeness is not None and self.outcome != SystemOutcome.ANSWERED:
            raise ValueError(
                "completeness may only be set when outcome is ANSWERED."
            )
        if self.outcome == SystemOutcome.ANSWERED and self.completeness is None:
            raise ValueError(
                "completeness must be set when outcome is ANSWERED."
            )
        return self

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

class Trace(BaseModel):
    """
    Structured pipeline trace — the single audit record for one pipeline run.

    Fields are populated progressively as the pipeline advances through modules.
    All fields after `incident` are optional at construction time so that
    partial traces remain valid during execution.
    """

    trace_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique run identifier.",
    )

    # M2 populates
    incident: TroubleshootingIncident = Field(
        ..., description="The input incident (after normalization in M2)."
    )
    extracted_signals: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Deterministic signals extracted from the incident "
            "(e.g. error codes, version strings, technical terms)."
        ),
    )

    # M3 populates
    retrieved_results: list[RetrievalResult] = Field(
        default_factory=list,
        description="Raw retrieval results before applicability filtering.",
    )

    # M4 populates
    applicability_decisions: list[ApplicabilityResult] = Field(
        default_factory=list,
        description="Per-chunk applicability decisions.",
    )
    applicable_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs that passed applicability filtering.",
    )
    generated_claims: list[DiagnosisClaim] = Field(
        default_factory=list,
        description="Raw claims from the diagnosis generator (pre-verification).",
    )

    # M5 populates
    verification_verdicts: list[VerificationVerdict] = Field(
        default_factory=list,
        description="Per-claim verification verdicts.",
    )
    retry_attempted: bool = Field(
        default=False,
        description="True if the one-shot retry was triggered.",
    )
    retry_claims: list[DiagnosisClaim] = Field(
        default_factory=list,
        description="Claims from the retry generation (if retry was triggered).",
    )
    retry_verdicts: list[VerificationVerdict] = Field(
        default_factory=list,
        description="Verification verdicts for retry claims.",
    )

    # M6 populates
    final_answer: FinalAnswer | None = Field(
        default=None,
        description="The assembled, verified final answer.",
    )
    system_errors: list[str] = Field(
        default_factory=list,
        description="Non-fatal errors collected during the pipeline run.",
    )
