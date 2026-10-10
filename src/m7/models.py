"""
DevTrace — Module 7: Evaluation data models.

All Pydantic models and dataclasses representing per-case results,
aggregate metrics, baseline outputs, and the final evaluation report.

These models are the schema contract between:
  - the runner (which populates them)
  - the metrics layer (which reads from them)
  - the report writer (which serialises them)

Design principles:
  - No production logic here. Pure data containers.
  - All fields optional where the system under evaluation may not produce them.
  - Machine-readable: JSON-serialisable via Pydantic .model_dump().
  - Gold labels are stored separately from system outputs to prevent leakage.
"""

from __future__ import annotations

from enum import Enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# System identifiers
# ---------------------------------------------------------------------------

class SystemLabel(str, Enum):
    """Label for the system that produced a result."""

    BASELINE_A = "BASELINE_A"
    """Naive RAG: retrieve → generate, no applicability or verification."""

    BASELINE_B = "BASELINE_B"
    """Retrieve + score threshold: generate only if top result exceeds threshold."""

    DEVTRACE = "DEVTRACE"
    """Full M6 orchestration: retrieve → applicability → diagnose → verify → survive → retry."""


# ---------------------------------------------------------------------------
# Baseline A result
# ---------------------------------------------------------------------------

class BaselineAResult(BaseModel):
    """
    Result from Baseline A (Naive RAG).

    Retrieve all candidates, pass them to the LLM with no applicability
    filtering, no verification, and no retry. Present the raw LLM response.

    Citation structure is NOT enforced by Baseline A (plain text answer).
    The cited_doc_ids field is populated by simple heuristic extraction
    of doc IDs mentioned in the response (best-effort, for metric logging only).
    """

    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs returned by retrieval (pre-applicability).",
    )
    retrieved_doc_ids: list[str] = Field(
        default_factory=list,
        description="Deduplicated doc IDs from retrieved chunks.",
    )
    answer_text: str | None = Field(
        default=None,
        description="Raw LLM-generated answer (no verification applied).",
    )
    cited_doc_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Doc IDs heuristically extracted from answer_text for metric logging. "
            "Not structured citations — best-effort only."
        ),
    )
    abstained: bool = Field(
        default=False,
        description="True if no answer was produced (e.g. retrieval returned nothing).",
    )
    error: str | None = Field(
        default=None,
        description="System error message if the baseline failed to run.",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Baseline B result
# ---------------------------------------------------------------------------

class BaselineBResult(BaseModel):
    """
    Result from Baseline B (Retrieve + Score Threshold).

    Retrieve candidates; if the top candidate score exceeds the configured
    threshold, generate an answer. Otherwise abstain.

    No applicability filtering, no verification, no retry.
    """

    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs returned by retrieval (pre-filtering).",
    )
    retrieved_doc_ids: list[str] = Field(
        default_factory=list,
        description="Deduplicated doc IDs from retrieved chunks.",
    )
    top_score: float | None = Field(
        default=None,
        description="The highest retrieval score among retrieved candidates.",
    )
    threshold_used: float = Field(
        description="The score threshold that was configured for this run.",
    )
    threshold_passed: bool = Field(
        description="True if top_score >= threshold_used.",
    )
    answer_text: str | None = Field(
        default=None,
        description="Raw LLM-generated answer (None if abstained).",
    )
    cited_doc_ids: list[str] = Field(
        default_factory=list,
        description="Doc IDs heuristically extracted from answer_text.",
    )
    abstained: bool = Field(
        description="True if the system abstained (threshold not passed or retrieval empty).",
    )
    error: str | None = Field(
        default=None,
        description="System error if the baseline failed to run.",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# DevTrace result (wraps TroubleshootingResult, adds eval metadata)
# ---------------------------------------------------------------------------

class DevTraceEvalResult(BaseModel):
    """
    DevTrace result for a single evaluation case.

    Wraps the M6 TroubleshootingResult (serialised as a dict) with
    additional extracted fields needed by the metric layer.
    """

    final_outcome: str = Field(
        description="M6 FinalOutcome enum value (e.g. ANSWERED_FULL, NEEDS_INFO).",
    )
    answer_text: str | None = Field(
        default=None,
        description="Assembled verified answer text (None if not answered).",
    )
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
        description="All chunks returned by M3 retrieval.",
    )
    retrieved_doc_ids: list[str] = Field(
        default_factory=list,
        description="Deduplicated doc IDs from all retrieved chunks.",
    )
    applicable_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunks that passed M4 applicability filtering.",
    )
    applicable_doc_ids: list[str] = Field(
        default_factory=list,
        description="Deduplicated doc IDs from applicable chunks.",
    )
    cited_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs cited in verified claims (from initial or retry diagnosis).",
    )
    cited_doc_ids: list[str] = Field(
        default_factory=list,
        description="Deduplicated doc IDs from cited chunk IDs.",
    )
    forbidden_doc_ids_cited: list[str] = Field(
        default_factory=list,
        description="Forbidden doc IDs that appeared in retrieved results.",
    )
    retry_attempted: bool = Field(
        default=False,
        description="True if M6 triggered a retry.",
    )
    retry_succeeded: bool = Field(
        default=False,
        description="True if the retry produced a verified root cause.",
    )
    abstained: bool = Field(
        description=(
            "True if the system produced NEEDS_INFO, INSUFFICIENT_EVIDENCE, or DEGRADED "
            "(i.e. no final answer was produced)."
        ),
    )
    full_result: dict[str, Any] = Field(
        default_factory=dict,
        description="Serialised TroubleshootingResult for full traceability.",
    )
    error: str | None = Field(
        default=None,
        description="System error if the DevTrace pipeline failed to run.",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Per-metric results (system-level for a single case)
# ---------------------------------------------------------------------------

class CaseMetrics(BaseModel):
    """
    Computed metrics for one (system, case) pair.

    Boolean fields model a binary score per case; None means the metric
    is not applicable (e.g. citation_correct is N/A when no citations exist).
    """

    retrieval_hit: bool | None = Field(
        default=None,
        description="True if at least one gold_doc_id was in retrieved_doc_ids.",
    )
    forbidden_retrieved: bool | None = Field(
        default=None,
        description="True if at least one forbidden_doc_id was retrieved.",
    )
    citation_valid: bool | None = Field(
        default=None,
        description="True if all cited IDs are within the retrieved/applicable set.",
    )
    citation_correct: bool | None = Field(
        default=None,
        description="True if at least one gold_doc_id appears in cited_doc_ids.",
    )
    false_answer: bool | None = Field(
        default=None,
        description=(
            "True if the system gave a confident answer for a case where gold says "
            "INSUFFICIENT_EVIDENCE or NEEDS_INFO, OR cited forbidden evidence."
        ),
    )
    false_abstention: bool | None = Field(
        default=None,
        description=(
            "True if the system abstained when gold expected an ANSWERED outcome."
        ),
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Per-case evaluation record
# ---------------------------------------------------------------------------

class EvaluationCaseRecord(BaseModel):
    """
    Complete evaluation record for one case across all three systems.

    This is the machine-readable per-case output. One record per eval case.
    """

    # Identification
    case_id: str = Field(description="Evaluation case identifier.")
    case_class: str = Field(description="Case category (e.g. version_conflict).")

    # Gold labels (NOT passed to systems; used for scoring only)
    gold_expected_outcome: str = Field(description="Expected outcome from dataset.")
    gold_expected_completeness: str | None = Field(default=None)
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_forbidden_doc_ids: list[str] = Field(default_factory=list)

    # System outputs
    baseline_a: BaselineAResult | None = Field(default=None)
    baseline_b: BaselineBResult | None = Field(default=None)
    devtrace: DevTraceEvalResult | None = Field(default=None)

    # Per-system metrics
    metrics_a: CaseMetrics | None = Field(default=None)
    metrics_b: CaseMetrics | None = Field(default=None)
    metrics_devtrace: CaseMetrics | None = Field(default=None)

    # Notes
    notes: str = Field(default="", description="Evaluation case notes from dataset.")

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Aggregate metric summary
# ---------------------------------------------------------------------------

class AggregateMetrics(BaseModel):
    """Aggregate metrics over all cases for a single system."""

    total_cases: int = Field(description="Number of evaluation cases.")
    retrieval_hit_rate: float | None = Field(
        default=None,
        description="Fraction of cases where ≥1 gold doc was retrieved.",
    )
    forbidden_retrieval_rate: float | None = Field(
        default=None,
        description="Fraction of cases where ≥1 forbidden doc was retrieved.",
    )
    citation_validity_rate: float | None = Field(
        default=None,
        description="Fraction of cases where all citations were valid.",
    )
    citation_correctness_rate: float | None = Field(
        default=None,
        description="Fraction of answered cases where ≥1 gold doc was cited.",
    )
    false_answer_rate: float | None = Field(
        default=None,
        description="Fraction of cases with false confident answers.",
    )
    false_abstention_rate: float | None = Field(
        default=None,
        description="Fraction of expected-answered cases where system abstained.",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Outcome distribution (DevTrace only)
# ---------------------------------------------------------------------------

class OutcomeDistribution(BaseModel):
    """Distribution of M6 FinalOutcome values across all evaluation cases."""

    answered_full: int = 0
    answered_partial: int = 0
    insufficient_evidence: int = 0
    needs_info: int = 0
    degraded: int = 0
    total: int = 0

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Retry statistics (DevTrace only)
# ---------------------------------------------------------------------------

class RetryStats(BaseModel):
    """Statistics about M6 retry behaviour across all evaluation cases."""

    initial_root_cause_failures: int = Field(
        default=0,
        description="Cases where the initial root cause was NOT verified.",
    )
    retries_attempted: int = Field(
        default=0,
        description="Cases where the retry was triggered.",
    )
    retries_succeeded: int = Field(
        default=0,
        description="Retries that produced a verified root cause.",
    )
    retries_failed: int = Field(
        default=0,
        description="Retries that still failed to produce a verified root cause.",
    )
    retry_recovery_rate: float | None = Field(
        default=None,
        description="retries_succeeded / retries_attempted (None if no retries).",
    )
    max_retry_count_exceeded: bool = Field(
        default=False,
        description="True if any case attempted more than 1 retry (invariant violation).",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Category-level metrics
# ---------------------------------------------------------------------------

class CategoryMetrics(BaseModel):
    """Metrics aggregated for a specific case_class category."""

    category: str = Field(description="Case class name.")
    case_count: int = Field(description="Number of cases in this category.")
    metrics_a: AggregateMetrics | None = Field(default=None)
    metrics_b: AggregateMetrics | None = Field(default=None)
    metrics_devtrace: AggregateMetrics | None = Field(default=None)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Version-conflict analysis
# ---------------------------------------------------------------------------

class VersionConflictAnalysis(BaseModel):
    """
    Dedicated analysis of version_conflict cases.

    Measures whether systems use wrong-version evidence confidently.
    """

    total_version_conflict_cases: int
    # How often each system retrieved a forbidden (wrong-version) doc
    baseline_a_forbidden_retrieved_rate: float | None = None
    baseline_b_forbidden_retrieved_rate: float | None = None
    devtrace_forbidden_retrieved_rate: float | None = None
    # How often each system produced a confident answer using wrong-version docs
    baseline_a_false_answer_rate: float | None = None
    baseline_b_false_answer_rate: float | None = None
    devtrace_false_answer_rate: float | None = None
    # How often DevTrace correctly excluded forbidden docs via applicability
    devtrace_correct_exclusion_rate: float | None = None

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Unsupported-case analysis
# ---------------------------------------------------------------------------

class UnsupportedCaseAnalysis(BaseModel):
    """Analysis of unsupported, near_miss, and missing_info cases."""

    total_abstention_expected_cases: int = Field(
        description="Cases where expected_outcome != ANSWERED.",
    )
    baseline_a_false_answer_rate: float | None = None
    baseline_b_false_answer_rate: float | None = None
    devtrace_false_answer_rate: float | None = None
    devtrace_correct_abstention_rate: float | None = None

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Final evaluation report
# ---------------------------------------------------------------------------

class EvaluationReport(BaseModel):
    """
    The complete M7 evaluation report.

    Contains per-case records, aggregate metrics for all three systems,
    category-level breakdown, and dedicated analyses.
    """

    # Metadata
    dataset_path: str = Field(description="Path to the evaluation dataset.")
    total_cases: int = Field(description="Total number of evaluation cases run.")
    evaluation_mode: str = Field(
        description="'deterministic' (FakeLLM) or 'live' (real LLM).",
    )
    model_config_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Model/configuration details for reproducibility.",
    )
    baseline_b_threshold: float = Field(
        description="Score threshold used for Baseline B.",
    )
    generated_at_utc: datetime | None = Field(
        default=None,
        description="UTC timestamp when this report was generated.",
    )
    git_commit: str | None = Field(
        default=None,
        description="Source commit evaluated, when Git metadata is available.",
    )
    python_version: str | None = Field(
        default=None,
        description="Python runtime version used for the evaluation.",
    )

    # Per-case records (machine-readable)
    case_records: list[EvaluationCaseRecord] = Field(default_factory=list)

    # Aggregate metrics
    aggregate_a: AggregateMetrics | None = Field(default=None)
    aggregate_b: AggregateMetrics | None = Field(default=None)
    aggregate_devtrace: AggregateMetrics | None = Field(default=None)

    # Outcome distribution (DevTrace only)
    outcome_distribution: OutcomeDistribution | None = Field(default=None)

    # Retry analysis (DevTrace only)
    retry_stats: RetryStats | None = Field(default=None)

    # Category-level breakdown
    category_metrics: list[CategoryMetrics] = Field(default_factory=list)

    # Specialised analyses
    version_conflict_analysis: VersionConflictAnalysis | None = Field(default=None)
    unsupported_case_analysis: UnsupportedCaseAnalysis | None = Field(default=None)

    # Summary (human-readable answers to the key questions)
    summary: dict[str, str] = Field(
        default_factory=dict,
        description="Human-readable answers to the DevTrace thesis questions.",
    )

    model_config = {"frozen": False}  # Allow post-construction field updates
