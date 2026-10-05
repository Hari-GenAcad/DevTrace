"""
DevTrace — Module 4: Baseline end-to-end pipeline.

This is the orchestrator that connects M2 → M3 → M4 into the first
working DevTrace troubleshooting pipeline.

Pipeline steps
--------------
1. Normalize incident  (M2)
2. Retrieve candidates  (M3 HybridRetriever)
3. NEEDS_INFO gate  (M4 deterministic check)
4. Applicability filtering  (M4 checker)
5. Diagnosis generation  (M4 DiagnosisGenerator + Gemini)

Output
------
BaselinePipelineResult — a transparent record of everything that happened,
including retrieval results, applicability decisions, and the raw diagnosis.

The generated diagnosis is NOT trusted yet.
M5 will verify whether cited evidence actually supports each claim.

Constraints
-----------
- Does NOT implement evidence verification (M5).
- Does NOT implement retry (M6).
- Does NOT implement final outcome assembly (M6).
- Does NOT build the trace (M6 assembles the final Trace).
- Keeps orchestration simple: linear function, no graph, no agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.applicability.checker import filter_applicable
from src.diagnosis.generator import DiagnosisGenerator
from src.diagnosis.needs_info import NeedsInfoResult, needs_info_check
from src.errors import LLMError, SchemaValidationError
from src.llm.base import LLMClient
from src.models.contracts import (
    ApplicabilityResult,
    DiagnosisResult,
    RetrievalResult,
)
from src.models.enums import SystemOutcome
from src.normalization.normalizer import NormalizedIncident, normalize_incident
from src.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class BaselinePipelineResult:
    """
    Complete result of a baseline pipeline run.

    All intermediate outputs are preserved for traceability, auditing, and
    later consumption by M5 (verification) and M6 (trace assembly).

    Attributes:
        normalized:            Normalized incident from M2.
        retrieval_results:     Ranked M3 candidates (pre-applicability).
        needs_info:            NEEDS_INFO gate result.
        applicability_decisions: Per-chunk applicability decisions.
        applicable_results:    Chunks that passed applicability filtering.
        diagnosis:             Structured diagnosis from Gemini (may be None).
        outcome:               Top-level pipeline outcome.
        error:                 Error message if a system failure occurred (may be None).
    """

    normalized: NormalizedIncident
    retrieval_results: list[RetrievalResult] = field(default_factory=list)
    needs_info: NeedsInfoResult | None = None
    applicability_decisions: list[ApplicabilityResult] = field(default_factory=list)
    applicable_results: list[RetrievalResult] = field(default_factory=list)
    diagnosis: DiagnosisResult | None = None
    outcome: SystemOutcome | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_baseline_diagnosis(
    *,
    description: str,
    current_version: str | None = None,
    previous_version: str | None = None,
    error_codes: list[str] | None = None,
    product: str | None = None,
    context: dict[str, Any] | None = None,
    retriever: HybridRetriever,
    llm_client: LLMClient,
) -> BaselinePipelineResult:
    """
    Run the full baseline DevTrace diagnosis pipeline.

    Args:
        description:      Free-text incident description (required).
        current_version:  Explicit current SDK/API version (optional).
        previous_version: Explicit previous version (optional).
        error_codes:      Explicit error codes (optional; merged with extracted).
        product:          Product/component name (optional).
        context:          Arbitrary structured context dict (optional).
        retriever:        A loaded HybridRetriever instance (M3).
                          The caller is responsible for loading the retriever.
        llm_client:       An LLMClient implementation (GeminiClient in production,
                          FakeLLMClient in tests).

    Returns:
        BaselinePipelineResult — see the dataclass for field documentation.

    Notes:
        - NEEDS_INFO is returned early (before diagnosis).
        - LLM failures produce outcome=DEGRADED and set error on the result.
        - No retry is attempted here (that is M6's responsibility).
    """
    # ------------------------------------------------------------------
    # Step 1: Normalize
    # ------------------------------------------------------------------
    logger.info("Step 1: Normalizing incident.")
    normalized = normalize_incident(
        description,
        current_version=current_version,
        previous_version=previous_version,
        error_codes=error_codes,
        product=product,
        context=context,
    )

    result = BaselinePipelineResult(normalized=normalized)

    # ------------------------------------------------------------------
    # Step 2: Retrieve
    # ------------------------------------------------------------------
    logger.info("Step 2: Retrieving candidates from M3 HybridRetriever.")
    try:
        retrieval_results: list[RetrievalResult] = retriever.retrieve_as_contracts(
            normalized
        )
    except Exception as exc:
        logger.exception("Retrieval failed.")
        result.outcome = SystemOutcome.DEGRADED
        result.error = f"Retrieval failed: {type(exc).__name__}: {exc}"
        return result

    result.retrieval_results = retrieval_results
    logger.info("Retrieved %d candidates.", len(retrieval_results))

    # ------------------------------------------------------------------
    # Step 3: NEEDS_INFO gate
    # ------------------------------------------------------------------
    logger.info("Step 3: Running NEEDS_INFO gate.")
    needs_info = needs_info_check(normalized, retrieval_results)
    result.needs_info = needs_info

    if needs_info.triggered:
        logger.info("NEEDS_INFO triggered: %s", needs_info.reason)
        result.outcome = SystemOutcome.NEEDS_INFO
        return result

    # ------------------------------------------------------------------
    # Step 4: Applicability filtering
    # ------------------------------------------------------------------
    logger.info("Step 4: Filtering by applicability.")
    applicable_results, applicability_decisions = filter_applicable(
        retrieval_results,
        current_version=normalized.incident.current_version,
    )
    result.applicability_decisions = applicability_decisions
    result.applicable_results = applicable_results

    applicable_count = len(applicable_results)
    not_applicable_count = len(retrieval_results) - applicable_count
    logger.info(
        "Applicability: %d applicable, %d excluded.",
        applicable_count,
        not_applicable_count,
    )

    # ------------------------------------------------------------------
    # Step 5: Diagnosis generation
    # ------------------------------------------------------------------
    logger.info("Step 5: Generating diagnosis via LLM.")
    generator = DiagnosisGenerator(llm_client)

    try:
        diagnosis = generator.generate(normalized, applicable_results)
    except (LLMError, SchemaValidationError) as exc:
        logger.error("Diagnosis generation failed: %s", exc)
        result.outcome = SystemOutcome.DEGRADED
        result.error = str(exc)
        return result

    result.diagnosis = diagnosis
    # M4 produces a diagnosis — its trustworthiness is for M5 to determine.
    # We leave outcome=None here; M6 will set the final SystemOutcome.
    logger.info(
        "Diagnosis generated: %d claim(s).", len(diagnosis.claims)
    )

    return result
