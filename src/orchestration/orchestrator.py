"""
DevTrace — Module 6: Main orchestration entry point.

This module wires together M4 (baseline pipeline), M5 (evidence verification),
and M6's own retry and outcome-classification logic into a single reliable
troubleshooting call.

Pipeline flow
-------------
1.  Run M4+M5 combined pipeline (run_verified_diagnosis).
2.  If NEEDS_INFO → preserve NEEDS_INFO, return immediately.
3.  If DEGRADED → preserve DEGRADED, return immediately.
4.  If M5 verification system failure → DEGRADED, return immediately.
5.  Inspect initial verification for root-cause survival.
6.  If root_cause VERIFIED → classify final outcome, assemble answer, return.
7.  If root_cause NOT VERIFIED:
      a. Check whether retry is allowed (has applicable evidence).
      b. If retry NOT allowed → INSUFFICIENT_EVIDENCE.
      c. If retry allowed → run ONE targeted retry.
8.  Verify retry diagnosis.
9.  If retry root_cause VERIFIED → classify final outcome from retry.
10. If retry root_cause still NOT VERIFIED → INSUFFICIENT_EVIDENCE.
11. Assemble final answer using only verified claims.

Key invariants enforced here
-----------------------------
- MAX ONE RETRY. No loop. Tracked by retry_attempted flag.
- Retry uses SAME applicable evidence. No new retrieval.
- Final answer contains ONLY verified claims.
- System failure (LLM/schema error) → DEGRADED, never INSUFFICIENT_EVIDENCE.
- Claim-quality failure → INSUFFICIENT_EVIDENCE, never DEGRADED.

Modules NOT touched by M6
--------------------------
- Retrieval (M3 HybridRetriever): unchanged.
- Applicability (M4 filter_applicable): unchanged.
- Baseline diagnosis (M4 DiagnosisGenerator): reused for retry via targeted prompt.
- Evidence verification (M5 verify_diagnosis): reused for retry verification.
- FakeLLMClient (M1): unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from src.diagnosis.generator import parse_diagnosis
from src.errors import LLMError, SchemaValidationError
from src.llm.base import LLMClient
from src.models.contracts import DiagnosisResult, RetrievalResult
from src.models.enums import SystemOutcome
from src.normalization.normalizer import NormalizedIncident
from src.orchestration.assembly import (
    build_answered_answer,
    build_degraded_answer,
    build_insufficient_answer,
    build_needs_info_answer,
    collect_verified_claims,
    get_root_cause_rejection_reason,
    root_cause_survived,
)
from src.orchestration.models import FinalOutcome, TroubleshootingResult, VerifiedAnswer
from src.orchestration.retry_prompt import build_retry_prompt
from src.pipeline.verified import run_verified_diagnosis
from src.retrieval.hybrid import HybridRetriever
from src.verification.models import DiagnosisVerification
from src.verification.verifier import verify_diagnosis

logger = logging.getLogger(__name__)

# Maximum number of retries.  Spec is explicit: exactly one.
_MAX_RETRIES: int = 1


# ---------------------------------------------------------------------------
# Retry guard
# ---------------------------------------------------------------------------

def _retry_is_allowed(
    applicable_results: list[RetrievalResult],
) -> bool:
    """
    Return True when a retry is permitted.

    Retry requires applicable evidence to diagnose against.
    If there is nothing applicable, a retry cannot produce a better result.

    Args:
        applicable_results: The M4 applicable evidence bundle.

    Returns:
        True if retry should be attempted.
    """
    return len(applicable_results) > 0


# ---------------------------------------------------------------------------
# Targeted retry: one LLM call + one M5 verification
# ---------------------------------------------------------------------------

def _run_targeted_retry(
    *,
    normalized: NormalizedIncident,
    applicable_results: list[RetrievalResult],
    first_diagnosis: DiagnosisResult,
    first_verification: DiagnosisVerification,
    llm_client: LLMClient,
    verifier_llm_client: LLMClient,
) -> tuple[DiagnosisResult | None, DiagnosisVerification | None, str | None]:
    """
    Run one targeted retry: new diagnosis + new M5 verification.

    Args:
        normalized:           Normalized incident from M2.
        applicable_results:   Same applicable evidence as first attempt.
        first_diagnosis:      Initial DiagnosisResult (for feedback).
        first_verification:   Initial verification (failure context).
        llm_client:           LLM client for retry diagnosis generation.
        verifier_llm_client:  LLM client for retry verification (M5).

    Returns:
        Tuple of (retry_diagnosis, retry_verification, error_message).
        error_message is None on success, or a string on system failure.
    """
    incident = normalized.incident
    signals = normalized.signals

    # Build targeted retry prompt with verification feedback.
    retry_prompt = build_retry_prompt(
        incident_description=incident.description,
        current_version=incident.current_version,
        previous_version=incident.previous_version,
        error_codes=signals.error_codes,
        applicable_results=applicable_results,
        first_diagnosis=first_diagnosis,
        first_verification=first_verification,
    )

    # --- Retry diagnosis generation ---
    logger.info("Running targeted retry diagnosis generation.")
    try:
        response = llm_client.generate(retry_prompt)
    except LLMError as exc:
        logger.error("Retry diagnosis LLM failed: %s", exc)
        return None, None, f"Retry LLM call failed: {exc}"
    except Exception as exc:
        logger.error("Retry diagnosis unexpected error: %s", exc)
        return None, None, f"Retry unexpected error: {type(exc).__name__}: {exc}"

    if not response.success or not response.text:
        return None, None, "Retry LLM returned empty/unsuccessful response."

    try:
        retry_diagnosis = parse_diagnosis(response.text)
    except SchemaValidationError as exc:
        logger.error("Retry diagnosis schema validation failed: %s", exc)
        return None, None, f"Retry diagnosis schema error: {exc}"

    logger.info("Retry diagnosis produced %d claim(s).", len(retry_diagnosis.claims))

    # --- Retry verification (M5) ---
    logger.info("Running M5 verification on retry diagnosis.")
    try:
        retry_verification = verify_diagnosis(
            diagnosis=retry_diagnosis,
            applicable_results=applicable_results,
            normalized=normalized,
            llm_client=verifier_llm_client,
        )
    except (LLMError, SchemaValidationError) as exc:
        logger.error("Retry verification failed: %s", exc)
        return retry_diagnosis, None, f"Retry verification system error: {exc}"

    logger.info(
        "Retry verification complete: %d/%d claims verified.",
        len(retry_verification.verified_claim_ids),
        len(retry_verification.claim_verifications),
    )
    return retry_diagnosis, retry_verification, None


# ---------------------------------------------------------------------------
# Main orchestration entry point
# ---------------------------------------------------------------------------

def run_troubleshooting(
    *,
    description: str,
    current_version: str | None = None,
    previous_version: str | None = None,
    error_codes: list[str] | None = None,
    product: str | None = None,
    context: dict[str, Any] | None = None,
    retriever: HybridRetriever,
    llm_client: LLMClient,
    verifier_llm_client: LLMClient | None = None,
) -> TroubleshootingResult:
    """
    Run the complete DevTrace troubleshooting pipeline (M6 orchestration).

    This is the main end-to-end entry point for DevTrace. It coordinates:
      - M4: normalize → retrieve → applicability → diagnosis
      - M5: evidence verification
      - M6: root-cause survival check → optional targeted retry → outcome classification

    Args:
        description:          Free-text incident description (required).
        current_version:      Explicit current SDK/API version (optional).
        previous_version:     Explicit previous version (optional).
        error_codes:          Explicit error codes (optional).
        product:              Product/component name (optional).
        context:              Arbitrary structured context dict (optional).
        retriever:            A loaded HybridRetriever instance (M3).
        llm_client:           LLM client for diagnosis generation.
        verifier_llm_client:  Separate LLM client for evidence verification (M5).
                              If None, llm_client is used for both (called separately).

    Returns:
        TroubleshootingResult — complete record of the full pipeline run.

    Notes:
        - NEEDS_INFO exits before diagnosis and returns immediately.
        - DEGRADED is returned on system failures (not on diagnosis-quality failures).
        - Retry is attempted at most once when root_cause fails verification.
        - Final answer contains only verified claims.
    """
    verifier_client = verifier_llm_client if verifier_llm_client is not None else llm_client
    system_errors: list[str] = []

    # -----------------------------------------------------------------------
    # Step 1: Run M4 + M5 combined pipeline
    # -----------------------------------------------------------------------
    logger.info("M6: Running M4+M5 combined pipeline.")
    verified_result = run_verified_diagnosis(
        description=description,
        current_version=current_version,
        previous_version=previous_version,
        error_codes=error_codes,
        product=product,
        context=context,
        retriever=retriever,
        llm_client=llm_client,
        verifier_llm_client=verifier_client,
    )

    baseline = verified_result.baseline
    normalized = verified_result.normalized

    # Convenience references to M4 outputs
    retrieval_results = baseline.retrieval_results
    applicability_decisions = baseline.applicability_decisions
    applicable_results = baseline.applicable_results

    # -----------------------------------------------------------------------
    # Step 2: NEEDS_INFO early exit
    # -----------------------------------------------------------------------
    if baseline.outcome == SystemOutcome.NEEDS_INFO:
        logger.info("M6: NEEDS_INFO — returning early without diagnosis.")
        needs_info_reason = (
            baseline.needs_info.reason if baseline.needs_info else None
        )
        final_answer = build_needs_info_answer(needs_info_reason)
        return TroubleshootingResult(
            incident_description=description,
            current_version=current_version,
            retrieval_results=retrieval_results,
            applicability_decisions=applicability_decisions,
            applicable_results=applicable_results,
            final_answer=final_answer,
            final_outcome=FinalOutcome.NEEDS_INFO,
            system_errors=system_errors,
        )

    # -----------------------------------------------------------------------
    # Step 3: DEGRADED early exit (M4 system failure)
    # -----------------------------------------------------------------------
    if baseline.outcome == SystemOutcome.DEGRADED:
        logger.error("M6: M4 pipeline DEGRADED: %s", baseline.error)
        final_answer = build_degraded_answer(baseline.error)
        if baseline.error:
            system_errors.append(baseline.error)
        return TroubleshootingResult(
            incident_description=description,
            current_version=current_version,
            retrieval_results=retrieval_results,
            applicability_decisions=applicability_decisions,
            applicable_results=applicable_results,
            initial_diagnosis=baseline.diagnosis,
            final_answer=final_answer,
            final_outcome=FinalOutcome.DEGRADED,
            system_errors=system_errors,
        )

    # -----------------------------------------------------------------------
    # Step 4: M5 verification system failure → DEGRADED
    # -----------------------------------------------------------------------
    if verified_result.verification_error is not None:
        logger.error(
            "M6: M5 verification system failure: %s", verified_result.verification_error
        )
        system_errors.append(verified_result.verification_error)
        final_answer = build_degraded_answer(verified_result.verification_error)
        return TroubleshootingResult(
            incident_description=description,
            current_version=current_version,
            retrieval_results=retrieval_results,
            applicability_decisions=applicability_decisions,
            applicable_results=applicable_results,
            initial_diagnosis=baseline.diagnosis,
            final_answer=final_answer,
            final_outcome=FinalOutcome.DEGRADED,
            system_errors=system_errors,
        )

    # -----------------------------------------------------------------------
    # At this point we have a diagnosis AND a verification result.
    # -----------------------------------------------------------------------
    initial_diagnosis = baseline.diagnosis
    initial_verification = verified_result.verification

    # Defensive guard (these cannot be None here, but make the type checker happy)
    if initial_diagnosis is None or initial_verification is None:
        logger.error("M6: Unexpected state — diagnosis or verification is None.")
        final_answer = build_degraded_answer(
            "Unexpected pipeline state: diagnosis or verification missing."
        )
        return TroubleshootingResult(
            incident_description=description,
            current_version=current_version,
            retrieval_results=retrieval_results,
            applicability_decisions=applicability_decisions,
            applicable_results=applicable_results,
            final_answer=final_answer,
            final_outcome=FinalOutcome.DEGRADED,
            system_errors=system_errors,
        )

    # -----------------------------------------------------------------------
    # Step 5: Root-cause survival check on initial diagnosis
    # -----------------------------------------------------------------------
    logger.info("M6: Checking initial root-cause survival.")

    if root_cause_survived(initial_diagnosis, initial_verification):
        logger.info("M6: Initial root cause VERIFIED — classifying outcome.")
        final_answer = build_answered_answer(initial_diagnosis, initial_verification)
        return TroubleshootingResult(
            incident_description=description,
            current_version=current_version,
            retrieval_results=retrieval_results,
            applicability_decisions=applicability_decisions,
            applicable_results=applicable_results,
            initial_diagnosis=initial_diagnosis,
            initial_verification=initial_verification,
            final_answer=final_answer,
            final_outcome=final_answer.outcome,
            system_errors=system_errors,
        )

    # -----------------------------------------------------------------------
    # Step 6: Root cause did NOT survive — decide on retry
    # -----------------------------------------------------------------------
    retry_reason = get_root_cause_rejection_reason(initial_diagnosis, initial_verification)
    logger.info("M6: Initial root cause REJECTED. Reason: %s", retry_reason)

    if not _retry_is_allowed(applicable_results):
        logger.info("M6: Retry not allowed (no applicable evidence). INSUFFICIENT_EVIDENCE.")
        final_answer = build_insufficient_answer()
        return TroubleshootingResult(
            incident_description=description,
            current_version=current_version,
            retrieval_results=retrieval_results,
            applicability_decisions=applicability_decisions,
            applicable_results=applicable_results,
            initial_diagnosis=initial_diagnosis,
            initial_verification=initial_verification,
            retry_attempted=False,
            retry_reason=retry_reason,
            final_answer=final_answer,
            final_outcome=FinalOutcome.INSUFFICIENT_EVIDENCE,
            system_errors=system_errors,
        )

    # -----------------------------------------------------------------------
    # Step 7: ONE targeted retry
    # -----------------------------------------------------------------------
    logger.info("M6: Attempting one targeted retry.")

    retry_diagnosis, retry_verification, retry_error = _run_targeted_retry(
        normalized=normalized,
        applicable_results=applicable_results,
        first_diagnosis=initial_diagnosis,
        first_verification=initial_verification,
        llm_client=llm_client,
        verifier_llm_client=verifier_client,
    )

    # Retry system failure → DEGRADED (not INSUFFICIENT_EVIDENCE)
    if retry_error is not None:
        logger.error("M6: Retry system error: %s", retry_error)
        system_errors.append(retry_error)
        final_answer = build_degraded_answer(retry_error)
        return TroubleshootingResult(
            incident_description=description,
            current_version=current_version,
            retrieval_results=retrieval_results,
            applicability_decisions=applicability_decisions,
            applicable_results=applicable_results,
            initial_diagnosis=initial_diagnosis,
            initial_verification=initial_verification,
            retry_attempted=True,
            retry_reason=retry_reason,
            retry_diagnosis=retry_diagnosis,
            retry_verification=retry_verification,
            final_answer=final_answer,
            final_outcome=FinalOutcome.DEGRADED,
            system_errors=system_errors,
        )

    # Retry verification produced no verification → DEGRADED
    if retry_diagnosis is None or retry_verification is None:
        logger.error("M6: Retry produced no diagnosis/verification — DEGRADED.")
        err = "Retry produced no diagnosis or verification."
        system_errors.append(err)
        final_answer = build_degraded_answer(err)
        return TroubleshootingResult(
            incident_description=description,
            current_version=current_version,
            retrieval_results=retrieval_results,
            applicability_decisions=applicability_decisions,
            applicable_results=applicable_results,
            initial_diagnosis=initial_diagnosis,
            initial_verification=initial_verification,
            retry_attempted=True,
            retry_reason=retry_reason,
            retry_diagnosis=retry_diagnosis,
            retry_verification=retry_verification,
            final_answer=final_answer,
            final_outcome=FinalOutcome.DEGRADED,
            system_errors=system_errors,
        )

    # -----------------------------------------------------------------------
    # Step 8: Root-cause survival check on retry diagnosis
    # -----------------------------------------------------------------------
    logger.info("M6: Checking retry root-cause survival.")

    if root_cause_survived(retry_diagnosis, retry_verification):
        logger.info("M6: Retry root cause VERIFIED — classifying outcome from retry.")
        final_answer = build_answered_answer(retry_diagnosis, retry_verification)
        return TroubleshootingResult(
            incident_description=description,
            current_version=current_version,
            retrieval_results=retrieval_results,
            applicability_decisions=applicability_decisions,
            applicable_results=applicable_results,
            initial_diagnosis=initial_diagnosis,
            initial_verification=initial_verification,
            retry_attempted=True,
            retry_reason=retry_reason,
            retry_diagnosis=retry_diagnosis,
            retry_verification=retry_verification,
            final_answer=final_answer,
            final_outcome=final_answer.outcome,
            system_errors=system_errors,
        )

    # -----------------------------------------------------------------------
    # Step 9: Retry root cause also failed → INSUFFICIENT_EVIDENCE
    # STOP. Do NOT retry again.
    # -----------------------------------------------------------------------
    logger.info(
        "M6: Retry root cause also REJECTED. INSUFFICIENT_EVIDENCE. "
        "(Retry was exhausted: max_retries=%d)", _MAX_RETRIES
    )
    final_answer = build_insufficient_answer()
    return TroubleshootingResult(
        incident_description=description,
        current_version=current_version,
        retrieval_results=retrieval_results,
        applicability_decisions=applicability_decisions,
        applicable_results=applicable_results,
        initial_diagnosis=initial_diagnosis,
        initial_verification=initial_verification,
        retry_attempted=True,
        retry_reason=retry_reason,
        retry_diagnosis=retry_diagnosis,
        retry_verification=retry_verification,
        final_answer=final_answer,
        final_outcome=FinalOutcome.INSUFFICIENT_EVIDENCE,
        system_errors=system_errors,
    )
