"""
DevTrace — Module 5: Verified pipeline.

This module extends the M4 baseline pipeline by adding M5 evidence verification.

The M4 baseline pipeline (src/pipeline/baseline.py) is FROZEN and unchanged.
This module wraps it and adds the verification step.

Pipeline steps
--------------
1. Normalize incident  (M2)
2. Retrieve candidates  (M3 HybridRetriever)
3. NEEDS_INFO gate  (M4 deterministic check)
4. Applicability filtering  (M4 checker)
5. Diagnosis generation  (M4 DiagnosisGenerator + LLM)
6. Evidence verification  (M5 EvidenceVerifier + independent LLM call)

Output
------
VerifiedPipelineResult — extends M4's BaselinePipelineResult with M5's
DiagnosisVerification.

M5 Boundary
-----------
- Adds verification only.
- Does NOT implement retry.
- Does NOT implement final outcome classification.
- Does NOT implement answer assembly.
- Those belong to M6.

The verification step uses an independent LLM call, separate from diagnosis.
The same LLMClient instance may be passed for both (it is called twice), or
different instances may be injected for independent separation in tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.errors import LLMError, SchemaValidationError
from src.llm.base import LLMClient
from src.models.enums import SystemOutcome
from src.pipeline.baseline import BaselinePipelineResult, run_baseline_diagnosis
from src.retrieval.hybrid import HybridRetriever
from src.verification.models import DiagnosisVerification
from src.verification.verifier import verify_diagnosis
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extended result type
# ---------------------------------------------------------------------------

@dataclass
class VerifiedPipelineResult:
    """
    Complete result of the M4+M5 pipeline run.

    Inherits all M4 fields via the embedded baseline result, and adds M5
    verification output.

    Attributes:
        baseline:      Full M4 BaselinePipelineResult (all M4 fields accessible).
        verification:  M5 DiagnosisVerification (per-claim verdicts). None if
                       diagnosis did not reach M5 (e.g. NEEDS_INFO or DEGRADED).
        verification_error: Error message if M5 verification itself failed.
    """

    baseline: BaselinePipelineResult
    verification: DiagnosisVerification | None = None
    verification_error: str | None = None

    # Convenience pass-throughs to baseline fields
    @property
    def normalized(self):
        return self.baseline.normalized

    @property
    def retrieval_results(self):
        return self.baseline.retrieval_results

    @property
    def applicable_results(self):
        return self.baseline.applicable_results

    @property
    def applicability_decisions(self):
        return self.baseline.applicability_decisions

    @property
    def diagnosis(self):
        return self.baseline.diagnosis

    @property
    def needs_info(self):
        return self.baseline.needs_info

    @property
    def outcome(self):
        return self.baseline.outcome

    @property
    def error(self):
        return self.baseline.error


# ---------------------------------------------------------------------------
# Verified pipeline function
# ---------------------------------------------------------------------------

def run_verified_diagnosis(
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
) -> VerifiedPipelineResult:
    """
    Run the full M4+M5 DevTrace pipeline: diagnosis + evidence verification.

    Args:
        description:          Free-text incident description (required).
        current_version:      Explicit current SDK/API version (optional).
        previous_version:     Explicit previous version (optional).
        error_codes:          Explicit error codes (optional).
        product:              Product/component name (optional).
        context:              Arbitrary structured context dict (optional).
        retriever:            A loaded HybridRetriever instance (M3).
        llm_client:           LLM client for diagnosis generation (M4).
        verifier_llm_client:  Separate LLM client for verification (M5).
                              If None, uses the same llm_client for both.
                              Separating them allows independent test control.

    Returns:
        VerifiedPipelineResult with baseline M4 results + M5 verification.

    Notes:
        - If M4 returns NEEDS_INFO or DEGRADED, M5 verification is skipped.
        - M5 verification failure sets verification_error but does not override
          the baseline outcome. M6 will handle degraded verification.
        - No retry is attempted (M6's responsibility).
    """
    # ------------------------------------------------------------------
    # Run M4 baseline pipeline
    # ------------------------------------------------------------------
    logger.info("Running M4 baseline pipeline.")
    baseline = run_baseline_diagnosis(
        description=description,
        current_version=current_version,
        previous_version=previous_version,
        error_codes=error_codes,
        product=product,
        context=context,
        retriever=retriever,
        llm_client=llm_client,
    )

    result = VerifiedPipelineResult(baseline=baseline)

    # ------------------------------------------------------------------
    # Skip M5 if M4 did not produce a diagnosis
    # ------------------------------------------------------------------
    if baseline.diagnosis is None:
        logger.info(
            "Skipping M5 verification: M4 did not produce a diagnosis (outcome=%s).",
            baseline.outcome,
        )
        return result

    # ------------------------------------------------------------------
    # M5 Evidence Verification
    # ------------------------------------------------------------------
    logger.info("Step 6: Running M5 evidence verification.")

    verifier_client = verifier_llm_client if verifier_llm_client is not None else llm_client

    try:
        verification = verify_diagnosis(
            diagnosis=baseline.diagnosis,
            applicable_results=baseline.applicable_results,
            normalized=baseline.normalized,
            llm_client=verifier_client,
        )
        result.verification = verification
        logger.info(
            "M5 verification complete: %d/%d claims verified.",
            len(verification.verified_claim_ids),
            len(verification.claim_verifications),
        )
    except (LLMError, SchemaValidationError) as exc:
        logger.error("M5 verification failed: %s", exc)
        result.verification_error = str(exc)

    return result
