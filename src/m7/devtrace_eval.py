"""
DevTrace — Module 7: DevTrace evaluation adapter.

Runs the existing M6 orchestration pipeline (run_troubleshooting) on a single
evaluation case and translates the TroubleshootingResult into the M7
DevTraceEvalResult schema needed by the metric layer.

Key design constraints:
  - Gold labels (gold_doc_ids, forbidden_doc_ids) are NEVER passed into the
    DevTrace pipeline. They are used only by this adapter AFTER the pipeline
    has run, purely for metric extraction.
  - No production logic is modified.
  - This is a thin translation layer only.
"""

from __future__ import annotations

import logging
from typing import Any

from src.llm.base import LLMClient
from src.m7.models import DevTraceEvalResult
from src.orchestration.models import FinalOutcome, TroubleshootingResult
from src.orchestration.orchestrator import run_troubleshooting
from src.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

# Outcomes where no answer is produced
_ABSTENTION_OUTCOMES: frozenset[str] = frozenset({
    FinalOutcome.INSUFFICIENT_EVIDENCE.value,
    FinalOutcome.NEEDS_INFO.value,
    FinalOutcome.DEGRADED.value,
})


def _deduplicate(ids: list[str]) -> list[str]:
    """Return a list with duplicates removed, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _extract_doc_ids_from_chunks(chunk_ids: list[str]) -> list[str]:
    """
    Extract doc IDs from chunk IDs using the DevCore naming convention.
    Chunk IDs are formatted as '<DOC_ID>-C<N>' (e.g. AUTH-002-C01).
    Falls back to the full chunk_id if the pattern doesn't match.
    """
    doc_ids: list[str] = []
    for cid in chunk_ids:
        # Attempt to strip the '-C<digits>' suffix
        parts = cid.rsplit("-C", 1)
        if len(parts) == 2 and parts[1].isdigit():
            doc_ids.append(parts[0])
        else:
            doc_ids.append(cid)
    return _deduplicate(doc_ids)


def _collect_cited_chunk_ids(result: TroubleshootingResult) -> list[str]:
    """
    Collect chunk IDs cited in verified claims.

    Uses the final_answer.verified_claims (which are the surviving verified claims).
    If retry was attempted, the retry diagnosis claims are used if available.
    """
    cited: set[str] = set()

    # Verified claims in the final answer are always the authoritative source
    for claim in result.final_answer.verified_claims:
        for eid in claim.evidence_ids:
            cited.add(eid)

    return list(cited)


def run_devtrace_eval(
    *,
    description: str,
    current_version: str | None = None,
    previous_version: str | None = None,
    error_codes: list[str] | None = None,
    product: str | None = None,
    retriever: HybridRetriever,
    llm_client: LLMClient,
    verifier_llm_client: LLMClient | None = None,
    # Gold labels — used ONLY after the pipeline runs, for metric extraction.
    gold_forbidden_doc_ids: list[str] | None = None,
) -> DevTraceEvalResult:
    """
    Run the full DevTrace M6 orchestration and translate the result into
    the M7 DevTraceEvalResult schema.

    Gold labels are passed here ONLY for post-hoc metric extraction
    (e.g. computing which forbidden docs appeared in retrieval results).
    They are NEVER injected into the pipeline itself.

    Args:
        description:          Incident description text.
        current_version:      Current SDK/API version.
        previous_version:     Previous version (if known).
        error_codes:          Observed error codes.
        product:              Product name.
        retriever:            Loaded HybridRetriever.
        llm_client:           LLM client for diagnosis.
        verifier_llm_client:  LLM client for verification (optional).
        gold_forbidden_doc_ids: Forbidden doc IDs from the eval dataset
                                (NOT passed to the pipeline).

    Returns:
        DevTraceEvalResult — M7-ready structured result.
    """
    gold_forbidden = gold_forbidden_doc_ids or []

    # --- Run the production pipeline (unchanged) ---
    try:
        result: TroubleshootingResult = run_troubleshooting(
            description=description,
            current_version=current_version,
            previous_version=previous_version,
            error_codes=error_codes,
            product=product,
            retriever=retriever,
            llm_client=llm_client,
            verifier_llm_client=verifier_llm_client,
        )
    except Exception as exc:
        logger.error("DevTrace pipeline raised unexpected exception: %s", exc)
        return DevTraceEvalResult(
            final_outcome=FinalOutcome.DEGRADED.value,
            abstained=True,
            error=f"Pipeline exception: {type(exc).__name__}: {exc}",
        )

    # --- Extract fields needed by M7 metric layer ---

    # Retrieval results
    retrieved_chunk_ids = [r.chunk_id for r in result.retrieval_results]
    retrieved_doc_ids = _deduplicate([r.doc_id for r in result.retrieval_results])

    # Applicable results
    applicable_chunk_ids = [r.chunk_id for r in result.applicable_results]
    applicable_doc_ids = _deduplicate([r.doc_id for r in result.applicable_results])

    # Citations from verified claims (the only trusted citations in DevTrace)
    cited_chunk_ids = _collect_cited_chunk_ids(result)
    cited_doc_ids = _extract_doc_ids_from_chunks(cited_chunk_ids)

    # Forbidden docs that ended up in retrieved results (for reporting)
    forbidden_retrieved = [
        d for d in gold_forbidden if d in set(retrieved_doc_ids)
    ]

    # Determine abstention
    final_outcome_str = result.final_outcome.value
    abstained = final_outcome_str in _ABSTENTION_OUTCOMES

    # Retry outcome
    retry_attempted = result.retry_attempted
    retry_succeeded = False
    if retry_attempted and result.retry_verification is not None:
        # Retry succeeded if the retry root cause survived
        for cv in result.retry_verification.claim_verifications:
            if cv.role == "root_cause" and cv.is_verified:
                retry_succeeded = True
                break

    # Serialise the full result for traceability
    try:
        full_result_dict = result.model_dump()
    except Exception:
        full_result_dict = {}

    logger.info(
        "DevTrace eval: outcome=%s, retry=%s, retry_ok=%s",
        final_outcome_str,
        retry_attempted,
        retry_succeeded,
    )

    return DevTraceEvalResult(
        final_outcome=final_outcome_str,
        answer_text=result.final_answer.answer_text,
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_doc_ids=retrieved_doc_ids,
        applicable_chunk_ids=applicable_chunk_ids,
        applicable_doc_ids=applicable_doc_ids,
        cited_chunk_ids=cited_chunk_ids,
        cited_doc_ids=cited_doc_ids,
        forbidden_doc_ids_cited=forbidden_retrieved,
        retry_attempted=retry_attempted,
        retry_succeeded=retry_succeeded,
        abstained=abstained,
        full_result=full_result_dict,
    )
