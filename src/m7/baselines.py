"""
DevTrace — Module 7: Baseline implementations.

Implements Baseline A (Naive RAG) and Baseline B (Retrieve + Threshold).

Design principles:
  - Both baselines use the SAME corpus and HybridRetriever as DevTrace.
  - Both baselines use the SAME LLM client interface (LLMClient abstraction).
  - Neither baseline has applicability filtering, verification, or retry.
  - Gold labels are NEVER passed to baselines.
  - Cited doc IDs are extracted heuristically from response text (not structured).
  - Both baselines produce a plain-text answer with no claim structure.

Baseline A:
    Retrieve → Build prompt with ALL retrieved chunks → Generate → Return text.
    No threshold, no filtering, no verification.

Baseline B:
    Retrieve → If top_score >= threshold → Generate. Else → Abstain.
    Same raw retrieval as Baseline A; only the threshold guard differs.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.llm.base import LLMClient
from src.m7.models import BaselineAResult, BaselineBResult
from src.models.contracts import RetrievalResult
from src.normalization.normalizer import NormalizedIncident
from src.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared retrieval helper
# ---------------------------------------------------------------------------

def _run_retrieval(
    normalized: NormalizedIncident,
    retriever: HybridRetriever,
) -> list[RetrievalResult]:
    """Run hybrid retrieval and return the RetrievalResult list."""
    try:
        return retriever.retrieve_as_contracts(normalized)
    except Exception as exc:
        logger.error("Baseline retrieval failed: %s", exc)
        return []


def _extract_doc_ids(results: list[RetrievalResult]) -> list[str]:
    """Deduplicated doc IDs from a retrieval result list, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for r in results:
        if r.doc_id not in seen:
            seen.add(r.doc_id)
            out.append(r.doc_id)
    return out


def _extract_chunk_ids(results: list[RetrievalResult]) -> list[str]:
    """Ordered chunk IDs from a retrieval result list."""
    return [r.chunk_id for r in results]


# ---------------------------------------------------------------------------
# Citation extraction heuristic
# ---------------------------------------------------------------------------

# Known doc-id patterns in the DevCore corpus (prefix-based).
# This is purely for metric logging — it does NOT affect scoring logic.
_DOC_ID_PATTERN = re.compile(
    r"\b("
    r"AUTH-\d+|SDK-\d+|RATE-\d+|WEBHOOK-\d+|TS-\d+|CFG-\d+|"
    r"PERF-\d+|API-\d+|ERR-\d+|SECURITY-\d+"
    r")\b"
)


def _heuristic_cited_doc_ids(answer_text: str) -> list[str]:
    """
    Extract doc IDs mentioned in the answer text via pattern matching.

    This is a best-effort heuristic for metric logging purposes only.
    Baselines do not produce structured citation IDs.
    """
    if not answer_text:
        return []
    matches = _DOC_ID_PATTERN.findall(answer_text)
    seen: set[str] = set()
    out: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Shared naive prompt builder
# ---------------------------------------------------------------------------

def _build_naive_prompt(
    incident_description: str,
    current_version: str | None,
    previous_version: str | None,
    error_codes: list[str],
    retrieved_results: list[RetrievalResult],
) -> str:
    """
    Build a simple retrieve-then-generate prompt.

    No applicability filtering — ALL retrieved chunks are included.
    The LLM is instructed to answer from this evidence.
    """
    context_lines = [f"Description: {incident_description}"]
    if current_version:
        context_lines.append(f"Current version: {current_version}")
    if previous_version:
        context_lines.append(f"Previous version: {previous_version}")
    if error_codes:
        context_lines.append(f"Error codes: {', '.join(error_codes)}")
    incident_block = "\n".join(context_lines)

    evidence_lines: list[str] = []
    for rr in retrieved_results:
        content = rr.metadata.get("content", "(no content)")
        applies_to = rr.metadata.get("applies_to", "*")
        evidence_lines.append(f"--- [{rr.chunk_id} | doc:{rr.doc_id} | range:{applies_to}] ---")
        evidence_lines.append(content)
        evidence_lines.append("")

    evidence_block = "\n".join(evidence_lines) if evidence_lines else "(No evidence retrieved.)"

    return f"""\
You are a developer support engineer. Answer the following incident based ONLY on the provided documentation.

INCIDENT:
{incident_block}

DOCUMENTATION:
{evidence_block}

Provide a clear troubleshooting answer. If you reference specific documentation, mention the document ID (e.g. AUTH-002) in your response.
"""


# ---------------------------------------------------------------------------
# Baseline A: Naive RAG
# ---------------------------------------------------------------------------

def run_baseline_a(
    *,
    normalized: NormalizedIncident,
    retriever: HybridRetriever,
    llm_client: LLMClient,
) -> BaselineAResult:
    """
    Run Baseline A: Naive RAG (Retrieve → Generate).

    No applicability filtering, no verification, no retry.
    All retrieved chunks are passed directly to the LLM.

    Args:
        normalized:  Normalized incident from M2.
        retriever:   Loaded HybridRetriever (same corpus as DevTrace).
        llm_client:  LLM client for generation.

    Returns:
        BaselineAResult with answer_text and heuristic cited_doc_ids.
    """
    incident = normalized.incident
    signals = normalized.signals

    # --- Retrieval ---
    retrieved = _run_retrieval(normalized, retriever)
    if not retrieved:
        logger.info("Baseline A: no retrieval results — abstaining.")
        return BaselineAResult(
            retrieved_chunk_ids=[],
            retrieved_doc_ids=[],
            abstained=True,
        )

    chunk_ids = _extract_chunk_ids(retrieved)
    doc_ids = _extract_doc_ids(retrieved)

    # --- Generation ---
    prompt = _build_naive_prompt(
        incident_description=incident.description,
        current_version=incident.current_version,
        previous_version=incident.previous_version,
        error_codes=signals.error_codes,
        retrieved_results=retrieved,
    )

    try:
        response = llm_client.generate(prompt)
    except Exception as exc:
        logger.error("Baseline A LLM call failed: %s", exc)
        return BaselineAResult(
            retrieved_chunk_ids=chunk_ids,
            retrieved_doc_ids=doc_ids,
            abstained=True,
            error=f"LLM error: {type(exc).__name__}: {exc}",
        )

    if not response.success or not response.text:
        return BaselineAResult(
            retrieved_chunk_ids=chunk_ids,
            retrieved_doc_ids=doc_ids,
            abstained=True,
            error="LLM returned empty response.",
        )

    answer_text = response.text.strip()
    cited_doc_ids = _heuristic_cited_doc_ids(answer_text)

    logger.info(
        "Baseline A: generated answer (%d chars), heuristic cited docs: %s",
        len(answer_text),
        cited_doc_ids,
    )

    return BaselineAResult(
        retrieved_chunk_ids=chunk_ids,
        retrieved_doc_ids=doc_ids,
        answer_text=answer_text,
        cited_doc_ids=cited_doc_ids,
        abstained=False,
    )


# ---------------------------------------------------------------------------
# Baseline B: Retrieve + Score Threshold
# ---------------------------------------------------------------------------

def run_baseline_b(
    *,
    normalized: NormalizedIncident,
    retriever: HybridRetriever,
    llm_client: LLMClient,
    threshold: float = 0.40,
) -> BaselineBResult:
    """
    Run Baseline B: Retrieve + Score Threshold.

    Retrieves candidates, then checks whether the top retrieval score
    exceeds the configured threshold. If yes, generates an answer.
    If no, abstains (no answer produced).

    No applicability filtering, no verification, no retry.

    Args:
        normalized:  Normalized incident from M2.
        retriever:   Loaded HybridRetriever (same corpus as DevTrace).
        llm_client:  LLM client for generation.
        threshold:   Score threshold in [0.0, 1.0] (default 0.40).
                     Tune this via config, not on the evaluation cases.

    Returns:
        BaselineBResult including threshold_passed flag and answer.
    """
    incident = normalized.incident
    signals = normalized.signals

    # --- Retrieval ---
    retrieved = _run_retrieval(normalized, retriever)
    if not retrieved:
        logger.info("Baseline B: no retrieval results — abstaining.")
        return BaselineBResult(
            retrieved_chunk_ids=[],
            retrieved_doc_ids=[],
            top_score=None,
            threshold_used=threshold,
            threshold_passed=False,
            abstained=True,
        )

    chunk_ids = _extract_chunk_ids(retrieved)
    doc_ids = _extract_doc_ids(retrieved)
    top_score = retrieved[0].score

    # --- Threshold check ---
    if top_score < threshold:
        logger.info(
            "Baseline B: top score %.4f < threshold %.4f — abstaining.",
            top_score,
            threshold,
        )
        return BaselineBResult(
            retrieved_chunk_ids=chunk_ids,
            retrieved_doc_ids=doc_ids,
            top_score=top_score,
            threshold_used=threshold,
            threshold_passed=False,
            abstained=True,
        )

    logger.info(
        "Baseline B: top score %.4f >= threshold %.4f — generating.",
        top_score,
        threshold,
    )

    # --- Generation ---
    prompt = _build_naive_prompt(
        incident_description=incident.description,
        current_version=incident.current_version,
        previous_version=incident.previous_version,
        error_codes=signals.error_codes,
        retrieved_results=retrieved,
    )

    try:
        response = llm_client.generate(prompt)
    except Exception as exc:
        logger.error("Baseline B LLM call failed: %s", exc)
        return BaselineBResult(
            retrieved_chunk_ids=chunk_ids,
            retrieved_doc_ids=doc_ids,
            top_score=top_score,
            threshold_used=threshold,
            threshold_passed=True,
            abstained=True,
            error=f"LLM error: {type(exc).__name__}: {exc}",
        )

    if not response.success or not response.text:
        return BaselineBResult(
            retrieved_chunk_ids=chunk_ids,
            retrieved_doc_ids=doc_ids,
            top_score=top_score,
            threshold_used=threshold,
            threshold_passed=True,
            abstained=True,
            error="LLM returned empty response.",
        )

    answer_text = response.text.strip()
    cited_doc_ids = _heuristic_cited_doc_ids(answer_text)

    logger.info(
        "Baseline B: generated answer (%d chars), heuristic cited docs: %s",
        len(answer_text),
        cited_doc_ids,
    )

    return BaselineBResult(
        retrieved_chunk_ids=chunk_ids,
        retrieved_doc_ids=doc_ids,
        top_score=top_score,
        threshold_used=threshold,
        threshold_passed=True,
        answer_text=answer_text,
        cited_doc_ids=cited_doc_ids,
        abstained=False,
    )
