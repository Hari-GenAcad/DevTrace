"""
DevTrace — Module 8: Pipeline adapter.

This module is the thin bridge between the Streamlit UI and the existing
DevTrace M6 orchestration entry point.

Design constraints:
  - Does NOT reimplement retrieval, applicability, verification, or retry.
  - Does NOT modify any M1–M7 module.
  - Adapts to the existing run_troubleshooting API.
  - Builds the correct LLM client from environment configuration.
  - Builds the retriever from the existing corpus.

The UI calls run_pipeline(...) which internally calls run_troubleshooting(...).
This ensures the UI always exercises the actual DevTrace architecture.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import settings
from src.errors import ConfigurationError
from src.orchestration import run_troubleshooting, TroubleshootingResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Client factory
# ---------------------------------------------------------------------------

def build_llm_client(mode: str = "auto") -> Any:
    """
    Build the LLM client based on configuration.

    Args:
        mode: "auto"  — use Gemini if GEMINI_API_KEY is set, else FakeLLM.
              "fake"  — always use FakeLLMClient (safe for demos without API key).
              "live"  — always use GeminiClient (raises if key not set).

    Returns:
        An LLMClient implementation.

    Raises:
        ConfigurationError: If mode="live" and GEMINI_API_KEY is not set.
    """
    if mode == "fake":
        from src.llm.fake import FakeLLMClient  # noqa: PLC0415
        logger.info("UI: Using FakeLLMClient (deterministic mode).")
        return FakeLLMClient(
            response=(
                '{"claims": [{"role": "root_cause", "text": "Demo mode: '
                'configure GEMINI_API_KEY for live results.", "evidence_ids": []}]}'
            )
        )

    if mode == "live" or (mode == "auto" and settings.gemini_api_key):
        from src.llm.gemini import GeminiClient  # noqa: PLC0415
        try:
            client = GeminiClient()
            logger.info("UI: Using GeminiClient (live mode).")
            return client
        except ConfigurationError:
            if mode == "live":
                raise
            logger.warning("UI: GEMINI_API_KEY not available; falling back to FakeLLM.")

    # Fallback / auto without key
    from src.llm.fake import FakeLLMClient  # noqa: PLC0415
    logger.info("UI: Using FakeLLMClient (auto mode — no API key found).")
    return FakeLLMClient(
        response=(
            '{"claims": [{"role": "root_cause", '
            '"text": "Configure GEMINI_API_KEY in your .env file to get a live diagnosis.", '
            '"evidence_ids": []}]}'
        )
    )


# ---------------------------------------------------------------------------
# Retriever factory
# ---------------------------------------------------------------------------

def build_retriever() -> Any:
    """
    Build a live HybridRetriever from the existing corpus.

    Returns the HybridRetriever on success, or None if the corpus/index
    cannot be loaded (the UI handles this gracefully).

    Returns:
        HybridRetriever | None
    """
    try:
        from src.ingestion.loader import load_corpus_and_chunks  # noqa: PLC0415
        from src.retrieval.hybrid import HybridRetriever  # noqa: PLC0415

        _, chunks = load_corpus_and_chunks()
        if not chunks:
            logger.warning("UI: Corpus loaded but returned 0 chunks.")
            return None

        retriever = HybridRetriever(chunks)
        retriever.load()
        logger.info("UI: HybridRetriever loaded with %d chunks.", len(chunks))
        return retriever
    except Exception as exc:
        logger.error("UI: Failed to build retriever: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Pipeline adapter
# ---------------------------------------------------------------------------

def run_pipeline(
    *,
    description: str,
    current_version: str | None = None,
    previous_version: str | None = None,
    error_codes: list[str] | None = None,
    product: str | None = None,
    context: dict[str, Any] | None = None,
    retriever: Any,
    llm_client: Any,
) -> TroubleshootingResult:
    """
    Run the full DevTrace troubleshooting pipeline (M6 orchestration).

    This is the ONLY way M8 invokes the pipeline. The UI must never call
    M3, M4, M5, or M6 components directly — this function ensures the
    complete architecture is exercised in the correct order.

    Args:
        description:      Incident description text (required).
        current_version:  Current SDK/API version (optional).
        previous_version: Previous SDK/API version (optional).
        error_codes:      Observed error code list (optional).
        product:          Product/component name (optional).
        context:          Arbitrary structured context (optional).
        retriever:        A loaded HybridRetriever (or compatible stub).
        llm_client:       LLM client for diagnosis and verification.

    Returns:
        TroubleshootingResult — the complete M6 result record.
    """
    logger.info(
        "UI: Invoking M6 orchestration for incident: %r (version=%r)",
        description[:80],
        current_version,
    )

    result = run_troubleshooting(
        description=description,
        current_version=current_version or None,
        previous_version=previous_version or None,
        error_codes=error_codes or [],
        product=product or None,
        context=context or {},
        retriever=retriever,
        llm_client=llm_client,
        # Use same client for verification — the M6 orchestrator handles
        # independent calls internally.
        verifier_llm_client=llm_client,
    )

    logger.info(
        "UI: Pipeline complete. Outcome=%s, retrieval=%d, applicable=%d.",
        result.final_outcome.value,
        len(result.retrieval_results),
        len(result.applicable_results),
    )
    return result
