"""
DevTrace — Module 8: Streamlit UI, Final Integration & Demonstration.

Provides the presentation layer for DevTrace.

M8 is NOT a reasoning module. It consumes results from M1–M7 and renders them.

Public surface:
    build_llm_client  — constructs the correct LLM client (FakeLLM or Gemini)
    build_retriever   — constructs a live HybridRetriever from the corpus
    run_pipeline      — thin wrapper around run_troubleshooting (M6 entry point)
    render_*          — UI rendering helpers (pure functions over pipeline results)
"""

from src.m8.pipeline import build_llm_client, build_retriever, run_pipeline
from src.m8.render import (
    render_evidence_section,
    render_final_answer,
    render_pipeline_trace,
    render_verification_section,
)

__all__ = [
    "build_llm_client",
    "build_retriever",
    "run_pipeline",
    "render_evidence_section",
    "render_final_answer",
    "render_pipeline_trace",
    "render_verification_section",
]
