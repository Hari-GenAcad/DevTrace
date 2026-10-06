"""
DevTrace — Unit tests for Module 8: Presentation & UI Layer.

Tests:
  - LLM client builder (auto, fake, live modes)
  - Retriever builder
  - Pipeline adapter function
  - Pure rendering helper functions (trace data, evidence display, verification display)
  - Outcome labeling & color mappings
  - Error sanitization helper
"""

import pytest
from unittest.mock import MagicMock, patch

from src.errors import ConfigurationError
from src.m8.pipeline import build_llm_client, build_retriever, run_pipeline
from src.m8.render import (
    _sanitize_error,
    build_evidence_display_data,
    build_pipeline_trace_data,
    build_verification_display_data,
    claim_role_label,
    outcome_color,
    outcome_label,
)
from src.models.contracts import (
    ApplicabilityResult,
    DiagnosisClaim,
    DiagnosisResult,
    RetrievalResult,
    TroubleshootingIncident,
)
from src.models.enums import ClaimRole, RetrievalSource
from src.normalization.normalizer import normalize_incident
from src.orchestration.models import (
    FinalOutcome,
    TroubleshootingResult,
    VerifiedAnswer,
)
from src.verification.models import (
    CitationValidity,
    ClaimVerification,
    DiagnosisVerification,
    VerificationVerdict,
)


# ---------------------------------------------------------------------------
# Test LLM Client Builder
# ---------------------------------------------------------------------------

def test_build_llm_client_fake_mode():
    client = build_llm_client(mode="fake")
    assert client.__class__.__name__ == "FakeLLMClient"


def test_build_llm_client_auto_mode_no_key(monkeypatch):
    monkeypatch.setattr("src.config.settings.gemini_api_key", None)
    client = build_llm_client(mode="auto")
    assert client.__class__.__name__ == "FakeLLMClient"


def test_build_llm_client_live_mode_no_key(monkeypatch):
    monkeypatch.setattr("src.config.settings.gemini_api_key", None)
    with pytest.raises(ConfigurationError):
        build_llm_client(mode="live")


# ---------------------------------------------------------------------------
# Test Retriever Builder
# ---------------------------------------------------------------------------

def test_build_retriever_success():
    retriever = build_retriever()
    assert retriever is not None


# ---------------------------------------------------------------------------
# Test Pipeline Adapter
# ---------------------------------------------------------------------------

def test_run_pipeline_adapter():
    mock_retriever = MagicMock()
    mock_llm = MagicMock()

    with patch("src.m8.pipeline.run_troubleshooting") as mock_run_troubleshooting:
        mock_result = MagicMock(spec=TroubleshootingResult)
        mock_result.final_outcome = FinalOutcome.ANSWERED_FULL
        mock_result.retrieval_results = []
        mock_result.applicable_results = []
        mock_run_troubleshooting.return_value = mock_result

        res = run_pipeline(
            description="Test incident desc",
            current_version="3.1",
            previous_version="2.8",
            error_codes=["AUTH_401"],
            product="DevCore SDK",
            retriever=mock_retriever,
            llm_client=mock_llm,
        )

        assert res == mock_result
        mock_run_troubleshooting.assert_called_once()
        kwargs = mock_run_troubleshooting.call_args.kwargs
        assert kwargs["description"] == "Test incident desc"
        assert kwargs["current_version"] == "3.1"
        assert kwargs["previous_version"] == "2.8"
        assert kwargs["error_codes"] == ["AUTH_401"]
        assert kwargs["product"] == "DevCore SDK"


# ---------------------------------------------------------------------------
# Test Rendering Helper Functions
# ---------------------------------------------------------------------------

def test_outcome_labels_and_colors():
    assert "Answered (Full)" in outcome_label(FinalOutcome.ANSWERED_FULL)
    assert outcome_color(FinalOutcome.ANSWERED_FULL) == "success"
    assert outcome_color(FinalOutcome.DEGRADED) == "error"
    assert outcome_color(FinalOutcome.NEEDS_INFO) == "info"

    assert claim_role_label(ClaimRole.ROOT_CAUSE) == "Root Cause"
    assert claim_role_label(ClaimRole.FIX) == "Fix"
    assert claim_role_label(ClaimRole.EXPLANATION) == "Explanation"


def test_build_pipeline_trace_data():
    norm_inc = normalize_incident(description="AUTH_401 error", current_version="3.1")
    ret_results = [
        RetrievalResult(chunk_id="chunk_1", doc_id="doc_1", score=0.9, source=RetrievalSource.SEMANTIC, metadata={}),
        RetrievalResult(chunk_id="chunk_2", doc_id="doc_2", score=0.8, source=RetrievalSource.KEYWORD, metadata={}),
    ]
    app_results = [
        RetrievalResult(chunk_id="chunk_1", doc_id="doc_1", score=0.9, source=RetrievalSource.SEMANTIC, metadata={}),
    ]
    diag = DiagnosisResult(
        claims=[
            DiagnosisClaim(claim_id="c1", role=ClaimRole.ROOT_CAUSE, text="Root cause text", evidence_ids=["chunk_1"]),
        ]
    )

    tr = TroubleshootingResult(
        incident_description="AUTH_401 error",
        current_version="3.1",
        final_outcome=FinalOutcome.ANSWERED_FULL,
        final_answer=VerifiedAnswer(
            outcome=FinalOutcome.ANSWERED_FULL,
            answer_text="Root cause text",
            verified_claims=diag.claims,
        ),
        retrieval_results=ret_results,
        applicable_results=app_results,
        initial_diagnosis=diag,
    )

    trace_steps = build_pipeline_trace_data(tr)
    assert len(trace_steps) >= 3
    assert trace_steps[0]["label"] == "Retrieved"
    assert trace_steps[0]["value"] == 2
    assert trace_steps[1]["label"] == "Applicable"
    assert trace_steps[1]["value"] == 1


def test_build_evidence_display_data():
    ret_results = [
        RetrievalResult(
            chunk_id="chunk_1", doc_id="doc_1", score=0.9, source=RetrievalSource.SEMANTIC, metadata={"content": "Content 1", "applies_to": "3.x"}
        ),
        RetrievalResult(
            chunk_id="chunk_2", doc_id="doc_2", score=0.8, source=RetrievalSource.KEYWORD, metadata={"content": "Content 2", "applies_to": "1.x"}
        ),
    ]
    app_results = [ret_results[0]]
    decisions = [
        ApplicabilityResult(chunk_id="chunk_1", doc_id="doc_1", applicable=True, reason="Applies to 3.x"),
        ApplicabilityResult(chunk_id="chunk_2", doc_id="doc_2", applicable=False, reason="Version mismatch"),
    ]
    claim = DiagnosisClaim(claim_id="c1", role=ClaimRole.ROOT_CAUSE, text="Root cause", evidence_ids=["chunk_1"])

    tr = TroubleshootingResult(
        incident_description="AUTH_401 error",
        current_version="3.1",
        final_outcome=FinalOutcome.ANSWERED_FULL,
        final_answer=VerifiedAnswer(outcome=FinalOutcome.ANSWERED_FULL, answer_text="Root cause", verified_claims=[claim]),
        retrieval_results=ret_results,
        applicable_results=app_results,
        applicability_decisions=decisions,
    )

    display_items = build_evidence_display_data(tr)
    assert len(display_items) == 2
    # Applicable item comes first due to sorting
    assert display_items[0]["chunk_id"] == "chunk_1"
    assert display_items[0]["applicable"] is True
    assert display_items[0]["cited_in_final"] is True
    assert display_items[1]["chunk_id"] == "chunk_2"
    assert display_items[1]["applicable"] is False


def test_build_verification_display_data():
    claim = DiagnosisClaim(claim_id="c1", role=ClaimRole.ROOT_CAUSE, text="Root cause text", evidence_ids=["chunk_1"])
    diag = DiagnosisResult(claims=[claim])
    cv = ClaimVerification(
        claim_id="c1",
        verdict=VerificationVerdict.VERIFIED,
        citation_validity=CitationValidity.VALID,
        citation_correct=True,
        sufficient=True,
        contradicted=False,
        reason="Evidence supports claim",
        supporting_evidence_ids=["chunk_1"],
    )
    vr = DiagnosisVerification(claim_verifications=[cv])

    tr = TroubleshootingResult(
        incident_description="Test incident",
        final_outcome=FinalOutcome.ANSWERED_FULL,
        final_answer=VerifiedAnswer(outcome=FinalOutcome.ANSWERED_FULL, answer_text="Root cause text", verified_claims=[claim]),
        initial_diagnosis=diag,
        initial_verification=vr,
    )

    verif_items = build_verification_display_data(tr)
    assert len(verif_items) == 1
    assert verif_items[0]["claim_id"] == "c1"
    assert verif_items[0]["verdict"] == "VERIFIED"
    assert verif_items[0]["citation_correct"] is True
    assert verif_items[0]["in_final_answer"] is True


def test_sanitize_error():
    secret_key = "AIzaSyD-1234567890abcdefghijklmnopqrstuvwxyz"
    raw_error = f"API request failed with key {secret_key} on endpoint"
    sanitized = _sanitize_error(raw_error)
    assert secret_key not in sanitized
    assert "[REDACTED]" in sanitized
