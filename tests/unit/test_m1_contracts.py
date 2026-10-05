"""
M1 Tests — Contracts, Core Types & Test Harness

Covers all 15 acceptance-criteria test cases from the M1 spec.
All tests are fully offline and deterministic (no Gemini calls, no network).
"""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from src.config import DevTraceConfig
from src.errors import ConfigurationError, LLMError, SchemaValidationError
from src.llm.fake import FakeLLMClient
from src.llm.gemini import GeminiClient
from src.models import (
    AnswerCompleteness,
    ApplicabilityResult,
    ClaimRole,
    ContradictionDecision,
    DiagnosisClaim,
    DiagnosisResult,
    Document,
    EvidenceChunk,
    FinalAnswer,
    RetrievalResult,
    RetrievalSource,
    SupportDecision,
    SystemOutcome,
    Trace,
    TroubleshootingIncident,
    VerificationStatus,
    VerificationVerdict,
)


# ===========================================================================
# Test 1 — Incident contract: valid full incident
# ===========================================================================

class TestIncidentContract:
    def test_valid_full_incident(self):
        """A fully populated TroubleshootingIncident validates correctly."""
        incident = TroubleshootingIncident(
            description="AUTH_401 started after upgrading SDK from 2.8 to 3.1.",
            current_version="3.1",
            previous_version="2.8",
            error_codes=["AUTH_401", "HTTP_401"],
            product="DevCore SDK",
            context={"runtime": "Node 20", "env": "production"},
        )
        assert incident.description == "AUTH_401 started after upgrading SDK from 2.8 to 3.1."
        assert incident.current_version == "3.1"
        assert incident.previous_version == "2.8"
        assert "AUTH_401" in incident.error_codes
        assert incident.context["runtime"] == "Node 20"

    # Test 2 — Optional incident fields
    def test_description_only_incident(self):
        """An incident with only a description is valid — all other fields are optional."""
        incident = TroubleshootingIncident(description="Something is broken.")
        assert incident.current_version is None
        assert incident.previous_version is None
        assert incident.error_codes == []
        assert incident.product is None
        assert incident.context == {}

    # Test 3 — Structured incident fields validate
    def test_structured_incident_fields(self):
        """Structured fields (list, dict, optional str) all validate."""
        incident = TroubleshootingIncident(
            description="Webhook delivery failing.",
            current_version="4.0",
            error_codes=["WEBHOOK_503", "RETRY_LIMIT"],
            product="Payments API",
            context={"region": "us-east-1", "retry_count": 5},
        )
        assert len(incident.error_codes) == 2
        assert incident.context["retry_count"] == 5

    # Test 4a — Missing description is rejected
    def test_missing_description_rejected(self):
        """An incident without a description fails validation."""
        with pytest.raises(ValidationError):
            TroubleshootingIncident()  # type: ignore[call-arg]

    # Test 4b — Empty description is rejected
    def test_empty_description_rejected(self):
        """An empty string description is rejected by min_length=1."""
        with pytest.raises(ValidationError):
            TroubleshootingIncident(description="")

    # Test 4c — Wrong type for error_codes is rejected
    def test_wrong_type_error_codes_rejected(self):
        """Passing a plain string for error_codes is rejected."""
        with pytest.raises(ValidationError):
            TroubleshootingIncident(
                description="Some issue.",
                error_codes="AUTH_401",  # must be a list  # type: ignore[arg-type]
            )


# ===========================================================================
# Test 5 — Document contract
# ===========================================================================

class TestDocumentContract:
    def test_valid_document(self):
        """A fully populated Document validates correctly."""
        doc = Document(
            doc_id="doc-auth-3x-001",
            title="AUTH_401 Troubleshooting — SDK 3.x",
            content="If you receive AUTH_401 after upgrading to SDK 3.x ...",
            applies_to=">=3.0,<4.0",
            topic="authentication",
            metadata={"author": "platform-team", "last_updated": "2024-01"},
        )
        assert doc.doc_id == "doc-auth-3x-001"
        assert doc.applies_to == ">=3.0,<4.0"
        assert doc.topic == "authentication"

    def test_version_agnostic_document(self):
        """A version-agnostic document uses applies_to='*'."""
        doc = Document(
            doc_id="doc-general-001",
            title="General Debugging Guide",
            content="General steps for debugging API issues.",
        )
        assert doc.applies_to == "*"

    def test_evidence_chunk_traces_to_document(self):
        """An EvidenceChunk carries doc_id for source traceability."""
        chunk = EvidenceChunk(
            chunk_id="chunk-auth-3x-001-0",
            doc_id="doc-auth-3x-001",
            content="Ensure the Authorization header uses Bearer token format.",
            applies_to=">=3.0,<4.0",
            topic="authentication",
        )
        assert chunk.doc_id == "doc-auth-3x-001"
        assert chunk.chunk_id != chunk.doc_id


# ===========================================================================
# Test 6 — Retrieval result contract
# ===========================================================================

class TestRetrievalResultContract:
    def test_valid_retrieval_result(self):
        """A RetrievalResult validates with score and source."""
        result = RetrievalResult(
            chunk_id="chunk-001",
            doc_id="doc-001",
            score=0.87,
            source=RetrievalSource.HYBRID,
        )
        assert result.score == pytest.approx(0.87)
        assert result.source == RetrievalSource.HYBRID

    def test_score_out_of_range_rejected(self):
        """A score outside [0, 1] is rejected."""
        with pytest.raises(ValidationError):
            RetrievalResult(
                chunk_id="chunk-001",
                doc_id="doc-001",
                score=1.5,
                source=RetrievalSource.SEMANTIC,
            )

    def test_invalid_retrieval_source_rejected(self):
        """An unknown retrieval source string is rejected."""
        with pytest.raises(ValidationError):
            RetrievalResult(
                chunk_id="chunk-001",
                doc_id="doc-001",
                score=0.5,
                source="magic",  # type: ignore[arg-type]
            )


# ===========================================================================
# Test 7 — Diagnosis claim
# ===========================================================================

class TestDiagnosisClaimContract:
    def test_valid_root_cause_claim(self):
        """A root_cause claim with evidence IDs is valid."""
        claim = DiagnosisClaim(
            role=ClaimRole.ROOT_CAUSE,
            text="The SDK 3.x requires a Bearer token; the old API key format is rejected.",
            evidence_ids=["chunk-auth-3x-001-0"],
        )
        assert claim.role == ClaimRole.ROOT_CAUSE
        assert claim.claim_id  # auto-generated UUID
        assert len(claim.evidence_ids) == 1

    def test_claim_id_auto_generated(self):
        """Two claims without explicit IDs get distinct auto-generated IDs."""
        c1 = DiagnosisClaim(role=ClaimRole.FIX, text="Update auth header.", evidence_ids=[])
        c2 = DiagnosisClaim(role=ClaimRole.FIX, text="Update auth header.", evidence_ids=[])
        assert c1.claim_id != c2.claim_id

    # Test 8 — Invalid claim role
    def test_invalid_role_rejected(self):
        """An unrecognised claim role string is rejected."""
        with pytest.raises(ValidationError):
            DiagnosisClaim(
                role="root_cause_maybe",  # type: ignore[arg-type]
                text="Something.",
                evidence_ids=[],
            )

    def test_diagnosis_result_requires_exactly_one_root_cause(self):
        """DiagnosisResult rejects 0 or 2+ root_cause claims."""
        fix = DiagnosisClaim(role=ClaimRole.FIX, text="Apply patch.", evidence_ids=[])

        # Zero root_cause
        with pytest.raises(ValidationError, match="root_cause"):
            DiagnosisResult(claims=[fix])

        # Two root_cause claims
        rc1 = DiagnosisClaim(role=ClaimRole.ROOT_CAUSE, text="Cause A.", evidence_ids=[])
        rc2 = DiagnosisClaim(role=ClaimRole.ROOT_CAUSE, text="Cause B.", evidence_ids=[])
        with pytest.raises(ValidationError, match="root_cause"):
            DiagnosisResult(claims=[rc1, rc2, fix])

    def test_diagnosis_result_valid(self):
        """A valid DiagnosisResult with one root_cause and one fix passes."""
        rc = DiagnosisClaim(role=ClaimRole.ROOT_CAUSE, text="Root cause.", evidence_ids=["c-1"])
        fix = DiagnosisClaim(role=ClaimRole.FIX, text="Apply fix.", evidence_ids=["c-2"])
        result = DiagnosisResult(claims=[rc, fix])
        assert len(result.claims) == 2


# ===========================================================================
# Test 9 — Outcome enums
# ===========================================================================

class TestOutcomeEnums:
    def test_all_system_outcomes_exist(self):
        """All four primary outcomes are defined."""
        outcomes = {o.value for o in SystemOutcome}
        assert outcomes == {"ANSWERED", "INSUFFICIENT_EVIDENCE", "NEEDS_INFO", "DEGRADED"}

    def test_answer_completeness_values(self):
        """FULL and PARTIAL completeness values exist."""
        completeness = {c.value for c in AnswerCompleteness}
        assert completeness == {"FULL", "PARTIAL"}

    def test_claim_roles_exist(self):
        """ROOT_CAUSE, FIX, EXPLANATION roles exist."""
        roles = {r.value for r in ClaimRole}
        assert roles == {"root_cause", "fix", "explanation"}

    def test_final_answer_completeness_requires_answered(self):
        """Setting completeness on a non-ANSWERED outcome raises ValidationError."""
        with pytest.raises(ValidationError, match="completeness"):
            FinalAnswer(
                outcome=SystemOutcome.INSUFFICIENT_EVIDENCE,
                completeness=AnswerCompleteness.FULL,  # invalid
            )

    def test_final_answer_answered_requires_completeness(self):
        """An ANSWERED FinalAnswer without completeness raises ValidationError."""
        with pytest.raises(ValidationError, match="completeness"):
            FinalAnswer(
                outcome=SystemOutcome.ANSWERED,
                completeness=None,
            )

    def test_final_answer_full(self):
        """A FULL ANSWERED FinalAnswer is valid."""
        rc = DiagnosisClaim(role=ClaimRole.ROOT_CAUSE, text="Cause.", evidence_ids=["c-1"])
        answer = FinalAnswer(
            outcome=SystemOutcome.ANSWERED,
            completeness=AnswerCompleteness.FULL,
            answer_text="The root cause is ...",
            verified_claims=[rc],
        )
        assert answer.completeness == AnswerCompleteness.FULL


# ===========================================================================
# Test 10 — Trace contract
# ===========================================================================

class TestTraceContract:
    def test_trace_with_incident_only(self):
        """A Trace can be created with just the incident; all other fields default."""
        incident = TroubleshootingIncident(description="API returning 500 errors.")
        trace = Trace(incident=incident)
        assert trace.trace_id  # auto UUID
        assert trace.retrieved_results == []
        assert trace.applicable_chunk_ids == []
        assert trace.final_answer is None
        assert trace.system_errors == []

    def test_trace_partial_population(self):
        """A Trace can be partially populated across multiple fields."""
        incident = TroubleshootingIncident(
            description="Webhook timeout.",
            current_version="2.0",
        )
        verdict = VerificationVerdict(
            claim_id="claim-1",
            status=VerificationStatus.PASS,
            support=SupportDecision.SUPPORTED,
            contradiction=ContradictionDecision.NONE,
        )
        trace = Trace(
            incident=incident,
            extracted_signals={"error_codes": [], "version": "2.0"},
            applicable_chunk_ids=["chunk-a", "chunk-b"],
            verification_verdicts=[verdict],
        )
        assert len(trace.applicable_chunk_ids) == 2
        assert trace.verification_verdicts[0].status == VerificationStatus.PASS

    def test_trace_ids_are_unique(self):
        """Two Trace objects get distinct trace_ids."""
        incident = TroubleshootingIncident(description="Test incident.")
        t1 = Trace(incident=incident)
        t2 = Trace(incident=incident)
        assert t1.trace_id != t2.trace_id


# ===========================================================================
# Tests 11–13 — FakeLLMClient
# ===========================================================================

class TestFakeLLMClient:
    # Test 11 — Fixed response
    def test_fixed_response_returned(self):
        """FakeLLMClient returns the configured deterministic response."""
        client = FakeLLMClient(response='{"diagnosis": "token format mismatch"}')
        resp = client.generate("What caused the AUTH_401?")
        assert resp.text == '{"diagnosis": "token format mismatch"}'
        assert resp.success is True

    def test_fixed_response_repeats(self):
        """A fixed response is returned on every call."""
        client = FakeLLMClient(response="always this")
        assert client.generate("q1").text == "always this"
        assert client.generate("q2").text == "always this"
        assert client.call_count == 2

    # Test 12 — Sequence of responses
    def test_sequence_responses_in_order(self):
        """FakeLLMClient returns predetermined responses in sequence."""
        client = FakeLLMClient(responses=["first response", "second response"])
        assert client.generate("prompt-1").text == "first response"
        assert client.generate("prompt-2").text == "second response"

    def test_sequence_exhausted_raises_llm_error(self):
        """Exhausting the response queue raises LLMError."""
        client = FakeLLMClient(responses=["only one"])
        client.generate("prompt-1")
        with pytest.raises(LLMError, match="exhausted"):
            client.generate("prompt-2")

    # Test 13 — Simulated failure
    def test_global_exception_raised(self):
        """FakeLLMClient raises the configured exception without network access."""
        client = FakeLLMClient(exception=LLMError("Gemini API timeout"))
        with pytest.raises(LLMError, match="timeout"):
            client.generate("any prompt")

    def test_exception_in_sequence(self):
        """An exception embedded in a response sequence is raised at the right position."""
        client = FakeLLMClient(
            responses=["ok response", LLMError("provider down")]
        )
        assert client.generate("q1").text == "ok response"
        with pytest.raises(LLMError, match="provider down"):
            client.generate("q2")

    def test_malformed_response_mode(self):
        """FakeLLMClient can return clearly malformed text for schema-failure tests."""
        client = FakeLLMClient(response="NOT_VALID_JSON {{{{")
        resp = client.generate("structured prompt")
        # The response is valid from the client's perspective — it's the
        # caller's job to detect the malformed content.
        assert "NOT_VALID_JSON" in resp.text

    def test_cannot_mix_exception_and_response(self):
        """Specifying both exception and response raises ValueError."""
        with pytest.raises(ValueError, match="not both"):
            FakeLLMClient(
                response="ok",
                exception=LLMError("fail"),
            )


# ===========================================================================
# Test 14 — Real Gemini client configuration (no real API calls)
# ===========================================================================

class TestGeminiClientConfiguration:
    def test_missing_api_key_raises_configuration_error(self, monkeypatch):
        """GeminiClient construction fails clearly when GEMINI_API_KEY is absent."""
        # Temporarily override the settings object to have no API key
        monkeypatch.setattr("src.llm.gemini.settings", DevTraceConfig(gemini_api_key=None))
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
            GeminiClient()

    def test_error_types_are_distinguishable(self):
        """The three error types are distinct and share a common base."""
        from src.errors import DevTraceError
        assert issubclass(ConfigurationError, DevTraceError)
        assert issubclass(LLMError, DevTraceError)
        assert issubclass(SchemaValidationError, DevTraceError)
        # They are distinct from each other
        assert not issubclass(LLMError, ConfigurationError)
        assert not issubclass(ConfigurationError, LLMError)


# ===========================================================================
# Test 15 — Configuration
# ===========================================================================

class TestConfiguration:
    def test_defaults_are_sane(self):
        """DevTraceConfig has sensible out-of-the-box defaults."""
        config = DevTraceConfig()
        assert config.gemini_model == "gemini-1.5-pro"
        assert config.gemini_temperature == pytest.approx(0.0)
        assert config.devtrace_env == "development"
        assert config.gemini_api_key is None

    def test_environment_overrides(self, monkeypatch):
        """Environment variables override default values."""
        monkeypatch.setenv("GEMINI_MODEL", "gemini-pro-vision")
        monkeypatch.setenv("GEMINI_TEMPERATURE", "0.2")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")

        config = DevTraceConfig()
        assert config.gemini_model == "gemini-pro-vision"
        assert config.gemini_temperature == pytest.approx(0.2)
        assert config.gemini_api_key == "test-key-123"

    def test_invalid_environment_value_rejected(self, monkeypatch):
        """An invalid environment value (e.g. out-of-range temperature) is rejected."""
        monkeypatch.setenv("GEMINI_TEMPERATURE", "5.0")
        with pytest.raises(Exception):
            DevTraceConfig()
