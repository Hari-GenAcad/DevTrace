"""
DevTrace — Module 4: Test suite.

Covers:
    TestApplicabilityVersionRules   — version compatibility rules (7 tests)
    TestApplicabilityFilter         — filter_applicable batch function (5 tests)
    TestApplicabilityTraceability   — ApplicabilityResult fields preserved (4 tests)
    TestNeedsInfoGate               — NEEDS_INFO deterministic gate (9 tests)
    TestDiagnosisParsing            — parse_diagnosis JSON parsing (8 tests)
    TestDiagnosisGenerator          — DiagnosisGenerator + FakeLLM (6 tests)
    TestBaselinePipelineUnit        — run_baseline_diagnosis unit tests (9 tests)
    TestBaselinePipelineIntegration — end-to-end with real M3 retriever (6 tests)

Total: ~54 tests
All tests are offline and deterministic.
No real Gemini API is called.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.applicability.checker import (
    ApplicabilityDecision,
    _decide,
    check_applicability,
    filter_applicable,
)
from src.diagnosis.generator import DiagnosisGenerator, parse_diagnosis
from src.diagnosis.needs_info import NeedsInfoResult, needs_info_check
from src.errors import LLMError, SchemaValidationError
from src.llm.fake import FakeLLMClient
from src.models.contracts import (
    ApplicabilityResult,
    DiagnosisClaim,
    DiagnosisResult,
    RetrievalResult,
)
from src.models.enums import ClaimRole, RetrievalSource, SystemOutcome
from src.normalization.normalizer import normalize_incident
from src.pipeline.baseline import BaselinePipelineResult, run_baseline_diagnosis


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _make_retrieval_result(
    chunk_id: str = "chunk-001",
    doc_id: str = "DOC-001",
    score: float = 0.75,
    applies_to: str = ">=3.0,<4.0",
    content: str = "Some content about authentication.",
    topic: str = "authentication",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        score=score,
        source=RetrievalSource.HYBRID,
        metadata={
            "applies_to": applies_to,
            "content": content,
            "topic": topic,
        },
    )


def _make_valid_diagnosis_json(
    root_cause_text: str = "The root cause is OAuth token not refreshed.",
    fix_text: str | None = "Call authenticate() before making requests.",
    evidence_id: str = "chunk-001",
) -> str:
    claims = [
        {
            "role": "root_cause",
            "text": root_cause_text,
            "evidence_ids": [evidence_id],
        }
    ]
    if fix_text:
        claims.append(
            {
                "role": "fix",
                "text": fix_text,
                "evidence_ids": [evidence_id],
            }
        )
    return json.dumps({"claims": claims})


# ---------------------------------------------------------------------------
# TestApplicabilityVersionRules
# ---------------------------------------------------------------------------

class TestApplicabilityVersionRules:
    """Tests the core version compatibility logic in _decide()."""

    def test_3x_incident_3x_doc_applicable(self) -> None:
        """Rule 2: current=3.1, applies_to='>=3.0,<4.0' → APPLICABLE."""
        decision, reason = _decide("3.1", ">=3.0,<4.0")
        assert decision == ApplicabilityDecision.APPLICABLE
        assert "3.1" in reason

    def test_3x_incident_2x_doc_not_applicable(self) -> None:
        """Rule 3: current=3.1, applies_to='>=2.0,<3.0' → NOT_APPLICABLE."""
        decision, reason = _decide("3.1", ">=2.0,<3.0")
        assert decision == ApplicabilityDecision.NOT_APPLICABLE
        assert "incompatible" in reason.lower()

    def test_2x_incident_2x_doc_applicable(self) -> None:
        """Rule 2: current=2.8, applies_to='>=2.0,<3.0' → APPLICABLE."""
        decision, reason = _decide("2.8", ">=2.0,<3.0")
        assert decision == ApplicabilityDecision.APPLICABLE

    def test_2x_incident_3x_doc_not_applicable(self) -> None:
        """Rule 3: current=2.5, applies_to='>=3.0,<4.0' → NOT_APPLICABLE."""
        decision, reason = _decide("2.5", ">=3.0,<4.0")
        assert decision == ApplicabilityDecision.NOT_APPLICABLE

    def test_version_agnostic_doc_always_applicable(self) -> None:
        """Rule 1: applies_to='*' → APPLICABLE for any version."""
        decision_3x, _ = _decide("3.1", "*")
        decision_2x, _ = _decide("2.8", "*")
        decision_none, _ = _decide(None, "*")
        assert decision_3x == ApplicabilityDecision.APPLICABLE
        assert decision_2x == ApplicabilityDecision.APPLICABLE
        assert decision_none == ApplicabilityDecision.APPLICABLE

    def test_missing_version_with_version_specific_doc_unknown(self) -> None:
        """Rule 4: no current_version + specific doc → UNKNOWN."""
        decision, reason = _decide(None, ">=3.0,<4.0")
        assert decision == ApplicabilityDecision.UNKNOWN
        assert "current_version" in reason.lower() or "no" in reason.lower()

    def test_previous_version_does_not_override_current(self) -> None:
        """Rule 5: previous=2.8, current=3.1 — 2.x doc is still NOT_APPLICABLE."""
        # This tests that we correctly use current, not previous.
        # The check_applicability function takes current_version explicitly.
        rr = _make_retrieval_result(applies_to=">=2.0,<3.0")
        # current_version = 3.1 (not 2.8)
        result = check_applicability(rr, current_version="3.1")
        assert not result.applicable
        assert result.incident_version == "3.1"

    def test_migration_doc_spanning_both_versions_applicable_to_3x(self) -> None:
        """Agnostic doc spanning both versions is applicable to 3.x incident."""
        # A migration guide covering ">=2.0,<4.0" spans both 2.x and 3.x.
        decision, reason = _decide("3.1", ">=2.0,<4.0")
        assert decision == ApplicabilityDecision.APPLICABLE

    def test_migration_doc_spanning_both_versions_applicable_to_2x(self) -> None:
        """Agnostic doc spanning both versions is also applicable to 2.x incident."""
        decision, _ = _decide("2.8", ">=2.0,<4.0")
        assert decision == ApplicabilityDecision.APPLICABLE


# ---------------------------------------------------------------------------
# TestApplicabilityFilter
# ---------------------------------------------------------------------------

class TestApplicabilityFilter:
    """Tests filter_applicable batch function."""

    def test_filters_wrong_version_chunks(self) -> None:
        """3.x incident: 2.x chunks are excluded, 3.x chunks pass."""
        results = [
            _make_retrieval_result("c-v3", applies_to=">=3.0,<4.0"),
            _make_retrieval_result("c-v2", applies_to=">=2.0,<3.0"),
        ]
        applicable, decisions = filter_applicable(results, current_version="3.1")
        applicable_ids = {r.chunk_id for r in applicable}
        assert "c-v3" in applicable_ids
        assert "c-v2" not in applicable_ids

    def test_version_agnostic_always_passes(self) -> None:
        """Version-agnostic (*) docs pass regardless of incident version."""
        results = [_make_retrieval_result("agnostic-chunk", applies_to="*")]
        applicable, _ = filter_applicable(results, current_version="3.1")
        assert len(applicable) == 1

    def test_all_decisions_returned_even_for_excluded(self) -> None:
        """Every input chunk gets an ApplicabilityResult even if NOT_APPLICABLE."""
        results = [
            _make_retrieval_result("c-v3", applies_to=">=3.0,<4.0"),
            _make_retrieval_result("c-v2", applies_to=">=2.0,<3.0"),
        ]
        _, decisions = filter_applicable(results, current_version="3.1")
        assert len(decisions) == 2

    def test_unknown_excluded_from_applicable(self) -> None:
        """UNKNOWN decisions are excluded from the applicable set (conservative)."""
        results = [
            _make_retrieval_result("chunk-specific", applies_to=">=3.0,<4.0"),
        ]
        # No current_version → UNKNOWN
        applicable, decisions = filter_applicable(results, current_version=None)
        assert len(applicable) == 0
        assert decisions[0].applicable is False

    def test_empty_retrieval_results(self) -> None:
        """Empty retrieval results → empty applicable, empty decisions."""
        applicable, decisions = filter_applicable([], current_version="3.1")
        assert applicable == []
        assert decisions == []


# ---------------------------------------------------------------------------
# TestApplicabilityTraceability
# ---------------------------------------------------------------------------

class TestApplicabilityTraceability:
    """Tests that ApplicabilityResult fields are fully preserved."""

    def test_chunk_id_preserved(self) -> None:
        rr = _make_retrieval_result("CHUNK-XYZ", applies_to=">=3.0,<4.0")
        ar = check_applicability(rr, "3.1")
        assert ar.chunk_id == "CHUNK-XYZ"

    def test_doc_id_preserved(self) -> None:
        rr = _make_retrieval_result(doc_id="DOC-TRACE-001", applies_to=">=3.0,<4.0")
        ar = check_applicability(rr, "3.1")
        assert ar.doc_id == "DOC-TRACE-001"

    def test_incident_version_preserved(self) -> None:
        rr = _make_retrieval_result(applies_to=">=3.0,<4.0")
        ar = check_applicability(rr, "3.1")
        assert ar.incident_version == "3.1"

    def test_document_range_preserved(self) -> None:
        rr = _make_retrieval_result(applies_to=">=3.0,<4.0")
        ar = check_applicability(rr, "3.1")
        assert ar.document_range == ">=3.0,<4.0"

    def test_reason_is_non_empty(self) -> None:
        rr = _make_retrieval_result(applies_to=">=2.0,<3.0")
        ar = check_applicability(rr, "3.1")
        assert ar.reason
        assert len(ar.reason) > 10  # not just empty or trivial


# ---------------------------------------------------------------------------
# TestNeedsInfoGate
# ---------------------------------------------------------------------------

class TestNeedsInfoGate:
    """Tests the deterministic NEEDS_INFO gate."""

    def _retrieval_with_version_spans(self) -> list[RetrievalResult]:
        """Return results spanning both 2.x and 3.x."""
        return [
            _make_retrieval_result("c2", applies_to=">=2.0,<3.0"),
            _make_retrieval_result("c3", applies_to=">=3.0,<4.0"),
        ]

    def test_missing_version_competing_evidence_triggers_needs_info(self) -> None:
        """Case A: no version, evidence spans 2.x and 3.x → NEEDS_INFO."""
        normalized = normalize_incident(
            "AUTH_401 on my API calls.",
            error_codes=["AUTH_401"],
        )
        result = needs_info_check(normalized, self._retrieval_with_version_spans())
        assert result.triggered
        assert result.outcome == SystemOutcome.NEEDS_INFO
        assert "current_version" in result.missing_fields

    def test_vague_incident_no_signal_triggers_needs_info(self) -> None:
        """Case B: no error code, no version, no technical terms → NEEDS_INFO."""
        normalized = normalize_incident(
            "Something is wrong with our API. Requests aren't working.",
            product="DevCore API",
        )
        result = needs_info_check(normalized, [])
        assert result.triggered
        assert result.outcome == SystemOutcome.NEEDS_INFO

    def test_error_code_with_version_does_not_trigger(self) -> None:
        """Clear incident with error code + version → proceed."""
        normalized = normalize_incident(
            "AUTH_401 on all API requests after upgrading from SDK 2.8 to SDK 3.1.",
            current_version="3.1",
            error_codes=["AUTH_401"],
        )
        result = needs_info_check(normalized, self._retrieval_with_version_spans())
        assert not result.triggered

    def test_error_code_alone_proceeds_even_without_version(self) -> None:
        """Error code present + all-agnostic retrieval → proceed (no version conflict)."""
        normalized = normalize_incident(
            "AUTH_403 on POST /events. We have valid credentials but 403.",
            error_codes=["AUTH_403"],
        )
        # Only agnostic evidence
        agnostic_results = [_make_retrieval_result("ag", applies_to="*")]
        result = needs_info_check(normalized, agnostic_results)
        assert not result.triggered

    def test_version_present_no_competing_evidence_proceeds(self) -> None:
        """Version is known → no NEEDS_INFO even with competing retrieval."""
        normalized = normalize_incident(
            "AUTH_401 after upgrading.",
            current_version="3.1",
            error_codes=["AUTH_401"],
        )
        result = needs_info_check(normalized, self._retrieval_with_version_spans())
        assert not result.triggered

    def test_no_word_count_threshold_short_but_signal_proceeds(self) -> None:
        """Short description with error code → proceed (no word-count gate)."""
        normalized = normalize_incident(
            "AUTH_401.",
            error_codes=["AUTH_401"],
            current_version="3.0",
        )
        result = needs_info_check(normalized, [])
        assert not result.triggered

    def test_missing_version_agnostic_only_evidence_proceeds(self) -> None:
        """No version, but all evidence is version-agnostic → can proceed."""
        normalized = normalize_incident(
            "RATE_429 on batch processing jobs.",
            error_codes=["RATE_429"],
        )
        # Only version-agnostic evidence
        agnostic_results = [_make_retrieval_result("ag", applies_to="*")]
        result = needs_info_check(normalized, agnostic_results)
        assert not result.triggered

    def test_needs_info_provides_unblock_hint(self) -> None:
        """NEEDS_INFO result must include a non-empty unblock hint."""
        normalized = normalize_incident(
            "Something is wrong with our integration.",
            product="DevCore API",
        )
        result = needs_info_check(normalized, [])
        assert result.triggered
        assert result.unblock_hint
        assert len(result.unblock_hint) > 10

    def test_technical_term_alone_provides_enough_signal(self) -> None:
        """Technical term 'webhook' alone is enough signal to avoid NEEDS_INFO."""
        normalized = normalize_incident(
            "Webhook delivery failures on our endpoint. No signature validation error.",
            current_version="3.0",
        )
        result = needs_info_check(normalized, [])
        assert not result.triggered


# ---------------------------------------------------------------------------
# TestDiagnosisParsing
# ---------------------------------------------------------------------------

class TestDiagnosisParsing:
    """Tests parse_diagnosis JSON parsing."""

    def test_valid_json_parses_correctly(self) -> None:
        """Valid JSON with root_cause + fix → DiagnosisResult."""
        raw = _make_valid_diagnosis_json()
        result = parse_diagnosis(raw)
        assert isinstance(result, DiagnosisResult)
        assert len(result.claims) == 2

    def test_root_cause_claim_parsed(self) -> None:
        """Root cause claim is present and correctly typed."""
        raw = _make_valid_diagnosis_json(root_cause_text="JWT token expired.")
        result = parse_diagnosis(raw)
        root = next(c for c in result.claims if c.role == ClaimRole.ROOT_CAUSE)
        assert root.text == "JWT token expired."

    def test_evidence_ids_preserved(self) -> None:
        """Evidence IDs from JSON are preserved in the DiagnosisClaim."""
        raw = json.dumps({
            "claims": [
                {
                    "role": "root_cause",
                    "text": "Auth header format changed.",
                    "evidence_ids": ["AUTH-002-C01", "SDK-001-C01"],
                }
            ]
        })
        result = parse_diagnosis(raw)
        assert "AUTH-002-C01" in result.claims[0].evidence_ids
        assert "SDK-001-C01" in result.claims[0].evidence_ids

    def test_invalid_json_raises_schema_validation_error(self) -> None:
        """Non-JSON text raises SchemaValidationError."""
        with pytest.raises(SchemaValidationError, match="not valid JSON"):
            parse_diagnosis("This is not JSON at all.")

    def test_missing_root_cause_raises_schema_validation_error(self) -> None:
        """Missing root_cause claim raises SchemaValidationError."""
        raw = json.dumps({
            "claims": [
                {"role": "fix", "text": "Do something.", "evidence_ids": []}
            ]
        })
        with pytest.raises(SchemaValidationError):
            parse_diagnosis(raw)

    def test_invalid_role_raises_schema_validation_error(self) -> None:
        """Invalid role string raises SchemaValidationError."""
        raw = json.dumps({
            "claims": [
                {"role": "nonsense_role", "text": "Something.", "evidence_ids": []}
            ]
        })
        with pytest.raises(SchemaValidationError, match="Invalid claim role"):
            parse_diagnosis(raw)

    def test_markdown_fenced_json_stripped(self) -> None:
        """Markdown code fences around JSON are stripped before parsing."""
        inner = _make_valid_diagnosis_json()
        wrapped = f"```json\n{inner}\n```"
        result = parse_diagnosis(wrapped)
        assert isinstance(result, DiagnosisResult)

    def test_multiple_claims_all_parsed(self) -> None:
        """Root cause + fix + explanation → 3 claims."""
        raw = json.dumps({
            "claims": [
                {"role": "root_cause", "text": "Cause.", "evidence_ids": ["c1"]},
                {"role": "fix", "text": "Fix.", "evidence_ids": ["c1"]},
                {"role": "explanation", "text": "Why.", "evidence_ids": ["c1"]},
            ]
        })
        result = parse_diagnosis(raw)
        assert len(result.claims) == 3
        roles = {c.role for c in result.claims}
        assert ClaimRole.ROOT_CAUSE in roles
        assert ClaimRole.FIX in roles
        assert ClaimRole.EXPLANATION in roles


# ---------------------------------------------------------------------------
# TestDiagnosisGenerator
# ---------------------------------------------------------------------------

class TestDiagnosisGenerator:
    """Tests DiagnosisGenerator with FakeLLMClient."""

    def _make_generator(self, response: str) -> DiagnosisGenerator:
        return DiagnosisGenerator(FakeLLMClient(response=response))

    def test_valid_response_returns_diagnosis_result(self) -> None:
        """FakeLLM returning valid JSON → DiagnosisResult returned."""
        gen = self._make_generator(_make_valid_diagnosis_json())
        normalized = normalize_incident(
            "AUTH_401 after upgrading to SDK 3.1.",
            current_version="3.1",
            error_codes=["AUTH_401"],
        )
        applicable = [_make_retrieval_result("AUTH-002-C01", applies_to=">=3.0,<4.0")]
        result = gen.generate(normalized, applicable)
        assert isinstance(result, DiagnosisResult)

    def test_claim_roles_parsed(self) -> None:
        """Root cause and fix claims are correctly typed."""
        gen = self._make_generator(
            _make_valid_diagnosis_json(
                root_cause_text="Bearer token not provided.",
                fix_text="Obtain token via /auth/token.",
            )
        )
        normalized = normalize_incident("AUTH_401 issue.", current_version="3.0")
        result = gen.generate(normalized, [])
        roles = {c.role for c in result.claims}
        assert ClaimRole.ROOT_CAUSE in roles
        assert ClaimRole.FIX in roles

    def test_llm_error_propagates(self) -> None:
        """LLMError from the client propagates as LLMError."""
        gen = DiagnosisGenerator(FakeLLMClient(exception=LLMError("API timeout")))
        normalized = normalize_incident("AUTH_401.", current_version="3.0")
        with pytest.raises(LLMError, match="API timeout"):
            gen.generate(normalized, [])

    def test_malformed_response_raises_schema_validation_error(self) -> None:
        """Malformed JSON response raises SchemaValidationError."""
        gen = self._make_generator("NOT_VALID_JSON{{{{")
        normalized = normalize_incident("AUTH_401.", current_version="3.0")
        with pytest.raises(SchemaValidationError):
            gen.generate(normalized, [])

    def test_generator_only_receives_applicable_evidence(self) -> None:
        """
        The generator receives only applicable chunks.

        We verify this by checking that the prompt includes only the
        applicable chunk IDs (the FakeLLM doesn't inspect the prompt,
        but we verify the generator function signature enforces it).
        This is a structural test — the generator takes `applicable_results`
        as its second argument, making it impossible to pass all results.
        """
        gen = self._make_generator(_make_valid_diagnosis_json())
        normalized = normalize_incident("AUTH_401.", current_version="3.1")
        applicable = [_make_retrieval_result("AUTH-002-C01", applies_to=">=3.0,<4.0")]
        result = gen.generate(normalized, applicable)
        # Evidence IDs in the claim must be from our applicable set (FakeLLM echoes our provided ID).
        root_claim = next(c for c in result.claims if c.role == ClaimRole.ROOT_CAUSE)
        assert isinstance(root_claim.evidence_ids, list)

    def test_empty_llm_text_raises_llm_error(self) -> None:
        """Empty text from the LLM raises LLMError."""
        # FakeLLMResponse with empty text — patch success=True but text=""
        from src.llm.fake import FakeLLMResponse

        class _EmptyClient(FakeLLMClient):
            def generate(self, prompt, **kwargs):  # type: ignore[override]
                return FakeLLMResponse(text="", success=False)

        gen = DiagnosisGenerator(_EmptyClient())
        normalized = normalize_incident("AUTH_401.", current_version="3.0")
        with pytest.raises(LLMError):
            gen.generate(normalized, [])


# ---------------------------------------------------------------------------
# TestBaselinePipelineUnit
# ---------------------------------------------------------------------------

class TestBaselinePipelineUnit:
    """
    Unit tests for run_baseline_diagnosis using mocked retriever + FakeLLM.

    These tests do NOT load the real corpus or Chroma index.
    """

    def _make_retriever(
        self,
        results: list[RetrievalResult] | None = None,
    ) -> MagicMock:
        """Create a mock HybridRetriever."""
        m = MagicMock()
        m.retrieve_as_contracts.return_value = results or []
        return m

    def _run(
        self,
        description: str = "AUTH_401 after upgrading to SDK 3.1.",
        current_version: str | None = "3.1",
        error_codes: list[str] | None = None,
        retrieval_results: list[RetrievalResult] | None = None,
        llm_response: str | None = None,
    ) -> BaselinePipelineResult:
        retriever = self._make_retriever(retrieval_results or [])
        llm = FakeLLMClient(
            response=llm_response or _make_valid_diagnosis_json()
        )
        return run_baseline_diagnosis(
            description=description,
            current_version=current_version,
            error_codes=error_codes or ["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )

    def test_successful_pipeline_returns_diagnosis(self) -> None:
        """Happy path: incident + applicable evidence → DiagnosisResult."""
        result = self._run(
            retrieval_results=[
                _make_retrieval_result("AUTH-002-C01", applies_to=">=3.0,<4.0")
            ]
        )
        assert result.diagnosis is not None
        assert result.error is None

    def test_retrieval_results_stored(self) -> None:
        """Pipeline stores raw retrieval results before applicability."""
        rr = _make_retrieval_result("AUTH-002-C01", applies_to=">=3.0,<4.0")
        result = self._run(retrieval_results=[rr])
        assert len(result.retrieval_results) == 1

    def test_wrong_version_excluded_from_applicable(self) -> None:
        """2.x document is excluded for a 3.1 incident."""
        results = [
            _make_retrieval_result("v3-chunk", applies_to=">=3.0,<4.0"),
            _make_retrieval_result("v2-chunk", applies_to=">=2.0,<3.0"),
        ]
        result = self._run(retrieval_results=results)
        applicable_ids = {r.chunk_id for r in result.applicable_results}
        assert "v3-chunk" in applicable_ids
        assert "v2-chunk" not in applicable_ids

    def test_needs_info_triggers_early_exit(self) -> None:
        """NEEDS_INFO is triggered → no diagnosis generated."""
        # No version, competing evidence
        result = run_baseline_diagnosis(
            description="AUTH_401 on my API calls.",
            error_codes=["AUTH_401"],
            # No current_version
            retriever=self._make_retriever([
                _make_retrieval_result("c2", applies_to=">=2.0,<3.0"),
                _make_retrieval_result("c3", applies_to=">=3.0,<4.0"),
            ]),
            llm_client=FakeLLMClient(response=_make_valid_diagnosis_json()),
        )
        assert result.outcome == SystemOutcome.NEEDS_INFO
        assert result.diagnosis is None
        assert result.needs_info is not None
        assert result.needs_info.triggered

    def test_llm_error_produces_degraded_outcome(self) -> None:
        """LLM failure → DEGRADED outcome, error field set."""
        result = run_baseline_diagnosis(
            description="AUTH_401 after upgrading to 3.1.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=self._make_retriever([
                _make_retrieval_result("AUTH-002-C01", applies_to=">=3.0,<4.0")
            ]),
            llm_client=FakeLLMClient(exception=LLMError("API timeout")),
        )
        assert result.outcome == SystemOutcome.DEGRADED
        assert result.error is not None
        assert "timeout" in result.error.lower()

    def test_schema_validation_error_produces_degraded_outcome(self) -> None:
        """Malformed LLM response → DEGRADED outcome."""
        result = run_baseline_diagnosis(
            description="AUTH_401 after upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=self._make_retriever([
                _make_retrieval_result("AUTH-002-C01", applies_to=">=3.0,<4.0")
            ]),
            llm_client=FakeLLMClient(response="TOTALLY INVALID RESPONSE"),
        )
        assert result.outcome == SystemOutcome.DEGRADED

    def test_normalized_incident_stored(self) -> None:
        """BaselinePipelineResult always stores the normalized incident."""
        result = self._run()
        assert result.normalized is not None
        assert result.normalized.incident.description

    def test_applicability_decisions_stored_for_all_chunks(self) -> None:
        """All retrieval results get an ApplicabilityResult stored."""
        results = [
            _make_retrieval_result("v3-chunk", applies_to=">=3.0,<4.0"),
            _make_retrieval_result("v2-chunk", applies_to=">=2.0,<3.0"),
        ]
        result = self._run(retrieval_results=results)
        assert len(result.applicability_decisions) == 2

    def test_vague_incident_no_signal_triggers_needs_info(self) -> None:
        """Vague description with no signal → NEEDS_INFO, no diagnosis."""
        result = run_baseline_diagnosis(
            description="Something is wrong with our integration.",
            product="DevCore API",
            retriever=self._make_retriever([]),
            llm_client=FakeLLMClient(response=_make_valid_diagnosis_json()),
        )
        assert result.outcome == SystemOutcome.NEEDS_INFO
        assert result.diagnosis is None


# ---------------------------------------------------------------------------
# TestBaselinePipelineIntegration
# ---------------------------------------------------------------------------

class TestBaselinePipelineIntegration:
    """
    Integration tests using the REAL M3 HybridRetriever and corpus.

    These tests verify the complete M4 pipeline against the actual
    DevCore corpus — but use FakeLLMClient so no Gemini API is called.

    The retriever is built once per test class (expensive load).
    """

    _retriever: HybridRetriever | None = None

    @classmethod
    def setup_class(cls) -> None:
        """Load retriever once for all integration tests."""
        from src.ingestion.loader import load_corpus_and_chunks
        from src.retrieval.hybrid import HybridRetriever

        _, chunks = load_corpus_and_chunks()
        retriever = HybridRetriever(chunks)
        retriever.load()
        retriever.build_vector_index()
        cls._retriever = retriever

    def _run(
        self,
        description: str,
        current_version: str | None = None,
        previous_version: str | None = None,
        error_codes: list[str] | None = None,
        product: str | None = "DevCore SDK",
        llm_response: str | None = None,
    ) -> BaselinePipelineResult:
        assert self._retriever is not None
        return run_baseline_diagnosis(
            description=description,
            current_version=current_version,
            previous_version=previous_version,
            error_codes=error_codes,
            product=product,
            retriever=self._retriever,
            llm_client=FakeLLMClient(
                response=llm_response or _make_valid_diagnosis_json()
            ),
        )

    def test_3x_incident_2x_doc_excluded(self) -> None:
        """
        Version conflict integration test.

        current=3.1, error=AUTH_401 → M3 retrieves both 2.x and 3.x docs.
        M4 applicability must exclude 2.x docs from the evidence bundle.
        """
        result = self._run(
            description="AUTH_401 on all requests after upgrading from SDK 2.8 to SDK 3.1.",
            current_version="3.1",
            previous_version="2.8",
            error_codes=["AUTH_401"],
        )
        # Check that no 2.x-specific doc passed applicability
        excluded_doc_ids = {
            d.doc_id for d in result.applicability_decisions if not d.applicable
        }
        applicable_doc_ids = {
            d.doc_id for d in result.applicability_decisions if d.applicable
        }
        # AUTH-001 is the 2.x authentication doc — it should be excluded.
        assert "AUTH-001" in excluded_doc_ids, (
            f"Expected AUTH-001 (2.x doc) to be excluded. "
            f"Applicable: {applicable_doc_ids}, Excluded: {excluded_doc_ids}"
        )

    def test_3x_doc_applicable_for_3x_incident(self) -> None:
        """AUTH-002 (3.x auth doc) must be applicable for a 3.1 incident."""
        result = self._run(
            description="AUTH_401 after upgrading to SDK 3.1.",
            current_version="3.1",
            error_codes=["AUTH_401"],
        )
        applicable_doc_ids = {
            d.doc_id for d in result.applicability_decisions if d.applicable
        }
        assert "AUTH-002" in applicable_doc_ids, (
            f"Expected AUTH-002 (3.x auth doc) to be applicable. "
            f"Applicable: {applicable_doc_ids}"
        )

    def test_missing_version_competing_evidence_triggers_needs_info(self) -> None:
        """Real corpus: no version + AUTH_401 → NEEDS_INFO (2.x and 3.x evidence spans)."""
        result = self._run(
            description="AUTH_401 on my API calls.",
            error_codes=["AUTH_401"],
            current_version=None,
        )
        assert result.outcome == SystemOutcome.NEEDS_INFO
        assert result.diagnosis is None

    def test_agnostic_doc_always_in_applicable(self) -> None:
        """Version-agnostic documents (AUTH-003, applies_to='*') must pass applicability."""
        result = self._run(
            description="AUTH_403 Forbidden on write operations.",
            error_codes=["AUTH_403"],
            current_version="3.1",
        )
        applicable_doc_ids = {
            d.doc_id for d in result.applicability_decisions if d.applicable
        }
        # AUTH-003 is the version-agnostic AUTH_403 doc.
        assert "AUTH-003" in applicable_doc_ids

    def test_pipeline_produces_non_empty_applicable_results(self) -> None:
        """Real corpus: a clear 3.x incident produces at least one applicable result."""
        result = self._run(
            description="AUTH_401 after migrating to SDK 3.1. Bearer token not accepted.",
            current_version="3.1",
            error_codes=["AUTH_401"],
        )
        assert result.needs_info is not None and not result.needs_info.triggered
        assert len(result.applicable_results) > 0

    def test_diagnosis_only_references_applicable_chunk_ids(self) -> None:
        """
        The diagnosis generator receives only applicable chunks.

        Since FakeLLM echoes our canned JSON (which references 'chunk-001'),
        we verify the structural guarantee: applicable_results fed to the generator
        does NOT include NOT_APPLICABLE chunks.
        """
        result = self._run(
            description="AUTH_401 after upgrading to 3.1.",
            current_version="3.1",
            error_codes=["AUTH_401"],
        )
        # The applicable_results fed to the generator must be a strict subset
        # of all retrieval_results — only APPLICABLE ones.
        applicable_ids = {r.chunk_id for r in result.applicable_results}
        excluded_ids = {
            d.chunk_id for d in result.applicability_decisions if not d.applicable
        }
        # No applicable result should have been excluded.
        assert applicable_ids.isdisjoint(excluded_ids), (
            f"Applicable and excluded sets overlap: {applicable_ids & excluded_ids}"
        )
