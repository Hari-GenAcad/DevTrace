"""
DevTrace — Module 5: Test suite.

Covers:
    TestCitationValidityDeterministic  — citation validity checks (10 tests)
    TestVerifierResponseParsing        — _parse_verifier_response (8 tests)
    TestClaimVerificationLogic         — _verify_claim unit tests (10 tests)
    TestVerifyDiagnosis                — verify_diagnosis integration (8 tests)
    TestWrongVersionExclusion          — architectural property test (4 tests)
    TestVerifiedPipeline               — run_verified_diagnosis integration (8 tests)

Total: ~48 tests
All tests are offline and deterministic.
No real Gemini API is called.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.errors import LLMError, SchemaValidationError
from src.llm.fake import FakeLLMClient
from src.models.contracts import (
    DiagnosisClaim,
    DiagnosisResult,
    RetrievalResult,
)
from src.models.enums import ClaimRole, RetrievalSource, SystemOutcome
from src.normalization.normalizer import normalize_incident
from src.verification.citation import (
    build_applicable_id_set,
    check_citation_validity,
)
from src.verification.models import (
    CitationValidity,
    ClaimVerification,
    DiagnosisVerification,
    VerificationVerdict,
)
from src.verification.verifier import (
    _parse_verifier_response,
    _verify_claim,
    verify_diagnosis,
)
from src.pipeline.verified import VerifiedPipelineResult, run_verified_diagnosis


# ---------------------------------------------------------------------------
# Shared helpers / factories
# ---------------------------------------------------------------------------

def _make_retrieval_result(
    chunk_id: str = "chunk-001",
    doc_id: str = "DOC-001",
    score: float = 0.80,
    applies_to: str = ">=3.0,<4.0",
    content: str = "SDK 3.x requires OAuth Bearer token authentication.",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        score=score,
        source=RetrievalSource.HYBRID,
        metadata={
            "applies_to": applies_to,
            "content": content,
        },
    )


def _make_claim(
    role: ClaimRole = ClaimRole.ROOT_CAUSE,
    text: str = "SDK 3.x requires Bearer token for all API calls.",
    evidence_ids: list[str] | None = None,
) -> DiagnosisClaim:
    return DiagnosisClaim(
        role=role,
        text=text,
        evidence_ids=evidence_ids if evidence_ids is not None else ["chunk-001"],
    )


def _make_diagnosis(claims: list[DiagnosisClaim] | None = None) -> DiagnosisResult:
    """Build a DiagnosisResult with at least one root_cause claim."""
    if claims is None:
        claims = [_make_claim(role=ClaimRole.ROOT_CAUSE)]
    return DiagnosisResult(claims=claims)


def _make_verifier_json(
    citation_correct: bool = True,
    sufficient: bool = True,
    contradicted: bool = False,
    supporting_ids: list[str] | None = None,
    contradicting_ids: list[str] | None = None,
    reason: str = "The evidence directly supports the claim.",
) -> str:
    """Build a valid verifier JSON response string."""
    return json.dumps({
        "citation_correct": citation_correct,
        "sufficient": sufficient,
        "contradicted": contradicted,
        "supporting_evidence_ids": supporting_ids if supporting_ids is not None else ["chunk-001"],
        "contradicting_evidence_ids": contradicting_ids or [],
        "reason": reason,
    })


def _make_retriever(
    results: list[RetrievalResult] | None = None,
) -> MagicMock:
    m = MagicMock()
    m.retrieve_as_contracts.return_value = results or []
    return m


# ---------------------------------------------------------------------------
# TestCitationValidityDeterministic
# ---------------------------------------------------------------------------

class TestCitationValidityDeterministic:
    """Tests the deterministic citation validity checker in citation.py."""

    def test_valid_citation_single_id(self) -> None:
        """A claim citing a chunk_id that exists in the bundle → VALID."""
        rr = _make_retrieval_result("AUTH-002-C01")
        applicable_set = build_applicable_id_set([rr])
        claim = _make_claim(evidence_ids=["AUTH-002-C01"])
        validity, invalid = check_citation_validity(claim, applicable_set)
        assert validity == CitationValidity.VALID
        assert invalid == []

    def test_invalid_citation_hallucinated_id(self) -> None:
        """A claim citing a nonexistent chunk_id → INVALID."""
        rr = _make_retrieval_result("AUTH-002-C01")
        applicable_set = build_applicable_id_set([rr])
        claim = _make_claim(evidence_ids=["FAKE-CHUNK-999"])
        validity, invalid = check_citation_validity(claim, applicable_set)
        assert validity == CitationValidity.INVALID
        assert "FAKE-CHUNK-999" in invalid

    def test_empty_citation_list(self) -> None:
        """A claim with no evidence_ids → EMPTY."""
        rr = _make_retrieval_result("AUTH-002-C01")
        applicable_set = build_applicable_id_set([rr])
        claim = _make_claim(evidence_ids=[])
        validity, invalid = check_citation_validity(claim, applicable_set)
        assert validity == CitationValidity.EMPTY
        assert invalid == []

    def test_valid_citation_multiple_ids(self) -> None:
        """Multiple cited IDs, all present → VALID."""
        chunks = [
            _make_retrieval_result("AUTH-002-C01"),
            _make_retrieval_result("SDK-001-C01"),
        ]
        applicable_set = build_applicable_id_set(chunks)
        claim = _make_claim(evidence_ids=["AUTH-002-C01", "SDK-001-C01"])
        validity, invalid = check_citation_validity(claim, applicable_set)
        assert validity == CitationValidity.VALID
        assert invalid == []

    def test_partially_invalid_citation(self) -> None:
        """One valid + one invalid ID → INVALID, only the bad ID listed."""
        rr = _make_retrieval_result("AUTH-002-C01")
        applicable_set = build_applicable_id_set([rr])
        claim = _make_claim(evidence_ids=["AUTH-002-C01", "HALLUCINATED-ID"])
        validity, invalid = check_citation_validity(claim, applicable_set)
        assert validity == CitationValidity.INVALID
        assert "HALLUCINATED-ID" in invalid
        assert "AUTH-002-C01" not in invalid

    def test_empty_applicable_bundle(self) -> None:
        """Any cited ID against an empty bundle → INVALID."""
        applicable_set = build_applicable_id_set([])
        claim = _make_claim(evidence_ids=["chunk-001"])
        validity, invalid = check_citation_validity(claim, applicable_set)
        assert validity == CitationValidity.INVALID
        assert "chunk-001" in invalid

    def test_build_applicable_id_set(self) -> None:
        """build_applicable_id_set returns a set of all chunk_ids."""
        chunks = [
            _make_retrieval_result("c1"),
            _make_retrieval_result("c2"),
            _make_retrieval_result("c3"),
        ]
        id_set = build_applicable_id_set(chunks)
        assert id_set == {"c1", "c2", "c3"}

    def test_build_applicable_id_set_empty(self) -> None:
        """Empty applicable results → empty set."""
        assert build_applicable_id_set([]) == set()

    def test_non_applicable_chunk_not_in_applicable_set(self) -> None:
        """
        A chunk that was NOT_APPLICABLE in M4 must not appear in the applicable set.

        This test simulates the M4 boundary: applicable_results only contains
        APPLICABLE chunks (NOT_APPLICABLE chunks are filtered before reaching M5).
        """
        # Simulate M4: only 3.x chunk passed applicability
        applicable_chunks = [_make_retrieval_result("AUTH-002-C01", applies_to=">=3.0,<4.0")]
        applicable_set = build_applicable_id_set(applicable_chunks)

        # The 2.x chunk was excluded by M4 and therefore not in applicable_results
        claim_citing_2x = _make_claim(evidence_ids=["AUTH-001-C01"])  # 2.x chunk
        validity, invalid = check_citation_validity(claim_citing_2x, applicable_set)
        assert validity == CitationValidity.INVALID
        assert "AUTH-001-C01" in invalid

    def test_multiple_invalid_ids_all_reported(self) -> None:
        """All invalid IDs are reported, not just the first."""
        applicable_set = build_applicable_id_set([_make_retrieval_result("real-chunk")])
        claim = _make_claim(evidence_ids=["fake-1", "fake-2", "fake-3"])
        validity, invalid = check_citation_validity(claim, applicable_set)
        assert validity == CitationValidity.INVALID
        assert set(invalid) == {"fake-1", "fake-2", "fake-3"}


# ---------------------------------------------------------------------------
# TestVerifierResponseParsing
# ---------------------------------------------------------------------------

class TestVerifierResponseParsing:
    """Tests the _parse_verifier_response JSON parser."""

    def test_valid_json_parses_correctly(self) -> None:
        """Valid verifier JSON → dict with all required keys."""
        raw = _make_verifier_json(citation_correct=True, sufficient=True)
        result = _parse_verifier_response(raw)
        assert result["citation_correct"] is True
        assert result["sufficient"] is True
        assert result["contradicted"] is False

    def test_invalid_json_raises_schema_error(self) -> None:
        """Non-JSON text → SchemaValidationError."""
        with pytest.raises(SchemaValidationError, match="not valid JSON"):
            _parse_verifier_response("This is not JSON.")

    def test_missing_required_key_raises_schema_error(self) -> None:
        """JSON missing 'citation_correct' → SchemaValidationError."""
        incomplete = json.dumps({
            "sufficient": True,
            "contradicted": False,
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "reason": "test",
        })
        with pytest.raises(SchemaValidationError, match="missing required keys"):
            _parse_verifier_response(incomplete)

    def test_markdown_fences_stripped(self) -> None:
        """Markdown-fenced JSON is parsed correctly."""
        inner = _make_verifier_json()
        wrapped = f"```json\n{inner}\n```"
        result = _parse_verifier_response(wrapped)
        assert "citation_correct" in result

    def test_string_boolean_coercion(self) -> None:
        """String 'true'/'false' values are coerced to booleans."""
        raw = json.dumps({
            "citation_correct": "true",
            "sufficient": "false",
            "contradicted": "true",
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "reason": "test",
        })
        result = _parse_verifier_response(raw)
        assert result["citation_correct"] is True
        assert result["sufficient"] is False
        assert result["contradicted"] is True

    def test_non_list_evidence_ids_coerced(self) -> None:
        """Non-list evidence IDs are coerced to empty list."""
        raw = json.dumps({
            "citation_correct": True,
            "sufficient": True,
            "contradicted": False,
            "supporting_evidence_ids": "chunk-001",  # string, not list
            "contradicting_evidence_ids": None,
            "reason": "test",
        })
        result = _parse_verifier_response(raw)
        assert isinstance(result["supporting_evidence_ids"], list)
        assert isinstance(result["contradicting_evidence_ids"], list)

    def test_reason_coerced_to_string(self) -> None:
        """Non-string reason is coerced to string."""
        raw = json.dumps({
            "citation_correct": True,
            "sufficient": True,
            "contradicted": False,
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "reason": 42,
        })
        result = _parse_verifier_response(raw)
        assert isinstance(result["reason"], str)

    def test_non_dict_json_raises_schema_error(self) -> None:
        """JSON array at root → SchemaValidationError."""
        with pytest.raises(SchemaValidationError, match="must be a JSON object"):
            _parse_verifier_response(json.dumps([1, 2, 3]))


# ---------------------------------------------------------------------------
# TestClaimVerificationLogic
# ---------------------------------------------------------------------------

class TestClaimVerificationLogic:
    """Tests _verify_claim for all key verification scenarios."""

    def _normalized(self, version: str | None = "3.1") -> Any:
        return normalize_incident(
            "AUTH_401 after upgrading to SDK 3.1.",
            current_version=version,
            error_codes=["AUTH_401"],
        )

    def _applicable(self) -> list[RetrievalResult]:
        return [_make_retrieval_result("chunk-001")]

    def _run_verify_claim(
        self,
        *,
        claim: DiagnosisClaim | None = None,
        llm_response: str | None = None,
        llm_exception: Exception | None = None,
        applicable: list[RetrievalResult] | None = None,
    ) -> ClaimVerification:
        if claim is None:
            claim = _make_claim()
        if applicable is None:
            applicable = self._applicable()
        applicable_id_set = build_applicable_id_set(applicable)
        normalized = self._normalized()

        if llm_exception:
            llm = FakeLLMClient(exception=llm_exception)
        else:
            llm = FakeLLMClient(response=llm_response or _make_verifier_json())

        return _verify_claim(
            claim=claim,
            normalized=normalized,
            applicable_results=applicable,
            applicable_id_set=applicable_id_set,
            llm_client=llm,
        )

    def test_fully_supported_claim_verified(self) -> None:
        """Valid citation + supported + sufficient + no contradiction → VERIFIED."""
        cv = self._run_verify_claim(
            llm_response=_make_verifier_json(
                citation_correct=True, sufficient=True, contradicted=False
            )
        )
        assert cv.verdict == VerificationVerdict.VERIFIED
        assert cv.citation_validity == CitationValidity.VALID
        assert cv.citation_correct is True
        assert cv.sufficient is True
        assert cv.contradicted is False

    def test_invalid_citation_id_rejected_without_llm(self) -> None:
        """Hallucinated chunk ID → REJECTED immediately, no LLM call."""
        claim = _make_claim(evidence_ids=["HALLUCINATED-999"])
        llm = FakeLLMClient(response=_make_verifier_json())
        applicable = self._applicable()
        cv = _verify_claim(
            claim=claim,
            normalized=self._normalized(),
            applicable_results=applicable,
            applicable_id_set=build_applicable_id_set(applicable),
            llm_client=llm,
        )
        assert cv.verdict == VerificationVerdict.REJECTED
        assert cv.citation_validity == CitationValidity.INVALID
        assert cv.citation_correct is False
        assert llm.call_count == 0  # LLM not called

    def test_empty_citation_rejected_without_llm(self) -> None:
        """Claim with no citations → REJECTED, no LLM call."""
        claim = _make_claim(evidence_ids=[])
        llm = FakeLLMClient(response=_make_verifier_json())
        applicable = self._applicable()
        cv = _verify_claim(
            claim=claim,
            normalized=self._normalized(),
            applicable_results=applicable,
            applicable_id_set=build_applicable_id_set(applicable),
            llm_client=llm,
        )
        assert cv.verdict == VerificationVerdict.REJECTED
        assert cv.citation_validity == CitationValidity.EMPTY
        assert llm.call_count == 0

    def test_valid_citation_but_unsupported_rejected(self) -> None:
        """Valid citation but citation_correct=False → REJECTED."""
        cv = self._run_verify_claim(
            llm_response=_make_verifier_json(citation_correct=False, sufficient=False)
        )
        assert cv.verdict == VerificationVerdict.REJECTED
        assert cv.citation_validity == CitationValidity.VALID
        assert cv.citation_correct is False

    def test_insufficient_evidence_rejected(self) -> None:
        """Valid citation + supported but insufficient → REJECTED."""
        cv = self._run_verify_claim(
            llm_response=_make_verifier_json(citation_correct=True, sufficient=False)
        )
        assert cv.verdict == VerificationVerdict.REJECTED
        assert cv.sufficient is False

    def test_contradicted_claim_rejected(self) -> None:
        """Valid citation + supported + sufficient + contradicted → REJECTED."""
        cv = self._run_verify_claim(
            llm_response=_make_verifier_json(
                citation_correct=True, sufficient=True, contradicted=True,
                contradicting_ids=["chunk-001"],
            )
        )
        assert cv.verdict == VerificationVerdict.REJECTED
        assert cv.contradicted is True
        assert "chunk-001" in cv.contradicting_evidence_ids

    def test_llm_error_propagates(self) -> None:
        """LLMError from the verifier propagates as LLMError."""
        with pytest.raises(LLMError, match="timeout"):
            self._run_verify_claim(llm_exception=LLMError("timeout"))

    def test_malformed_verifier_response_raises_schema_error(self) -> None:
        """Malformed verifier response → SchemaValidationError."""
        with pytest.raises(SchemaValidationError):
            self._run_verify_claim(llm_response="NOT_VALID_JSON{{{")

    def test_multiple_supporting_ids_preserved(self) -> None:
        """Multiple supporting evidence IDs are all preserved in the result."""
        applicable = [
            _make_retrieval_result("chunk-001"),
            _make_retrieval_result("chunk-002"),
        ]
        claim = _make_claim(evidence_ids=["chunk-001", "chunk-002"])
        cv = _verify_claim(
            claim=claim,
            normalized=self._normalized(),
            applicable_results=applicable,
            applicable_id_set=build_applicable_id_set(applicable),
            llm_client=FakeLLMClient(response=_make_verifier_json(
                supporting_ids=["chunk-001", "chunk-002"]
            )),
        )
        assert cv.verdict == VerificationVerdict.VERIFIED
        assert "chunk-001" in cv.supporting_evidence_ids
        assert "chunk-002" in cv.supporting_evidence_ids

    def test_multiple_contradicting_ids_preserved(self) -> None:
        """Multiple contradicting evidence IDs are all preserved in the result."""
        applicable = [
            _make_retrieval_result("chunk-001"),
            _make_retrieval_result("chunk-002"),
        ]
        claim = _make_claim(evidence_ids=["chunk-001"])
        cv = _verify_claim(
            claim=claim,
            normalized=self._normalized(),
            applicable_results=applicable,
            applicable_id_set=build_applicable_id_set(applicable),
            llm_client=FakeLLMClient(response=_make_verifier_json(
                citation_correct=True, sufficient=True, contradicted=True,
                contradicting_ids=["chunk-001", "chunk-002"],
            )),
        )
        assert cv.contradicted is True
        assert "chunk-001" in cv.contradicting_evidence_ids
        assert "chunk-002" in cv.contradicting_evidence_ids


# ---------------------------------------------------------------------------
# TestVerifyDiagnosis
# ---------------------------------------------------------------------------

class TestVerifyDiagnosis:
    """Tests verify_diagnosis (the main M5 entry point) with multiple claims."""

    def _normalized(self) -> Any:
        return normalize_incident(
            "AUTH_401 after upgrading to SDK 3.1.",
            current_version="3.1",
            error_codes=["AUTH_401"],
        )

    def test_all_claims_verified(self) -> None:
        """Happy path: all claims pass → all_verified=True."""
        applicable = [_make_retrieval_result("chunk-001")]
        claims = [
            DiagnosisClaim(role=ClaimRole.ROOT_CAUSE, text="Root cause.", evidence_ids=["chunk-001"]),
            DiagnosisClaim(role=ClaimRole.FIX, text="Fix.", evidence_ids=["chunk-001"]),
        ]
        diagnosis = DiagnosisResult(claims=claims)
        # Each claim needs one LLM call
        llm = FakeLLMClient(responses=[_make_verifier_json(), _make_verifier_json()])

        result = verify_diagnosis(
            diagnosis=diagnosis,
            applicable_results=applicable,
            normalized=self._normalized(),
            llm_client=llm,
        )
        assert isinstance(result, DiagnosisVerification)
        assert result.all_verified is True
        assert len(result.claim_verifications) == 2

    def test_one_claim_fails_mixed_outcome(self) -> None:
        """One claim verified, one rejected → all_verified=False, mixed results."""
        applicable = [_make_retrieval_result("chunk-001")]
        claims = [
            DiagnosisClaim(role=ClaimRole.ROOT_CAUSE, text="Root cause.", evidence_ids=["chunk-001"]),
            DiagnosisClaim(role=ClaimRole.FIX, text="Fix.", evidence_ids=["chunk-001"]),
        ]
        diagnosis = DiagnosisResult(claims=claims)
        # First call: VERIFIED; second call: REJECTED (not sufficient)
        llm = FakeLLMClient(responses=[
            _make_verifier_json(citation_correct=True, sufficient=True),
            _make_verifier_json(citation_correct=True, sufficient=False),
        ])

        result = verify_diagnosis(
            diagnosis=diagnosis,
            applicable_results=applicable,
            normalized=self._normalized(),
            llm_client=llm,
        )
        assert result.all_verified is False
        assert len(result.verified_claim_ids) == 1
        assert len(result.rejected_claim_ids) == 1

    def test_claim_with_invalid_citation_rejected(self) -> None:
        """Claim citing nonexistent ID → REJECTED with no LLM call for that claim."""
        applicable = [_make_retrieval_result("chunk-001")]
        # root_cause cites invalid ID; explanation cites valid ID
        claims = [
            DiagnosisClaim(role=ClaimRole.ROOT_CAUSE, text="Root cause.", evidence_ids=["FAKE-999"]),
            DiagnosisClaim(role=ClaimRole.EXPLANATION, text="Explanation.", evidence_ids=["chunk-001"]),
        ]
        diagnosis = DiagnosisResult(claims=claims)
        llm = FakeLLMClient(responses=[_make_verifier_json()])  # only 1 LLM call needed

        result = verify_diagnosis(
            diagnosis=diagnosis,
            applicable_results=applicable,
            normalized=self._normalized(),
            llm_client=llm,
        )
        root_cv = next(
            cv for cv in result.claim_verifications
            if ClaimRole.ROOT_CAUSE.value in cv.claim_id or True
        )
        # Find by checking citation_validity
        invalid_cvs = [cv for cv in result.claim_verifications if cv.citation_validity == CitationValidity.INVALID]
        valid_cvs = [cv for cv in result.claim_verifications if cv.citation_validity == CitationValidity.VALID]
        assert len(invalid_cvs) == 1
        assert len(valid_cvs) == 1
        assert invalid_cvs[0].verdict == VerificationVerdict.REJECTED

    def test_only_applicable_evidence_sent_to_verifier(self) -> None:
        """
        verify_diagnosis receives only applicable_results.
        This test verifies structural enforcement: the function signature
        takes applicable_results, so NOT_APPLICABLE chunks are never passed.
        """
        applicable = [_make_retrieval_result("chunk-001", applies_to=">=3.0,<4.0")]
        # If a 2.x chunk were somehow passed, it would be in applicable_id_set
        # and citation would become VALID — but M4 prevents this.
        claims = [
            DiagnosisClaim(role=ClaimRole.ROOT_CAUSE, text="Root cause.", evidence_ids=["chunk-001"])
        ]
        diagnosis = DiagnosisResult(claims=claims)
        llm = FakeLLMClient(response=_make_verifier_json())

        result = verify_diagnosis(
            diagnosis=diagnosis,
            applicable_results=applicable,
            normalized=self._normalized(),
            llm_client=llm,
        )
        assert result.claim_verifications[0].citation_validity == CitationValidity.VALID

    def test_verified_claim_ids_correct(self) -> None:
        """verified_claim_ids returns only VERIFIED claim IDs."""
        applicable = [_make_retrieval_result("chunk-001")]
        claim = DiagnosisClaim(role=ClaimRole.ROOT_CAUSE, text="Root.", evidence_ids=["chunk-001"])
        diagnosis = DiagnosisResult(claims=[claim])
        llm = FakeLLMClient(response=_make_verifier_json())

        result = verify_diagnosis(
            diagnosis=diagnosis,
            applicable_results=applicable,
            normalized=self._normalized(),
            llm_client=llm,
        )
        assert claim.claim_id in result.verified_claim_ids

    def test_rejected_claim_ids_correct(self) -> None:
        """rejected_claim_ids returns only REJECTED claim IDs."""
        applicable = [_make_retrieval_result("chunk-001")]
        claim = DiagnosisClaim(role=ClaimRole.ROOT_CAUSE, text="Root.", evidence_ids=["FAKE-ID"])
        diagnosis = DiagnosisResult(claims=[claim])
        llm = FakeLLMClient(response=_make_verifier_json())

        result = verify_diagnosis(
            diagnosis=diagnosis,
            applicable_results=applicable,
            normalized=self._normalized(),
            llm_client=llm,
        )
        assert claim.claim_id in result.rejected_claim_ids

    def test_empty_applicable_results_all_rejected(self) -> None:
        """If applicable_results is empty, all citations are invalid → all REJECTED."""
        claims = [
            DiagnosisClaim(role=ClaimRole.ROOT_CAUSE, text="Root.", evidence_ids=["chunk-001"])
        ]
        diagnosis = DiagnosisResult(claims=claims)
        llm = FakeLLMClient(response=_make_verifier_json())

        result = verify_diagnosis(
            diagnosis=diagnosis,
            applicable_results=[],  # nothing applicable
            normalized=self._normalized(),
            llm_client=llm,
        )
        assert result.all_verified is False
        assert all(cv.verdict == VerificationVerdict.REJECTED for cv in result.claim_verifications)

    def test_reason_field_propagated(self) -> None:
        """The LLM reason string is preserved in ClaimVerification.reason."""
        applicable = [_make_retrieval_result("chunk-001")]
        claim = DiagnosisClaim(role=ClaimRole.ROOT_CAUSE, text="Root.", evidence_ids=["chunk-001"])
        diagnosis = DiagnosisResult(claims=[claim])
        llm = FakeLLMClient(response=_make_verifier_json(reason="Explicit reason text."))

        result = verify_diagnosis(
            diagnosis=diagnosis,
            applicable_results=applicable,
            normalized=self._normalized(),
            llm_client=llm,
        )
        assert "Explicit reason text." in result.claim_verifications[0].reason


# ---------------------------------------------------------------------------
# TestWrongVersionExclusion
# ---------------------------------------------------------------------------

class TestWrongVersionExclusion:
    """
    Architectural property tests proving M4's applicability boundary protects M5.

    The key property:
        A 2.x chunk excluded by M4 must NOT be a valid citation in M5.
        M5 does not need version logic — M4 already filtered the bundle.
    """

    def test_2x_chunk_not_in_applicable_set_for_3x_incident(self) -> None:
        """
        M4 applicable bundle for a 3.x incident contains only 3.x chunks.
        Simulated: we pass only the 3.x chunk as applicable_results.
        The 2.x chunk is absent → citing it is citation_invalid.
        """
        # M4 produced: only AUTH-002-C01 (3.x) passed applicability
        applicable = [_make_retrieval_result("AUTH-002-C01", applies_to=">=3.0,<4.0")]
        applicable_set = build_applicable_id_set(applicable)

        # Diagnosis claims: root_cause cites the 2.x chunk (AUTH-001-C01) incorrectly
        claim_citing_2x = _make_claim(
            text="SDK 3.x requires new auth headers.",
            evidence_ids=["AUTH-001-C01"],  # 2.x chunk, excluded by M4
        )
        validity, invalid = check_citation_validity(claim_citing_2x, applicable_set)
        assert validity == CitationValidity.INVALID
        assert "AUTH-001-C01" in invalid

    def test_wrong_version_chunk_rejected_in_full_verify(self) -> None:
        """
        Full _verify_claim call where the cited ID is a 2.x chunk excluded by M4.
        Result: REJECTED with no LLM call (citation short-circuit).
        """
        applicable = [_make_retrieval_result("AUTH-002-C01", applies_to=">=3.0,<4.0")]
        claim = _make_claim(evidence_ids=["AUTH-001-C01"])
        normalized = normalize_incident(
            "AUTH_401 after upgrading to SDK 3.1.",
            current_version="3.1",
            error_codes=["AUTH_401"],
        )
        llm = FakeLLMClient(response=_make_verifier_json())

        cv = _verify_claim(
            claim=claim,
            normalized=normalized,
            applicable_results=applicable,
            applicable_id_set=build_applicable_id_set(applicable),
            llm_client=llm,
        )
        assert cv.verdict == VerificationVerdict.REJECTED
        assert cv.citation_validity == CitationValidity.INVALID
        assert llm.call_count == 0  # No LLM call — deterministic rejection

    def test_m4_boundary_enforced_by_absence_not_m5_version_logic(self) -> None:
        """
        M5 does NOT implement version logic.
        The protection comes purely from absence of the 2.x chunk in applicable_results.
        If the same 2.x chunk ID were (wrongly) included in applicable_results,
        M5 would consider the citation valid.
        This test proves M4's boundary is the correct enforcement point.
        """
        # WRONG scenario (simulates M4 failure): 2.x chunk accidentally in applicable
        accidental_applicable = [_make_retrieval_result("AUTH-001-C01", applies_to=">=2.0,<3.0")]
        claim = _make_claim(evidence_ids=["AUTH-001-C01"])
        applicable_set = build_applicable_id_set(accidental_applicable)

        # M5 would consider this VALID (it doesn't know about version applicability)
        validity, invalid = check_citation_validity(claim, applicable_set)
        assert validity == CitationValidity.VALID  # M5 trusts the bundle it receives

        # This confirms: the protection is M4's responsibility, not M5's.
        # M5 is correct to trust the applicable_results it is given.

    def test_integration_wrong_version_cite_in_diagnosis(self) -> None:
        """
        End-to-end test: diagnosis cites a 2.x doc for a 3.x incident.
        M4 applicable bundle excludes the 2.x chunk.
        M5 must reject the claim that cites it.
        """
        # M4 produced applicable_results containing only the 3.x chunk
        applicable_3x = [_make_retrieval_result("AUTH-002-C01", applies_to=">=3.0,<4.0")]

        # Diagnosis generator (incorrectly) cited a 2.x chunk
        claims = [
            DiagnosisClaim(
                role=ClaimRole.ROOT_CAUSE,
                text="SDK 3.x auth requires Bearer token.",
                evidence_ids=["AUTH-001-C01"],  # 2.x chunk, not in applicable
            )
        ]
        diagnosis = DiagnosisResult(claims=claims)
        normalized = normalize_incident(
            "AUTH_401 on SDK 3.1 after upgrade from 2.8.",
            current_version="3.1",
            previous_version="2.8",
            error_codes=["AUTH_401"],
        )
        llm = FakeLLMClient(response=_make_verifier_json())

        result = verify_diagnosis(
            diagnosis=diagnosis,
            applicable_results=applicable_3x,
            normalized=normalized,
            llm_client=llm,
        )
        assert result.all_verified is False
        root_cv = result.claim_verifications[0]
        assert root_cv.verdict == VerificationVerdict.REJECTED
        assert root_cv.citation_validity == CitationValidity.INVALID


# ---------------------------------------------------------------------------
# TestVerifiedPipeline
# ---------------------------------------------------------------------------

class TestVerifiedPipeline:
    """
    Tests run_verified_diagnosis — the M4+M5 combined pipeline.

    Uses mock retriever and FakeLLMClient (no real corpus or Gemini).
    """

    def _make_retriever(self, results: list[RetrievalResult] | None = None) -> MagicMock:
        m = MagicMock()
        m.retrieve_as_contracts.return_value = results or []
        return m

    def _valid_diagnosis_json(self, evidence_id: str = "chunk-001") -> str:
        return json.dumps({"claims": [
            {"role": "root_cause", "text": "Root cause.", "evidence_ids": [evidence_id]}
        ]})

    def test_successful_pipeline_produces_verification(self) -> None:
        """Happy path: M4 diagnosis + M5 verification → verification attached."""
        applicable = [_make_retrieval_result("chunk-001", applies_to=">=3.0,<4.0")]
        # LLM is called twice: once for diagnosis (M4), once for verification (M5)
        llm = FakeLLMClient(responses=[
            self._valid_diagnosis_json("chunk-001"),
            _make_verifier_json(),
        ])
        result = run_verified_diagnosis(
            description="AUTH_401 after upgrading to SDK 3.1.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=self._make_retriever(applicable),
            llm_client=llm,
        )
        assert result.diagnosis is not None
        assert result.verification is not None
        assert result.verification_error is None

    def test_needs_info_skips_verification(self) -> None:
        """NEEDS_INFO from M4 → M5 verification skipped."""
        llm = FakeLLMClient(response=self._valid_diagnosis_json())
        result = run_verified_diagnosis(
            description="AUTH_401 on my API.",  # no version
            error_codes=["AUTH_401"],
            retriever=self._make_retriever([
                _make_retrieval_result("c2", applies_to=">=2.0,<3.0"),
                _make_retrieval_result("c3", applies_to=">=3.0,<4.0"),
            ]),
            llm_client=llm,
        )
        assert result.outcome == SystemOutcome.NEEDS_INFO
        assert result.verification is None

    def test_degraded_m4_skips_verification(self) -> None:
        """Malformed M4 diagnosis → DEGRADED, M5 skipped."""
        llm = FakeLLMClient(response="INVALID_JSON")
        result = run_verified_diagnosis(
            description="AUTH_401 after upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=self._make_retriever([_make_retrieval_result("chunk-001")]),
            llm_client=llm,
        )
        assert result.outcome == SystemOutcome.DEGRADED
        assert result.verification is None

    def test_separate_verifier_llm_client_used(self) -> None:
        """When verifier_llm_client is passed separately, only it handles M5."""
        applicable = [_make_retrieval_result("chunk-001", applies_to=">=3.0,<4.0")]
        # diagnosis_llm only handles M4 (1 call)
        diagnosis_llm = FakeLLMClient(response=self._valid_diagnosis_json("chunk-001"))
        # verifier_llm only handles M5 (1 call)
        verifier_llm = FakeLLMClient(response=_make_verifier_json())

        result = run_verified_diagnosis(
            description="AUTH_401 after upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=self._make_retriever(applicable),
            llm_client=diagnosis_llm,
            verifier_llm_client=verifier_llm,
        )
        assert diagnosis_llm.call_count == 1  # M4 only
        assert verifier_llm.call_count == 1   # M5 only
        assert result.verification is not None

    def test_verification_error_captured_gracefully(self) -> None:
        """M5 LLMError is captured in verification_error (not raised)."""
        applicable = [_make_retrieval_result("chunk-001", applies_to=">=3.0,<4.0")]
        diagnosis_llm = FakeLLMClient(response=self._valid_diagnosis_json("chunk-001"))
        verifier_llm = FakeLLMClient(exception=LLMError("verifier timeout"))

        result = run_verified_diagnosis(
            description="AUTH_401 after upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=self._make_retriever(applicable),
            llm_client=diagnosis_llm,
            verifier_llm_client=verifier_llm,
        )
        assert result.verification is None
        assert result.verification_error is not None
        assert "timeout" in result.verification_error.lower()

    def test_baseline_fields_accessible_via_result(self) -> None:
        """All M4 baseline fields are accessible via VerifiedPipelineResult."""
        applicable = [_make_retrieval_result("chunk-001", applies_to=">=3.0,<4.0")]
        llm = FakeLLMClient(responses=[
            self._valid_diagnosis_json("chunk-001"),
            _make_verifier_json(),
        ])
        result = run_verified_diagnosis(
            description="AUTH_401 after upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=self._make_retriever(applicable),
            llm_client=llm,
        )
        assert result.normalized is not None
        assert result.applicable_results is not None
        assert result.applicability_decisions is not None
        assert result.diagnosis is not None

    def test_verified_pipeline_all_claims_verified(self) -> None:
        """End-to-end: valid diagnosis + valid verification → all_verified=True."""
        applicable = [_make_retrieval_result("chunk-001", applies_to=">=3.0,<4.0")]
        llm = FakeLLMClient(responses=[
            self._valid_diagnosis_json("chunk-001"),
            _make_verifier_json(citation_correct=True, sufficient=True, contradicted=False),
        ])
        result = run_verified_diagnosis(
            description="AUTH_401 after upgrade to 3.1.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=self._make_retriever(applicable),
            llm_client=llm,
        )
        assert result.verification is not None
        assert result.verification.all_verified is True

    def test_verified_pipeline_invalid_citation_claim_rejected(self) -> None:
        """M4 generates a claim with a hallucinated chunk ID → M5 rejects it."""
        applicable = [_make_retrieval_result("chunk-real", applies_to=">=3.0,<4.0")]
        # M4 diagnosis cites a chunk not in applicable
        diagnosis_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "Root.", "evidence_ids": ["FAKE-999"]}
        ]})
        llm = FakeLLMClient(responses=[
            diagnosis_json,           # M4 (diagnosis)
            _make_verifier_json(),    # M5 won't be called (short-circuit), but safe
        ])
        result = run_verified_diagnosis(
            description="AUTH_401 after upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=self._make_retriever(applicable),
            llm_client=llm,
        )
        assert result.verification is not None
        assert result.verification.all_verified is False
        assert result.verification.claim_verifications[0].citation_validity == CitationValidity.INVALID


# ---------------------------------------------------------------------------
# TestDiagnosisVerificationModel
# ---------------------------------------------------------------------------

class TestDiagnosisVerificationModel:
    """Tests DiagnosisVerification model properties."""

    def _make_cv(
        self,
        verdict: VerificationVerdict,
        claim_id: str = "claim-001",
    ) -> ClaimVerification:
        return ClaimVerification(
            claim_id=claim_id,
            verdict=verdict,
            citation_validity=CitationValidity.VALID,
            citation_correct=(verdict == VerificationVerdict.VERIFIED),
            sufficient=(verdict == VerificationVerdict.VERIFIED),
            contradicted=False,
            reason="test",
        )

    def test_all_verified_true_when_all_pass(self) -> None:
        dv = DiagnosisVerification(claim_verifications=[
            self._make_cv(VerificationVerdict.VERIFIED, "c1"),
            self._make_cv(VerificationVerdict.VERIFIED, "c2"),
        ])
        assert dv.all_verified is True

    def test_all_verified_false_when_any_fail(self) -> None:
        dv = DiagnosisVerification(claim_verifications=[
            self._make_cv(VerificationVerdict.VERIFIED, "c1"),
            self._make_cv(VerificationVerdict.REJECTED, "c2"),
        ])
        assert dv.all_verified is False

    def test_verified_claim_ids(self) -> None:
        dv = DiagnosisVerification(claim_verifications=[
            self._make_cv(VerificationVerdict.VERIFIED, "c1"),
            self._make_cv(VerificationVerdict.REJECTED, "c2"),
        ])
        assert dv.verified_claim_ids == ["c1"]
        assert dv.rejected_claim_ids == ["c2"]

    def test_empty_verifications(self) -> None:
        dv = DiagnosisVerification(claim_verifications=[])
        assert dv.all_verified is True  # vacuously true
        assert dv.verified_claim_ids == []
        assert dv.rejected_claim_ids == []
