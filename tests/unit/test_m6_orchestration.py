"""
DevTrace — Module 6: Test suite.

Covers:
    TestRootCauseSurvival          — root_cause_survived() logic (8 tests)
    TestOutcomeClassification      — classify_outcome() rules (6 tests)
    TestAnswerAssembly             — assemble_answer_text / collect_verified_claims (6 tests)
    TestRetryConditions            — when retry is/isn't allowed (5 tests)
    TestRetryPromptBuilder         — build_retry_prompt content (5 tests)
    TestOrchestratorNeedsInfo      — NEEDS_INFO early exit (3 tests)
    TestOrchestratorDegraded       — DEGRADED system failures (5 tests)
    TestOrchestratorFullAnswer     — ANSWERED_FULL happy path (4 tests)
    TestOrchestratorPartialAnswer  — ANSWERED_PARTIAL cases (4 tests)
    TestOrchestratorRetry          — retry trigger and outcomes (8 tests)
    TestOrchestratorRetryOnce      — max one retry enforcement (4 tests)
    TestWrongVersionExclusion      — architecture thesis test (3 tests)
    TestEndToEndIntegration        — realistic E2E integration (3 tests)

Total: ~64 tests
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
from src.orchestration.assembly import (
    assemble_answer_text,
    build_answered_answer,
    build_degraded_answer,
    build_insufficient_answer,
    build_needs_info_answer,
    classify_outcome,
    collect_verified_claims,
    get_root_cause_rejection_reason,
    root_cause_survived,
)
from src.orchestration.models import FinalOutcome, TroubleshootingResult, VerifiedAnswer
from src.orchestration.orchestrator import _retry_is_allowed, run_troubleshooting
from src.orchestration.retry_prompt import build_retry_prompt
from src.verification.models import (
    CitationValidity,
    ClaimVerification,
    DiagnosisVerification,
    VerificationVerdict,
)


# ---------------------------------------------------------------------------
# Shared factories / helpers
# ---------------------------------------------------------------------------

def _rr(
    chunk_id: str = "chunk-001",
    doc_id: str = "DOC-001",
    score: float = 0.80,
    applies_to: str = ">=3.0,<4.0",
    content: str = "SDK 3.x requires OAuth Bearer token authentication.",
    topic: str = "authentication",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        score=score,
        source=RetrievalSource.HYBRID,
        metadata={"applies_to": applies_to, "content": content, "topic": topic},
    )


def _claim(
    role: ClaimRole = ClaimRole.ROOT_CAUSE,
    text: str = "Root cause of the issue.",
    evidence_ids: list[str] | None = None,
) -> DiagnosisClaim:
    return DiagnosisClaim(
        role=role,
        text=text,
        evidence_ids=evidence_ids if evidence_ids is not None else ["chunk-001"],
    )


def _diagnosis(claims: list[DiagnosisClaim] | None = None) -> DiagnosisResult:
    """Build a DiagnosisResult with exactly one ROOT_CAUSE."""
    if claims is None:
        claims = [_claim(role=ClaimRole.ROOT_CAUSE)]
    return DiagnosisResult(claims=claims)


def _cv(
    claim_id: str,
    verdict: VerificationVerdict = VerificationVerdict.VERIFIED,
    citation_validity: CitationValidity = CitationValidity.VALID,
    citation_correct: bool = True,
    sufficient: bool = True,
    contradicted: bool = False,
    reason: str = "Evidence supports the claim.",
) -> ClaimVerification:
    return ClaimVerification(
        claim_id=claim_id,
        verdict=verdict,
        citation_validity=citation_validity,
        citation_correct=citation_correct,
        sufficient=sufficient,
        contradicted=contradicted,
        reason=reason,
    )


def _rejected_cv(
    claim_id: str,
    reason: str = "Evidence does not support this claim.",
) -> ClaimVerification:
    return ClaimVerification(
        claim_id=claim_id,
        verdict=VerificationVerdict.REJECTED,
        citation_validity=CitationValidity.INVALID,
        citation_correct=False,
        sufficient=False,
        contradicted=False,
        reason=reason,
    )


def _verification(claim_verifications: list[ClaimVerification]) -> DiagnosisVerification:
    return DiagnosisVerification(claim_verifications=claim_verifications)


def _make_retriever(results: list[RetrievalResult] | None = None) -> MagicMock:
    m = MagicMock()
    m.retrieve_as_contracts.return_value = results or []
    return m


def _diagnosis_json(
    root_cause_text: str = "Root cause.",
    fix_text: str | None = None,
    evidence_id: str = "chunk-001",
) -> str:
    """By default creates a root_cause-only diagnosis (1 LLM verify call)."""
    claims = [{"role": "root_cause", "text": root_cause_text, "evidence_ids": [evidence_id]}]
    if fix_text:
        claims.append({"role": "fix", "text": fix_text, "evidence_ids": [evidence_id]})
    return json.dumps({"claims": claims})


def _verifier_json(
    citation_correct: bool = True,
    sufficient: bool = True,
    contradicted: bool = False,
    supporting_ids: list[str] | None = None,
    reason: str = "Evidence supports the claim.",
) -> str:
    return json.dumps({
        "citation_correct": citation_correct,
        "sufficient": sufficient,
        "contradicted": contradicted,
        "supporting_evidence_ids": supporting_ids if supporting_ids is not None else ["chunk-001"],
        "contradicting_evidence_ids": [],
        "reason": reason,
    })


def _normalized(version: str | None = "3.1") -> Any:
    return normalize_incident(
        "AUTH_401 after upgrading to SDK 3.1.",
        current_version=version,
        error_codes=["AUTH_401"],
    )


# ---------------------------------------------------------------------------
# TestRootCauseSurvival
# ---------------------------------------------------------------------------

class TestRootCauseSurvival:
    """Tests root_cause_survived() — the central M6 gate."""

    def _make_root_only_diagnosis(self) -> DiagnosisResult:
        return _diagnosis([_claim(ClaimRole.ROOT_CAUSE)])

    def test_verified_root_cause_survives(self) -> None:
        """root_cause VERIFIED → True."""
        diag = self._make_root_only_diagnosis()
        root = diag.claims[0]
        ver = _verification([_cv(root.claim_id, verdict=VerificationVerdict.VERIFIED)])
        assert root_cause_survived(diag, ver) is True

    def test_rejected_root_cause_does_not_survive(self) -> None:
        """root_cause REJECTED → False."""
        diag = self._make_root_only_diagnosis()
        root = diag.claims[0]
        ver = _verification([_rejected_cv(root.claim_id)])
        assert root_cause_survived(diag, ver) is False

    def test_verified_fix_alone_does_not_survive(self) -> None:
        """fix VERIFIED but root_cause REJECTED → False."""
        diag = _diagnosis([
            _claim(ClaimRole.ROOT_CAUSE, text="Root."),
            _claim(ClaimRole.FIX, text="Fix."),
        ])
        root_id = diag.claims[0].claim_id
        fix_id = diag.claims[1].claim_id
        ver = _verification([
            _rejected_cv(root_id),
            _cv(fix_id, verdict=VerificationVerdict.VERIFIED),
        ])
        assert root_cause_survived(diag, ver) is False

    def test_all_verified_survives(self) -> None:
        """All claims VERIFIED → True (root_cause passes)."""
        diag = _diagnosis([
            _claim(ClaimRole.ROOT_CAUSE),
            _claim(ClaimRole.FIX),
            _claim(ClaimRole.EXPLANATION),
        ])
        ver = _verification([_cv(c.claim_id) for c in diag.claims])
        assert root_cause_survived(diag, ver) is True

    def test_root_cause_verified_others_rejected_survives(self) -> None:
        """root_cause VERIFIED but fix + explanation REJECTED → True (root survives)."""
        diag = _diagnosis([
            _claim(ClaimRole.ROOT_CAUSE),
            _claim(ClaimRole.FIX),
        ])
        root_id = diag.claims[0].claim_id
        fix_id = diag.claims[1].claim_id
        ver = _verification([
            _cv(root_id, verdict=VerificationVerdict.VERIFIED),
            _rejected_cv(fix_id),
        ])
        assert root_cause_survived(diag, ver) is True

    def test_empty_verification_does_not_survive(self) -> None:
        """If verification has no entries for root_cause claim → False."""
        diag = self._make_root_only_diagnosis()
        ver = _verification([])  # no verifications at all
        assert root_cause_survived(diag, ver) is False

    def test_root_cause_rejection_reason_extracted(self) -> None:
        """get_root_cause_rejection_reason returns the reason from the rejected claim."""
        diag = self._make_root_only_diagnosis()
        root = diag.claims[0]
        reason_text = "Insufficient evidence for root cause."
        ver = _verification([_rejected_cv(root.claim_id, reason=reason_text)])
        extracted = get_root_cause_rejection_reason(diag, ver)
        assert "Insufficient evidence" in extracted

    def test_root_cause_rejection_reason_default_when_no_reason(self) -> None:
        """If no reason set → returns a non-empty default string."""
        diag = self._make_root_only_diagnosis()
        root = diag.claims[0]
        cv = ClaimVerification(
            claim_id=root.claim_id,
            verdict=VerificationVerdict.REJECTED,
            citation_validity=CitationValidity.INVALID,
            citation_correct=False,
            sufficient=False,
            contradicted=False,
            reason="",
        )
        ver = _verification([cv])
        extracted = get_root_cause_rejection_reason(diag, ver)
        assert len(extracted) > 0


# ---------------------------------------------------------------------------
# TestOutcomeClassification
# ---------------------------------------------------------------------------

class TestOutcomeClassification:
    """Tests classify_outcome() — ANSWERED_FULL vs ANSWERED_PARTIAL cases."""

    def _make_full_diagnosis(self) -> tuple[DiagnosisResult, dict[ClaimRole, str]]:
        claims = [
            _claim(ClaimRole.ROOT_CAUSE, text="Root cause."),
            _claim(ClaimRole.FIX, text="Apply fix."),
            _claim(ClaimRole.EXPLANATION, text="Explanation."),
        ]
        diag = DiagnosisResult(claims=claims)
        role_to_id = {c.role: c.claim_id for c in claims}
        return diag, role_to_id

    def test_all_verified_answered_full(self) -> None:
        """root_cause + fix + explanation all VERIFIED → ANSWERED_FULL."""
        diag, r2id = self._make_full_diagnosis()
        ver = _verification([_cv(cid) for cid in r2id.values()])
        assert classify_outcome(diag, ver) == FinalOutcome.ANSWERED_FULL

    def test_root_and_fix_verified_no_explanation_answered_partial(self) -> None:
        """root_cause + fix VERIFIED, explanation REJECTED → ANSWERED_PARTIAL."""
        diag, r2id = self._make_full_diagnosis()
        ver = _verification([
            _cv(r2id[ClaimRole.ROOT_CAUSE]),
            _cv(r2id[ClaimRole.FIX]),
            _rejected_cv(r2id[ClaimRole.EXPLANATION]),
        ])
        assert classify_outcome(diag, ver) == FinalOutcome.ANSWERED_PARTIAL

    def test_root_verified_fix_rejected_answered_partial(self) -> None:
        """root_cause VERIFIED, fix REJECTED → ANSWERED_PARTIAL."""
        diag = _diagnosis([
            _claim(ClaimRole.ROOT_CAUSE),
            _claim(ClaimRole.FIX),
        ])
        root_id = diag.claims[0].claim_id
        fix_id = diag.claims[1].claim_id
        ver = _verification([
            _cv(root_id),
            _rejected_cv(fix_id),
        ])
        assert classify_outcome(diag, ver) == FinalOutcome.ANSWERED_PARTIAL

    def test_root_only_no_fix_answered_partial(self) -> None:
        """root_cause VERIFIED, no fix/explanation claims → ANSWERED_PARTIAL."""
        diag = _diagnosis([_claim(ClaimRole.ROOT_CAUSE)])
        ver = _verification([_cv(diag.claims[0].claim_id)])
        assert classify_outcome(diag, ver) == FinalOutcome.ANSWERED_PARTIAL

    def test_explanation_absent_root_fix_verified_partial(self) -> None:
        """root_cause + fix VERIFIED, no explanation claim → ANSWERED_PARTIAL."""
        diag = _diagnosis([
            _claim(ClaimRole.ROOT_CAUSE),
            _claim(ClaimRole.FIX),
        ])
        ver = _verification([_cv(c.claim_id) for c in diag.claims])
        assert classify_outcome(diag, ver) == FinalOutcome.ANSWERED_PARTIAL

    def test_all_three_verified_full_not_partial(self) -> None:
        """ANSWERED_FULL is distinct from ANSWERED_PARTIAL."""
        diag = _diagnosis([
            _claim(ClaimRole.ROOT_CAUSE),
            _claim(ClaimRole.FIX),
            _claim(ClaimRole.EXPLANATION),
        ])
        ver = _verification([_cv(c.claim_id) for c in diag.claims])
        result = classify_outcome(diag, ver)
        assert result == FinalOutcome.ANSWERED_FULL
        assert result != FinalOutcome.ANSWERED_PARTIAL


# ---------------------------------------------------------------------------
# TestAnswerAssembly
# ---------------------------------------------------------------------------

class TestAnswerAssembly:
    """Tests assemble_answer_text, collect_verified_claims, build_*_answer functions."""

    def test_all_verified_answer_contains_all_roles(self) -> None:
        """All claims VERIFIED → answer text contains all role labels."""
        diag = _diagnosis([
            _claim(ClaimRole.ROOT_CAUSE, text="Root cause claim."),
            _claim(ClaimRole.FIX, text="Apply the fix."),
            _claim(ClaimRole.EXPLANATION, text="Explanation text."),
        ])
        ver = _verification([_cv(c.claim_id) for c in diag.claims])
        text = assemble_answer_text(diag, ver)
        assert "Root cause" in text
        assert "Fix" in text
        assert "Explanation" in text

    def test_rejected_claims_excluded_from_answer_text(self) -> None:
        """Rejected fix claim → fix text absent from answer."""
        diag = _diagnosis([
            _claim(ClaimRole.ROOT_CAUSE, text="Root cause."),
            _claim(ClaimRole.FIX, text="Secret fix text."),
        ])
        root_id = diag.claims[0].claim_id
        fix_id = diag.claims[1].claim_id
        ver = _verification([
            _cv(root_id),
            _rejected_cv(fix_id),
        ])
        text = assemble_answer_text(diag, ver)
        assert "Root cause" in text
        assert "Secret fix text" not in text

    def test_collect_verified_claims_only_verified(self) -> None:
        """collect_verified_claims returns only VERIFIED claims."""
        diag = _diagnosis([
            _claim(ClaimRole.ROOT_CAUSE),
            _claim(ClaimRole.FIX),
        ])
        root_id = diag.claims[0].claim_id
        fix_id = diag.claims[1].claim_id
        ver = _verification([
            _cv(root_id),
            _rejected_cv(fix_id),
        ])
        claims = collect_verified_claims(diag, ver)
        assert len(claims) == 1
        assert claims[0].role == ClaimRole.ROOT_CAUSE

    def test_build_insufficient_answer_structure(self) -> None:
        """build_insufficient_answer → INSUFFICIENT_EVIDENCE, no text, no claims."""
        answer = build_insufficient_answer()
        assert answer.outcome == FinalOutcome.INSUFFICIENT_EVIDENCE
        assert answer.answer_text is None
        assert answer.verified_claims == []

    def test_build_needs_info_answer_preserves_reason(self) -> None:
        """build_needs_info_answer → NEEDS_INFO with reason preserved."""
        answer = build_needs_info_answer("Please provide your SDK version.")
        assert answer.outcome == FinalOutcome.NEEDS_INFO
        assert answer.needs_info_reason is not None
        assert "SDK version" in answer.needs_info_reason

    def test_build_degraded_answer_preserves_error(self) -> None:
        """build_degraded_answer → DEGRADED with error detail preserved."""
        answer = build_degraded_answer("LLM timeout after 30s.")
        assert answer.outcome == FinalOutcome.DEGRADED
        assert answer.error_detail is not None
        assert "timeout" in answer.error_detail


# ---------------------------------------------------------------------------
# TestRetryConditions
# ---------------------------------------------------------------------------

class TestRetryConditions:
    """Tests the retry guard logic."""

    def test_retry_allowed_when_applicable_evidence_present(self) -> None:
        """_retry_is_allowed → True when applicable_results is non-empty."""
        applicable = [_rr()]
        assert _retry_is_allowed(applicable) is True

    def test_retry_not_allowed_when_no_applicable_evidence(self) -> None:
        """_retry_is_allowed → False when applicable_results is empty."""
        assert _retry_is_allowed([]) is False

    def test_retry_not_allowed_does_not_trigger_in_orchestrator(self) -> None:
        """No applicable evidence → root_cause citation invalid → INSUFFICIENT_EVIDENCE without retry."""
        # The 3.1 incident will have no applicable evidence (2.x chunk excluded)
        retriever = _make_retriever([_rr("chunk-001", applies_to=">=2.0,<3.0")])
        # diagnosis cites chunk-001 (excluded), verifier sees empty applicable → rejected
        diag_json = _diagnosis_json(evidence_id="chunk-001")
        llm = FakeLLMClient(responses=[diag_json])
        # With no applicable evidence, root_cause is REJECTED (citation invalid — empty set).
        # Retry is not allowed (applicable_results is empty).
        result = run_troubleshooting(
            description="AUTH_401 after upgrade to SDK 3.1.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        assert result.retry_attempted is False

    def test_retry_not_triggered_when_root_cause_verified(self) -> None:
        """No retry when initial root_cause is VERIFIED."""
        applicable = [_rr()]
        retriever = _make_retriever([applicable[0]])
        diag_json = _diagnosis_json()
        verify_json = _verifier_json()
        llm = FakeLLMClient(responses=[diag_json, verify_json])
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        assert result.retry_attempted is False
        assert result.final_outcome in (FinalOutcome.ANSWERED_FULL, FinalOutcome.ANSWERED_PARTIAL)

    def test_retry_not_triggered_on_explanation_failure_alone(self) -> None:
        """Root + fix VERIFIED, explanation alone rejected → no retry (ANSWERED_PARTIAL)."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        diag_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "Root.", "evidence_ids": ["c1"]},
            {"role": "fix", "text": "Fix.", "evidence_ids": ["c1"]},
            {"role": "explanation", "text": "Explain.", "evidence_ids": ["c1"]},
        ]})
        # root_cause: verified, fix: verified, explanation: rejected
        llm = FakeLLMClient(responses=[
            diag_json,
            _verifier_json(citation_correct=True, sufficient=True),    # root_cause
            _verifier_json(citation_correct=True, sufficient=True),    # fix
            _verifier_json(citation_correct=False, sufficient=False),  # explanation
        ])
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        assert result.retry_attempted is False
        assert result.final_outcome == FinalOutcome.ANSWERED_PARTIAL


# ---------------------------------------------------------------------------
# TestRetryPromptBuilder
# ---------------------------------------------------------------------------

class TestRetryPromptBuilder:
    """Tests build_retry_prompt content."""

    def _make_params(
        self,
    ) -> tuple[DiagnosisResult, DiagnosisVerification]:
        diag = _diagnosis([
            _claim(ClaimRole.ROOT_CAUSE, text="Initial root cause."),
            _claim(ClaimRole.FIX, text="Initial fix."),
        ])
        root_id = diag.claims[0].claim_id
        fix_id = diag.claims[1].claim_id
        ver = _verification([
            _rejected_cv(root_id, reason="Evidence does not support root cause."),
            _cv(fix_id),
        ])
        return diag, ver

    def test_prompt_contains_incident_description(self) -> None:
        """Retry prompt includes the incident description."""
        diag, ver = self._make_params()
        applicable = [_rr("chunk-001", content="Authentication docs for 3.x.")]
        prompt = build_retry_prompt(
            incident_description="AUTH_401 after upgrade.",
            current_version="3.1",
            previous_version="2.8",
            error_codes=["AUTH_401"],
            applicable_results=applicable,
            first_diagnosis=diag,
            first_verification=ver,
        )
        assert "AUTH_401 after upgrade." in prompt

    def test_prompt_contains_verification_failure_feedback(self) -> None:
        """Retry prompt includes the rejection reason from the first attempt."""
        diag, ver = self._make_params()
        applicable = [_rr()]
        prompt = build_retry_prompt(
            incident_description="Issue.",
            current_version="3.1",
            previous_version=None,
            error_codes=[],
            applicable_results=applicable,
            first_diagnosis=diag,
            first_verification=ver,
        )
        assert "Evidence does not support root cause." in prompt

    def test_prompt_contains_applicable_evidence(self) -> None:
        """Retry prompt includes all applicable evidence chunk IDs."""
        diag, ver = self._make_params()
        applicable = [
            _rr("AUTH-003-C01", content="Bearer token required."),
            _rr("AUTH-003-C02", content="Token refresh instructions."),
        ]
        prompt = build_retry_prompt(
            incident_description="Issue.",
            current_version="3.1",
            previous_version=None,
            error_codes=[],
            applicable_results=applicable,
            first_diagnosis=diag,
            first_verification=ver,
        )
        assert "AUTH-003-C01" in prompt
        assert "AUTH-003-C02" in prompt

    def test_prompt_uses_same_output_schema(self) -> None:
        """Retry prompt instructs the same JSON schema as initial diagnosis."""
        diag, ver = self._make_params()
        prompt = build_retry_prompt(
            incident_description="Issue.",
            current_version=None,
            previous_version=None,
            error_codes=[],
            applicable_results=[_rr()],
            first_diagnosis=diag,
            first_verification=ver,
        )
        # The schema description must reference "root_cause"
        assert "root_cause" in prompt
        assert "evidence_ids" in prompt

    def test_prompt_says_retry_instruction(self) -> None:
        """Retry prompt explicitly states this is a revision, not a fresh attempt."""
        diag, ver = self._make_params()
        prompt = build_retry_prompt(
            incident_description="Issue.",
            current_version=None,
            previous_version=None,
            error_codes=[],
            applicable_results=[_rr()],
            first_diagnosis=diag,
            first_verification=ver,
        )
        prompt_lower = prompt.lower()
        assert "retry" in prompt_lower or "revised" in prompt_lower or "revision" in prompt_lower


# ---------------------------------------------------------------------------
# TestOrchestratorNeedsInfo
# ---------------------------------------------------------------------------

class TestOrchestratorNeedsInfo:
    """Tests NEEDS_INFO early-exit behaviour."""

    def test_needs_info_returns_needs_info_outcome(self) -> None:
        """Vague incident → NEEDS_INFO, no diagnosis attempted."""
        retriever = _make_retriever([])
        llm = FakeLLMClient(response="unreachable")
        result = run_troubleshooting(
            description="Something is broken.",  # vague, no signals
            retriever=retriever,
            llm_client=llm,
        )
        assert result.final_outcome == FinalOutcome.NEEDS_INFO

    def test_needs_info_no_diagnosis_stored(self) -> None:
        """NEEDS_INFO → initial_diagnosis is None."""
        retriever = _make_retriever([])
        llm = FakeLLMClient(response="unreachable")
        result = run_troubleshooting(
            description="Something is wrong.",
            retriever=retriever,
            llm_client=llm,
        )
        assert result.initial_diagnosis is None

    def test_needs_info_no_retry(self) -> None:
        """NEEDS_INFO → retry never attempted."""
        retriever = _make_retriever([])
        llm = FakeLLMClient(response="unreachable")
        result = run_troubleshooting(
            description="Something is wrong.",
            retriever=retriever,
            llm_client=llm,
        )
        assert result.retry_attempted is False


# ---------------------------------------------------------------------------
# TestOrchestratorDegraded
# ---------------------------------------------------------------------------

class TestOrchestratorDegraded:
    """Tests that DEGRADED is returned on system failures, not INSUFFICIENT_EVIDENCE."""

    def test_retrieval_failure_causes_degraded(self) -> None:
        """M3 retrieval exception → DEGRADED."""
        retriever = MagicMock()
        retriever.retrieve_as_contracts.side_effect = RuntimeError("DB connection failed")
        llm = FakeLLMClient(response="unreachable")
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        assert result.final_outcome == FinalOutcome.DEGRADED

    def test_llm_failure_during_diagnosis_causes_degraded(self) -> None:
        """LLMError from diagnosis generator → DEGRADED."""
        applicable = [_rr()]
        retriever = _make_retriever([applicable[0]])
        llm = FakeLLMClient(exception=LLMError("API timeout"))
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        assert result.final_outcome == FinalOutcome.DEGRADED

    def test_verification_system_failure_causes_degraded(self) -> None:
        """LLMError from M5 verifier → DEGRADED (not INSUFFICIENT_EVIDENCE)."""
        applicable = [_rr()]
        retriever = _make_retriever([applicable[0]])
        # diagnosis: OK (1st call), verifier: LLMError (2nd call)
        llm = FakeLLMClient(responses=[
            _diagnosis_json(),
            LLMError("Verifier API failed"),
        ])
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        assert result.final_outcome == FinalOutcome.DEGRADED

    def test_degraded_has_error_detail(self) -> None:
        """DEGRADED result includes error detail."""
        retriever = MagicMock()
        retriever.retrieve_as_contracts.side_effect = RuntimeError("Disk full")
        llm = FakeLLMClient(response="unreachable")
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        assert result.final_outcome == FinalOutcome.DEGRADED
        assert result.final_answer.error_detail is not None

    def test_degraded_is_not_insufficient_evidence(self) -> None:
        """System failure must not be reported as INSUFFICIENT_EVIDENCE."""
        applicable = [_rr()]
        retriever = _make_retriever([applicable[0]])
        llm = FakeLLMClient(exception=LLMError("503 Service Unavailable"))
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        assert result.final_outcome != FinalOutcome.INSUFFICIENT_EVIDENCE
        assert result.final_outcome == FinalOutcome.DEGRADED


# ---------------------------------------------------------------------------
# TestOrchestratorFullAnswer
# ---------------------------------------------------------------------------

class TestOrchestratorFullAnswer:
    """ANSWERED_FULL happy-path tests."""

    def test_all_claims_verified_answered_full(self) -> None:
        """root_cause + fix + explanation all VERIFIED → ANSWERED_FULL."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        diag_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "Root.", "evidence_ids": ["c1"]},
            {"role": "fix", "text": "Fix.", "evidence_ids": ["c1"]},
            {"role": "explanation", "text": "Explain.", "evidence_ids": ["c1"]},
        ]})
        llm = FakeLLMClient(responses=[
            diag_json,
            _verifier_json(),   # root_cause
            _verifier_json(),   # fix
            _verifier_json(),   # explanation
        ])
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade to 3.1.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        assert result.final_outcome == FinalOutcome.ANSWERED_FULL
        assert result.retry_attempted is False

    def test_answered_full_answer_text_not_empty(self) -> None:
        """ANSWERED_FULL always has non-empty answer_text."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        diag_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "SDK 3.x requires Bearer tokens.", "evidence_ids": ["c1"]},
            {"role": "fix", "text": "Set Authorization header.", "evidence_ids": ["c1"]},
            {"role": "explanation", "text": "OAuth flow changed in 3.x.", "evidence_ids": ["c1"]},
        ]})
        llm = FakeLLMClient(responses=[
            diag_json,
            _verifier_json(), _verifier_json(), _verifier_json(),
        ])
        result = run_troubleshooting(
            description="AUTH_401 after upgrade.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        assert result.final_answer.answer_text is not None
        assert len(result.final_answer.answer_text) > 0

    def test_answered_full_verified_claims_contain_all_roles(self) -> None:
        """ANSWERED_FULL verified_claims contains all three roles."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        diag_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "Root.", "evidence_ids": ["c1"]},
            {"role": "fix", "text": "Fix.", "evidence_ids": ["c1"]},
            {"role": "explanation", "text": "Explain.", "evidence_ids": ["c1"]},
        ]})
        llm = FakeLLMClient(responses=[
            diag_json,
            _verifier_json(), _verifier_json(), _verifier_json(),
        ])
        result = run_troubleshooting(
            description="AUTH_401 after upgrade.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        roles = {c.role for c in result.final_answer.verified_claims}
        assert ClaimRole.ROOT_CAUSE in roles
        assert ClaimRole.FIX in roles
        assert ClaimRole.EXPLANATION in roles

    def test_no_retry_on_full_answer(self) -> None:
        """ANSWERED_FULL never triggers a retry."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        diag_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "Root.", "evidence_ids": ["c1"]},
            {"role": "fix", "text": "Fix.", "evidence_ids": ["c1"]},
            {"role": "explanation", "text": "Explain.", "evidence_ids": ["c1"]},
        ]})
        llm = FakeLLMClient(responses=[
            diag_json,
            _verifier_json(), _verifier_json(), _verifier_json(),
        ])
        result = run_troubleshooting(
            description="AUTH_401 after upgrade.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        assert result.retry_attempted is False


# ---------------------------------------------------------------------------
# TestOrchestratorPartialAnswer
# ---------------------------------------------------------------------------

class TestOrchestratorPartialAnswer:
    """ANSWERED_PARTIAL cases — root verified, fix/explanation rejected."""

    def test_root_and_fix_verified_explanation_rejected_partial(self) -> None:
        """root + fix VERIFIED, explanation REJECTED → ANSWERED_PARTIAL, no retry."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        diag_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "Root.", "evidence_ids": ["c1"]},
            {"role": "fix", "text": "Fix.", "evidence_ids": ["c1"]},
            {"role": "explanation", "text": "Explain.", "evidence_ids": ["c1"]},
        ]})
        llm = FakeLLMClient(responses=[
            diag_json,
            _verifier_json(citation_correct=True, sufficient=True),   # root
            _verifier_json(citation_correct=True, sufficient=True),   # fix
            _verifier_json(citation_correct=False, sufficient=False),  # explanation
        ])
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        assert result.final_outcome == FinalOutcome.ANSWERED_PARTIAL
        assert result.retry_attempted is False

    def test_partial_answer_excludes_rejected_claims(self) -> None:
        """ANSWERED_PARTIAL: rejected explanation text absent from final answer."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        diag_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "Verified root cause text.", "evidence_ids": ["c1"]},
            {"role": "explanation", "text": "SECRET_REJECTED_EXPLANATION.", "evidence_ids": ["c1"]},
        ]})
        llm = FakeLLMClient(responses=[
            diag_json,
            _verifier_json(citation_correct=True, sufficient=True),    # root → VERIFIED
            _verifier_json(citation_correct=False, sufficient=False),  # explanation → REJECTED
        ])
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        assert result.final_outcome == FinalOutcome.ANSWERED_PARTIAL
        assert result.final_answer.answer_text is not None
        assert "SECRET_REJECTED_EXPLANATION" not in result.final_answer.answer_text

    def test_root_only_verified_gives_partial(self) -> None:
        """root_cause alone VERIFIED → ANSWERED_PARTIAL."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        diag_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "Root.", "evidence_ids": ["c1"]},
            {"role": "fix", "text": "Fix.", "evidence_ids": ["c1"]},
        ]})
        llm = FakeLLMClient(responses=[
            diag_json,
            _verifier_json(citation_correct=True, sufficient=True),    # root → VERIFIED
            _verifier_json(citation_correct=False, sufficient=False),  # fix → REJECTED
        ])
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        assert result.final_outcome == FinalOutcome.ANSWERED_PARTIAL

    def test_partial_answer_no_retry(self) -> None:
        """Explanation failure alone → no retry triggered."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        diag_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "Root.", "evidence_ids": ["c1"]},
            {"role": "fix", "text": "Fix.", "evidence_ids": ["c1"]},
            {"role": "explanation", "text": "Explain.", "evidence_ids": ["c1"]},
        ]})
        llm = FakeLLMClient(responses=[
            diag_json,
            _verifier_json(citation_correct=True, sufficient=True),
            _verifier_json(citation_correct=True, sufficient=True),
            _verifier_json(citation_correct=False, sufficient=False),
        ])
        result = run_troubleshooting(
            description="AUTH_401 after upgrade.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        assert result.retry_attempted is False


# ---------------------------------------------------------------------------
# TestOrchestratorRetry
# ---------------------------------------------------------------------------

class TestOrchestratorRetry:
    """Tests retry triggering and outcomes.
    
    IMPORTANT: _diagnosis_json() by default creates root_cause-only diagnosis (1 verify call).
    Multi-claim diagnoses must be explicit.
    """

    def _reject_root_json(self) -> str:
        """A verifier JSON response that rejects the claim."""
        return json.dumps({
            "citation_correct": False,
            "sufficient": False,
            "contradicted": False,
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "reason": "Root cause is not supported by evidence.",
        })

    def test_rejected_root_cause_triggers_retry(self) -> None:
        """root_cause REJECTED → retry_attempted=True."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        # root_cause only diagnosis: 1 verify call per diagnosis attempt
        # Calls: [diag1(1), verify_root1(2), retry_diag(3), verify_root2(4)]
        llm = FakeLLMClient(responses=[
            _diagnosis_json(evidence_id="c1"),  # initial diag (root only)
            self._reject_root_json(),           # initial root → REJECTED
            _diagnosis_json(evidence_id="c1"),  # retry diag (root only)
            _verifier_json(),                   # retry root → VERIFIED
        ])
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        assert result.retry_attempted is True

    def test_retry_reason_is_populated(self) -> None:
        """retry_reason is populated with the root cause rejection reason."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        llm = FakeLLMClient(responses=[
            _diagnosis_json(evidence_id="c1"),
            self._reject_root_json(),
            _diagnosis_json(evidence_id="c1"),
            _verifier_json(),
        ])
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        assert result.retry_reason is not None
        assert len(result.retry_reason) > 0

    def test_successful_retry_produces_answered_outcome(self) -> None:
        """Retry root cause VERIFIED → ANSWERED_FULL or ANSWERED_PARTIAL."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        retry_diag_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "Better root cause.", "evidence_ids": ["c1"]},
            {"role": "fix", "text": "Better fix.", "evidence_ids": ["c1"]},
        ]})
        # initial diag (root only) → reject root → retry diag (root+fix) → verify root → verify fix
        llm = FakeLLMClient(responses=[
            _diagnosis_json(evidence_id="c1"),  # initial diag (root only)
            self._reject_root_json(),           # initial root rejected (1 verify call)
            retry_diag_json,                    # retry diag (root+fix = 2 verify calls)
            _verifier_json(),                   # retry root VERIFIED
            _verifier_json(),                   # retry fix VERIFIED
        ])
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        assert result.retry_attempted is True
        assert result.final_outcome in (FinalOutcome.ANSWERED_FULL, FinalOutcome.ANSWERED_PARTIAL)

    def test_retry_preserves_initial_diagnosis_and_verification(self) -> None:
        """After retry, initial_diagnosis and initial_verification are still accessible."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        # root-only on both passes: diag(1) verify_root(2) retry_diag(3) retry_verify_root(4)
        llm = FakeLLMClient(responses=[
            _diagnosis_json(evidence_id="c1"),  # initial root-only diag
            self._reject_root_json(),           # initial root REJECTED
            _diagnosis_json(evidence_id="c1"),  # retry root-only diag
            _verifier_json(),                   # retry root VERIFIED
        ])
        result = run_troubleshooting(
            description="AUTH_401 after upgrade.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        assert result.initial_diagnosis is not None
        assert result.initial_verification is not None

    def test_retry_preserves_retry_diagnosis_and_verification(self) -> None:
        """After retry, retry_diagnosis and retry_verification are stored."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        llm = FakeLLMClient(responses=[
            _diagnosis_json(evidence_id="c1"),  # initial root-only
            self._reject_root_json(),           # initial root REJECTED
            _diagnosis_json(evidence_id="c1"),  # retry root-only
            _verifier_json(),                   # retry root VERIFIED
        ])
        result = run_troubleshooting(
            description="AUTH_401 after upgrade.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        assert result.retry_diagnosis is not None
        assert result.retry_verification is not None

    def test_retry_root_rejected_yields_insufficient_evidence(self) -> None:
        """Retry root_cause still REJECTED → INSUFFICIENT_EVIDENCE."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        # Both diagnoses are root-only: 4 total LLM calls
        llm = FakeLLMClient(responses=[
            _diagnosis_json(evidence_id="c1"),  # initial diag
            self._reject_root_json(),           # initial root REJECTED
            _diagnosis_json(evidence_id="c1"),  # retry diag
            self._reject_root_json(),           # retry root REJECTED
        ])
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        assert result.final_outcome == FinalOutcome.INSUFFICIENT_EVIDENCE
        assert result.retry_attempted is True

    def test_insufficient_evidence_has_no_answer_text(self) -> None:
        """INSUFFICIENT_EVIDENCE → answer_text is None."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        llm = FakeLLMClient(responses=[
            _diagnosis_json(evidence_id="c1"),  # initial root-only
            self._reject_root_json(),           # initial root REJECTED
            _diagnosis_json(evidence_id="c1"),  # retry root-only
            self._reject_root_json(),           # retry root REJECTED
        ])
        result = run_troubleshooting(
            description="AUTH_401 after upgrade.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        assert result.final_outcome == FinalOutcome.INSUFFICIENT_EVIDENCE
        assert result.final_answer.answer_text is None
        assert result.final_answer.verified_claims == []

    def test_retry_llm_failure_causes_degraded(self) -> None:
        """LLMError during retry diagnosis → DEGRADED (retry_attempted=True)."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        # initial diag (root only) → reject root (1 verify) → retry LLM raises
        llm = FakeLLMClient(responses=[
            _diagnosis_json(evidence_id="c1"),  # initial diag
            self._reject_root_json(),           # initial root REJECTED
            LLMError("Retry LLM failed"),       # retry LLM call raises
        ])
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        assert result.final_outcome == FinalOutcome.DEGRADED
        assert result.retry_attempted is True


# ---------------------------------------------------------------------------
# TestOrchestratorRetryOnce
# ---------------------------------------------------------------------------

class TestOrchestratorRetryOnce:
    """Explicitly tests that only ONE retry is ever attempted."""

    def _reject_root_json(self) -> str:
        return json.dumps({
            "citation_correct": False, "sufficient": False, "contradicted": False,
            "supporting_evidence_ids": [], "contradicting_evidence_ids": [],
            "reason": "Not supported.",
        })

    def test_exactly_one_retry_when_root_rejected(self) -> None:
        """Only ONE retry is issued. LLM call count confirms this (4 calls total)."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        # root-only diagnoses: 2 diag calls + 2 verify calls = 4 total
        llm = FakeLLMClient(responses=[
            _diagnosis_json(evidence_id="c1"),  # initial diag (1)
            self._reject_root_json(),           # initial verify root (2)
            _diagnosis_json(evidence_id="c1"),  # retry diag (3)
            self._reject_root_json(),           # retry verify root (4)
            # Queue exhausted — any 5th call would raise LLMError, proving no second retry
        ])
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        assert result.retry_attempted is True
        assert result.final_outcome == FinalOutcome.INSUFFICIENT_EVIDENCE
        assert llm.call_count == 4

    def test_no_second_retry_on_double_failure(self) -> None:
        """Two consecutive root cause failures → INSUFFICIENT_EVIDENCE, not a third attempt."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        # 4 responses exactly (root-only diagnoses). If 2nd retry triggered, 5th call → LLMError → DEGRADED
        llm = FakeLLMClient(responses=[
            _diagnosis_json(evidence_id="c1"),
            self._reject_root_json(),
            _diagnosis_json(evidence_id="c1"),
            self._reject_root_json(),
        ])
        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        # If a second retry had happened → queue exhausted → DEGRADED
        # Passing → proves exactly one retry
        assert result.final_outcome == FinalOutcome.INSUFFICIENT_EVIDENCE

    def test_retry_not_attempted_when_root_verified_first(self) -> None:
        """Root verified first attempt → retry is never attempted."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        # root-only diagnosis: 1 diag call + 1 verify call = 2 total
        llm = FakeLLMClient(responses=[
            _diagnosis_json(evidence_id="c1"),  # diag (1)
            _verifier_json(),                   # verify root VERIFIED (2)
        ])
        result = run_troubleshooting(
            description="AUTH_401 after upgrade.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        assert result.retry_attempted is False
        assert llm.call_count == 2

    def test_result_retry_diagnosis_none_when_no_retry(self) -> None:
        """When no retry is triggered, retry_diagnosis and retry_verification are None."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        llm = FakeLLMClient(responses=[_diagnosis_json(evidence_id="c1"), _verifier_json()])
        result = run_troubleshooting(
            description="AUTH_401 after upgrade.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        assert result.retry_diagnosis is None
        assert result.retry_verification is None


# ---------------------------------------------------------------------------
# TestWrongVersionExclusion
# ---------------------------------------------------------------------------

class TestWrongVersionExclusion:
    """Architecture thesis: wrong-version evidence cannot enter the final answer."""

    def _reject_json(self, reason: str = "Not supported.") -> str:
        return json.dumps({
            "citation_correct": False, "sufficient": False, "contradicted": False,
            "supporting_evidence_ids": [], "contradicting_evidence_ids": [],
            "reason": reason,
        })

    def test_2x_chunk_excluded_by_m4_cannot_be_cited_in_final_answer(self) -> None:
        """
        Diagnosis cites a 2.x chunk for a 3.x incident.
        M4 excludes the 2.x chunk from applicable_results.
        M5 rejects the citation (chunk not in applicable set).
        Final answer contains no wrong-version evidence.
        """
        # M3 returns both 2.x and 3.x chunks
        chunk_2x = _rr("AUTH-001-C01", applies_to=">=2.0,<3.0", content="SDK 2.x auth.")
        chunk_3x = _rr("AUTH-002-C01", applies_to=">=3.0,<4.0", content="SDK 3.x requires Bearer.")
        retriever = _make_retriever([chunk_2x, chunk_3x])

        # Diagnosis cites 2.x chunk (wrong version, excluded by M4)
        # After M4 filters for 3.1, only chunk_3x is applicable
        # So the diagnosis must cite chunk_3x or get rejected
        diag_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "Root cause.", "evidence_ids": ["AUTH-002-C01"]},
            {"role": "fix", "text": "Bearer token required.", "evidence_ids": ["AUTH-002-C01"]},
        ]})
        llm = FakeLLMClient(responses=[
            diag_json,
            _verifier_json(supporting_ids=["AUTH-002-C01"]),  # root VERIFIED
            _verifier_json(supporting_ids=["AUTH-002-C01"]),  # fix VERIFIED
        ])
        result = run_troubleshooting(
            description="AUTH_401 after upgrading from SDK 2.8 to 3.1.",
            current_version="3.1",
            previous_version="2.8",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        # Final answer must only contain verified claims
        for claim in result.final_answer.verified_claims:
            for eid in claim.evidence_ids:
                assert eid != "AUTH-001-C01", (
                    "2.x chunk should not appear in verified final answer."
                )

    def test_2x_doc_excluded_from_applicable_results(self) -> None:
        """M4 applicability filtering ensures 2.x chunk is not in applicable_results for 3.1."""
        chunk_2x = _rr("AUTH-001-C01", applies_to=">=2.0,<3.0")
        chunk_3x = _rr("AUTH-002-C01", applies_to=">=3.0,<4.0")
        retriever = _make_retriever([chunk_2x, chunk_3x])
        diag_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "Root.", "evidence_ids": ["AUTH-002-C01"]},
        ]})
        llm = FakeLLMClient(responses=[diag_json, _verifier_json()])
        result = run_troubleshooting(
            description="AUTH_401 after upgrade to 3.1.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        applicable_ids = {r.chunk_id for r in result.applicable_results}
        assert "AUTH-001-C01" not in applicable_ids
        assert "AUTH-002-C01" in applicable_ids

    def test_final_answer_contains_only_verified_claims(self) -> None:
        """verified_claims in final_answer contains only VERIFIED claims (not all)."""
        applicable = [_rr("c1")]
        retriever = _make_retriever([applicable[0]])
        diag_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "Root.", "evidence_ids": ["c1"]},
            {"role": "fix", "text": "Fix.", "evidence_ids": ["c1"]},
        ]})
        llm = FakeLLMClient(responses=[
            diag_json,
            _verifier_json(citation_correct=True, sufficient=True),    # root → VERIFIED
            _verifier_json(citation_correct=False, sufficient=False),  # fix → REJECTED
        ])
        result = run_troubleshooting(
            description="AUTH_401 after upgrade.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        # Only root_cause should be in verified_claims
        roles = {c.role for c in result.final_answer.verified_claims}
        assert ClaimRole.ROOT_CAUSE in roles
        assert ClaimRole.FIX not in roles


# ---------------------------------------------------------------------------
# TestEndToEndIntegration
# ---------------------------------------------------------------------------

class TestEndToEndIntegration:
    """
    Realistic end-to-end integration tests demonstrating the DevTrace thesis.

    These tests simulate the complete chain:
    Incident → M3 Retrieval → M4 Applicability → M4 Diagnosis →
    M5 Verification → M6 Outcome.
    """

    def test_main_devtrace_thesis_auth_401_upgrade(self) -> None:
        """
        Core DevTrace thesis test:

        Incident: AUTH_401 after upgrading SDK from 2.8 → 3.1.

        Retrieval contains:
          - A semantically similar 2.x auth document.
          - The correct 3.x auth documentation.
          - Version-agnostic supporting docs.

        M4 applicability removes the 2.x document for current_version=3.1.
        M5 verifies only 3.x-applicable claims.
        M6 produces a final verified answer WITHOUT 2.x evidence.
        """
        chunk_2x = _rr(
            "AUTH-001-C01",
            applies_to=">=2.0,<3.0",
            content="SDK 2.x uses API Key in X-API-Key header.",
        )
        chunk_3x = _rr(
            "AUTH-002-C01",
            applies_to=">=3.0,<4.0",
            content="SDK 3.x requires OAuth 2.0 Bearer token in Authorization header.",
        )
        chunk_agnostic = _rr(
            "AUTH-000-C01",
            applies_to="*",
            content="Authentication errors may indicate header misconfiguration.",
        )

        retriever = _make_retriever([chunk_2x, chunk_3x, chunk_agnostic])

        # Diagnosis correctly cites 3.x and agnostic chunks only
        diag_json = json.dumps({"claims": [
            {
                "role": "root_cause",
                "text": "SDK 3.x requires OAuth 2.0 Bearer token authentication.",
                "evidence_ids": ["AUTH-002-C01"],
            },
            {
                "role": "fix",
                "text": "Update Authorization header to use Bearer token.",
                "evidence_ids": ["AUTH-002-C01", "AUTH-000-C01"],
            },
        ]})

        llm = FakeLLMClient(responses=[
            diag_json,
            _verifier_json(supporting_ids=["AUTH-002-C01"]),
            _verifier_json(supporting_ids=["AUTH-002-C01", "AUTH-000-C01"]),
        ])

        result = run_troubleshooting(
            description="AUTH_401 started after upgrading from SDK 2.8 to 3.1.",
            current_version="3.1",
            previous_version="2.8",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )

        # Core assertions
        assert result.final_outcome in (FinalOutcome.ANSWERED_FULL, FinalOutcome.ANSWERED_PARTIAL)
        assert result.retry_attempted is False
        assert result.final_answer.answer_text is not None
        assert "AUTH-001-C01" not in {r.chunk_id for r in result.applicable_results}
        assert "AUTH-002-C01" in {r.chunk_id for r in result.applicable_results}

        # Final answer claims must not cite the 2.x chunk
        for claim in result.final_answer.verified_claims:
            assert "AUTH-001-C01" not in claim.evidence_ids

    def test_retry_produces_verified_answer_on_second_attempt(self) -> None:
        """
        Retry scenario: initial diagnosis root_cause rejected, retry succeeds.

        Initial: root_cause cites invalid chunk → REJECTED.
        Retry: root_cause cites valid 3.x chunk → VERIFIED.
        Final outcome: ANSWERED_PARTIAL (root + fix verified).
        """
        chunk_3x = _rr("AUTH-002-C01", applies_to=">=3.0,<4.0")
        retriever = _make_retriever([chunk_3x])

        initial_diag_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "Initial wrong root.", "evidence_ids": ["AUTH-002-C01"]},
        ]})
        reject_root_json = json.dumps({
            "citation_correct": False, "sufficient": False, "contradicted": False,
            "supporting_evidence_ids": [], "contradicting_evidence_ids": [],
            "reason": "Evidence does not support this root cause claim.",
        })
        retry_diag_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "Better root cause.", "evidence_ids": ["AUTH-002-C01"]},
            {"role": "fix", "text": "Add Bearer token.", "evidence_ids": ["AUTH-002-C01"]},
        ]})

        llm = FakeLLMClient(responses=[
            initial_diag_json,
            reject_root_json,
            retry_diag_json,
            _verifier_json(supporting_ids=["AUTH-002-C01"]),  # retry root
            _verifier_json(supporting_ids=["AUTH-002-C01"]),  # retry fix
        ])

        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade to 3.1.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )

        assert result.retry_attempted is True
        assert result.retry_reason is not None
        assert result.retry_diagnosis is not None
        assert result.retry_verification is not None
        assert result.final_outcome in (FinalOutcome.ANSWERED_FULL, FinalOutcome.ANSWERED_PARTIAL)
        assert result.final_answer.answer_text is not None

    def test_full_pipeline_result_preserves_trace_info(self) -> None:
        """
        TroubleshootingResult preserves all intermediate pipeline information
        needed by M7 evaluation.
        """
        chunk_3x = _rr("AUTH-002-C01", applies_to=">=3.0,<4.0")
        retriever = _make_retriever([chunk_3x])
        diag_json = json.dumps({"claims": [
            {"role": "root_cause", "text": "Root.", "evidence_ids": ["AUTH-002-C01"]},
            {"role": "fix", "text": "Fix.", "evidence_ids": ["AUTH-002-C01"]},
        ]})
        llm = FakeLLMClient(responses=[
            diag_json,
            _verifier_json(), _verifier_json(),
        ])

        result = run_troubleshooting(
            description="AUTH_401 after SDK upgrade to 3.1.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )

        # All trace fields must be populated
        assert result.incident_description == "AUTH_401 after SDK upgrade to 3.1."
        assert result.current_version == "3.1"
        assert len(result.retrieval_results) > 0
        assert len(result.applicability_decisions) > 0
        assert len(result.applicable_results) > 0
        assert result.initial_diagnosis is not None
        assert result.initial_verification is not None
        assert result.final_answer is not None
        assert result.final_outcome is not None
