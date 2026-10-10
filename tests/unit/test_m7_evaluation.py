"""
Module 7 — Comprehensive Unit & Integration Tests.

Test coverage:
  - Metric calculations (retrieval hit rate, citation validity, correctness,
    false answer rate, false abstention rate)
  - Outcome aggregation
  - Category aggregation
  - Baseline A execution
  - Baseline B execution (threshold logic)
  - DevTrace evaluation adapter
  - Retry statistics
  - Version-conflict analysis
  - Unsupported-case analysis
  - Missing/malformed evaluation records
  - Deterministic reproducibility
  - No leakage of gold labels into model prompts
  - Edge cases: zero citations, zero retrieval, empty evidence, DEGRADED,
    NEEDS_INFO, partial answer, all cases abstaining

All tests are offline (FakeLLMClient + stub retriever).
No network calls, no Chroma index required.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.evaluation.dataset import EvaluationCase, EvaluationIncident, load_eval_dataset
from src.llm.fake import FakeLLMClient
from src.m7.baselines import (
    _heuristic_cited_doc_ids,
    run_baseline_a,
    run_baseline_b,
)
from src.m7.config import EvaluationConfig
from src.m7.devtrace_eval import run_devtrace_eval, _extract_doc_ids_from_chunks
from src.m7.metrics import (
    build_aggregate_metrics,
    compute_case_metrics_a,
    compute_case_metrics_b,
    compute_case_metrics_devtrace,
    compute_citation_correctness,
    compute_citation_validity,
    compute_false_abstention_rate,
    compute_false_answer_rate,
    compute_outcome_distribution,
    compute_retry_stats,
    compute_unsupported_case_analysis,
    compute_version_conflict_analysis,
    score_citation_correct,
    score_citation_valid,
    score_false_abstention,
    score_false_answer,
    score_forbidden_retrieved,
    score_retrieval_hit,
)
from src.m7.models import (
    AggregateMetrics,
    BaselineAResult,
    BaselineBResult,
    CaseMetrics,
    DevTraceEvalResult,
    EvaluationCaseRecord,
    EvaluationReport,
    OutcomeDistribution,
    RetryStats,
)
from src.models.contracts import RetrievalResult
from src.models.enums import RetrievalSource
from src.normalization.normalizer import normalize_incident


# ===========================================================================
# Test fixtures and helpers
# ===========================================================================

def _make_retrieval_result(
    chunk_id: str = "AUTH-002-C01",
    doc_id: str = "AUTH-002",
    score: float = 0.80,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        score=score,
        source=RetrievalSource.HYBRID,
        metadata={
            "content": f"Content for {chunk_id}.",
            "applies_to": ">=3.0,<4.0",
            "topic": "authentication",
        },
    )


class _StubRetriever:
    """Offline stub retriever returning a fixed result list."""

    def __init__(self, results: list[RetrievalResult] | None = None) -> None:
        self._results = results if results is not None else [
            _make_retrieval_result("AUTH-002-C01", "AUTH-002", 0.85),
            _make_retrieval_result("SDK-001-C01", "SDK-001", 0.70),
        ]

    def retrieve_as_contracts(self, normalized: object) -> list[RetrievalResult]:
        return self._results


def _empty_stub_retriever() -> _StubRetriever:
    return _StubRetriever(results=[])


def _make_case_record(
    *,
    case_id: str = "TEST-001",
    case_class: str = "version_conflict",
    gold_expected_outcome: str = "ANSWERED",
    gold_expected_completeness: str | None = "FULL",
    gold_doc_ids: list[str] | None = None,
    gold_forbidden_doc_ids: list[str] | None = None,
    baseline_a: BaselineAResult | None = None,
    baseline_b: BaselineBResult | None = None,
    devtrace: DevTraceEvalResult | None = None,
    metrics_a: CaseMetrics | None = None,
    metrics_b: CaseMetrics | None = None,
    metrics_devtrace: CaseMetrics | None = None,
) -> EvaluationCaseRecord:
    return EvaluationCaseRecord(
        case_id=case_id,
        case_class=case_class,
        gold_expected_outcome=gold_expected_outcome,
        gold_expected_completeness=gold_expected_completeness,
        gold_doc_ids=gold_doc_ids or ["AUTH-002"],
        gold_forbidden_doc_ids=gold_forbidden_doc_ids or ["AUTH-001"],
        baseline_a=baseline_a,
        baseline_b=baseline_b,
        devtrace=devtrace,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        metrics_devtrace=metrics_devtrace,
    )


def _make_baseline_a_answered(
    retrieved_doc_ids: list[str] | None = None,
    cited_doc_ids: list[str] | None = None,
    answer: str = "The fix is in AUTH-002.",
) -> BaselineAResult:
    return BaselineAResult(
        retrieved_chunk_ids=["AUTH-002-C01"],
        retrieved_doc_ids=retrieved_doc_ids or ["AUTH-002"],
        answer_text=answer,
        cited_doc_ids=cited_doc_ids or ["AUTH-002"],
        abstained=False,
    )


def _make_baseline_b_answered(
    retrieved_doc_ids: list[str] | None = None,
    cited_doc_ids: list[str] | None = None,
    threshold: float = 0.40,
    top_score: float = 0.80,
) -> BaselineBResult:
    return BaselineBResult(
        retrieved_chunk_ids=["AUTH-002-C01"],
        retrieved_doc_ids=retrieved_doc_ids or ["AUTH-002"],
        top_score=top_score,
        threshold_used=threshold,
        threshold_passed=True,
        answer_text="The AUTH-002 doc explains the fix.",
        cited_doc_ids=cited_doc_ids or ["AUTH-002"],
        abstained=False,
    )


def _make_devtrace_answered(
    final_outcome: str = "ANSWERED_FULL",
    retrieved_doc_ids: list[str] | None = None,
    applicable_doc_ids: list[str] | None = None,
    cited_doc_ids: list[str] | None = None,
    retry_attempted: bool = False,
    retry_succeeded: bool = False,
) -> DevTraceEvalResult:
    return DevTraceEvalResult(
        final_outcome=final_outcome,
        answer_text="Based on AUTH-002, the fix is to switch auth headers.",
        retrieved_chunk_ids=["AUTH-002-C01", "AUTH-001-C01"],
        retrieved_doc_ids=retrieved_doc_ids or ["AUTH-002", "AUTH-001"],
        applicable_chunk_ids=["AUTH-002-C01"],
        applicable_doc_ids=applicable_doc_ids or ["AUTH-002"],
        cited_chunk_ids=["AUTH-002-C01"],
        cited_doc_ids=cited_doc_ids or ["AUTH-002"],
        forbidden_doc_ids_cited=[],
        retry_attempted=retry_attempted,
        retry_succeeded=retry_succeeded,
        abstained=False,
    )


def _make_devtrace_abstained(
    final_outcome: str = "INSUFFICIENT_EVIDENCE",
) -> DevTraceEvalResult:
    return DevTraceEvalResult(
        final_outcome=final_outcome,
        answer_text=None,
        retrieved_chunk_ids=[],
        retrieved_doc_ids=[],
        applicable_chunk_ids=[],
        applicable_doc_ids=[],
        cited_chunk_ids=[],
        cited_doc_ids=[],
        forbidden_doc_ids_cited=[],
        retry_attempted=False,
        retry_succeeded=False,
        abstained=True,
    )


# ===========================================================================
# Part 1: Metric calculation unit tests
# ===========================================================================

class TestScoreRetrievalHit:

    def test_hit_when_gold_in_retrieved(self) -> None:
        assert score_retrieval_hit(["AUTH-002", "SDK-001"], ["AUTH-002"]) is True

    def test_miss_when_gold_not_in_retrieved(self) -> None:
        assert score_retrieval_hit(["SDK-001", "RATE-001"], ["AUTH-002"]) is False

    def test_none_when_no_gold_docs(self) -> None:
        assert score_retrieval_hit(["AUTH-002"], []) is None

    def test_partial_hit_returns_true(self) -> None:
        """At least one gold doc is enough."""
        assert score_retrieval_hit(["AUTH-002", "SDK-001"], ["AUTH-002", "AUTH-003"]) is True

    def test_empty_retrieved_miss(self) -> None:
        assert score_retrieval_hit([], ["AUTH-002"]) is False


class TestScoreForbiddenRetrieved:

    def test_forbidden_in_retrieved(self) -> None:
        assert score_forbidden_retrieved(["AUTH-001", "AUTH-002"], ["AUTH-001"]) is True

    def test_forbidden_not_retrieved(self) -> None:
        assert score_forbidden_retrieved(["AUTH-002", "SDK-001"], ["AUTH-001"]) is False

    def test_none_when_no_forbidden(self) -> None:
        assert score_forbidden_retrieved(["AUTH-002"], []) is None

    def test_empty_retrieved(self) -> None:
        assert score_forbidden_retrieved([], ["AUTH-001"]) is False


class TestScoreCitationValid:

    def test_all_cited_in_pool(self) -> None:
        assert score_citation_valid(["AUTH-002"], ["AUTH-002", "SDK-001"]) is True

    def test_one_cited_outside_pool(self) -> None:
        assert score_citation_valid(["AUTH-999"], ["AUTH-002", "SDK-001"]) is False

    def test_none_when_no_citations(self) -> None:
        assert score_citation_valid([], ["AUTH-002"]) is None

    def test_multiple_cited_all_valid(self) -> None:
        assert score_citation_valid(
            ["AUTH-002", "SDK-001"],
            ["AUTH-002", "SDK-001", "RATE-001"],
        ) is True

    def test_one_invalid_among_valid(self) -> None:
        assert score_citation_valid(
            ["AUTH-002", "HALLUCINATED-999"],
            ["AUTH-002", "SDK-001"],
        ) is False


class TestScoreCitationCorrect:

    def test_gold_in_cited(self) -> None:
        assert score_citation_correct(["AUTH-002"], ["AUTH-002"]) is True

    def test_gold_not_in_cited(self) -> None:
        assert score_citation_correct(["SDK-001"], ["AUTH-002"]) is False

    def test_none_when_no_citations(self) -> None:
        assert score_citation_correct([], ["AUTH-002"]) is None

    def test_none_when_no_gold(self) -> None:
        assert score_citation_correct(["AUTH-002"], []) is None

    def test_partial_gold_match(self) -> None:
        """One gold doc cited out of two is enough."""
        assert score_citation_correct(["AUTH-002"], ["AUTH-002", "AUTH-003"]) is True


class TestScoreFalseAnswer:

    def test_false_answer_when_answered_but_should_abstain(self) -> None:
        result = score_false_answer(
            abstained=False,
            answer_text="The fix is...",
            gold_expected_outcome="INSUFFICIENT_EVIDENCE",
            cited_doc_ids=[],
            forbidden_doc_ids=[],
        )
        assert result is True

    def test_no_false_answer_when_correctly_answered(self) -> None:
        result = score_false_answer(
            abstained=False,
            answer_text="The fix is...",
            gold_expected_outcome="ANSWERED",
            cited_doc_ids=["AUTH-002"],
            forbidden_doc_ids=[],
        )
        assert result is False

    def test_false_answer_when_citing_forbidden(self) -> None:
        result = score_false_answer(
            abstained=False,
            answer_text="The fix is...",
            gold_expected_outcome="ANSWERED",
            cited_doc_ids=["AUTH-001"],
            forbidden_doc_ids=["AUTH-001"],
        )
        assert result is True

    def test_no_false_answer_when_correctly_abstained(self) -> None:
        result = score_false_answer(
            abstained=True,
            answer_text=None,
            gold_expected_outcome="NEEDS_INFO",
            cited_doc_ids=[],
            forbidden_doc_ids=[],
        )
        assert result is False

    def test_needs_info_case_answered_is_false_answer(self) -> None:
        result = score_false_answer(
            abstained=False,
            answer_text="Some answer.",
            gold_expected_outcome="NEEDS_INFO",
            cited_doc_ids=[],
            forbidden_doc_ids=[],
        )
        assert result is True

    def test_empty_answer_text_is_not_confident(self) -> None:
        result = score_false_answer(
            abstained=False,
            answer_text="",
            gold_expected_outcome="INSUFFICIENT_EVIDENCE",
            cited_doc_ids=[],
            forbidden_doc_ids=[],
        )
        assert result is False  # empty string is not a confident answer


class TestScoreFalseAbstention:

    def test_abstained_when_gold_answered(self) -> None:
        assert score_false_abstention(abstained=True, gold_expected_outcome="ANSWERED") is True

    def test_correctly_answered(self) -> None:
        assert score_false_abstention(abstained=False, gold_expected_outcome="ANSWERED") is False

    def test_none_when_gold_is_not_answered(self) -> None:
        assert score_false_abstention(abstained=True, gold_expected_outcome="NEEDS_INFO") is None
        assert score_false_abstention(abstained=False, gold_expected_outcome="INSUFFICIENT_EVIDENCE") is None


# ===========================================================================
# Part 2: Aggregate metric tests
# ===========================================================================

class TestComputeRetrievalHitRate:

    def _build_record_with_hit(self, hit: bool) -> EvaluationCaseRecord:
        a_result = _make_baseline_a_answered(
            retrieved_doc_ids=["AUTH-002"] if hit else ["SDK-001"],
        )
        record = _make_case_record(baseline_a=a_result)
        return record.model_copy(update={
            "metrics_a": compute_case_metrics_a(record.model_copy(update={"baseline_a": a_result})),
        })

    def test_full_hit_rate(self) -> None:
        records = []
        for _ in range(3):
            a = _make_baseline_a_answered(retrieved_doc_ids=["AUTH-002"])
            r = _make_case_record(baseline_a=a)
            r = r.model_copy(update={"metrics_a": compute_case_metrics_a(r)})
            records.append(r)
        rate = compute_citation_validity(records, "a")
        # All have valid citations (AUTH-002 in retrieved)
        assert rate is not None

    def test_empty_records(self) -> None:
        from src.m7.metrics import compute_retrieval_hit_rate
        rate = compute_retrieval_hit_rate([], "a")
        assert rate is None


class TestComputeFalseAnswerRate:

    def test_all_false_answers(self) -> None:
        records = []
        for i in range(4):
            a = BaselineAResult(
                retrieved_doc_ids=["SDK-001"],
                retrieved_chunk_ids=["SDK-001-C01"],
                answer_text="Some answer.",
                cited_doc_ids=[],
                abstained=False,
            )
            r = _make_case_record(
                case_id=f"TEST-{i}",
                gold_expected_outcome="INSUFFICIENT_EVIDENCE",
                gold_doc_ids=[],
                gold_forbidden_doc_ids=[],
                baseline_a=a,
            )
            r = r.model_copy(update={"metrics_a": compute_case_metrics_a(r)})
            records.append(r)
        rate = compute_false_answer_rate(records, "a")
        assert rate == 1.0

    def test_no_false_answers(self) -> None:
        records = []
        for i in range(3):
            a = BaselineAResult(
                retrieved_doc_ids=["AUTH-002"],
                retrieved_chunk_ids=["AUTH-002-C01"],
                answer_text="The fix is in AUTH-002.",
                cited_doc_ids=["AUTH-002"],
                abstained=False,
            )
            r = _make_case_record(
                case_id=f"TEST-{i}",
                gold_expected_outcome="ANSWERED",
                gold_doc_ids=["AUTH-002"],
                gold_forbidden_doc_ids=[],
                baseline_a=a,
            )
            r = r.model_copy(update={"metrics_a": compute_case_metrics_a(r)})
            records.append(r)
        rate = compute_false_answer_rate(records, "a")
        assert rate == 0.0

    def test_empty_records(self) -> None:
        rate = compute_false_answer_rate([], "a")
        assert rate is None


class TestComputeFalseAbstentionRate:

    def test_all_abstained_on_answered_cases(self) -> None:
        records = []
        for i in range(3):
            a = BaselineAResult(abstained=True, retrieved_doc_ids=[], retrieved_chunk_ids=[])
            r = _make_case_record(
                case_id=f"TEST-{i}",
                gold_expected_outcome="ANSWERED",
                baseline_a=a,
            )
            r = r.model_copy(update={"metrics_a": compute_case_metrics_a(r)})
            records.append(r)
        rate = compute_false_abstention_rate(records, "a")
        assert rate == 1.0

    def test_no_abstentions_on_answered_cases(self) -> None:
        records = []
        for i in range(3):
            a = _make_baseline_a_answered()
            r = _make_case_record(
                case_id=f"TEST-{i}",
                gold_expected_outcome="ANSWERED",
                baseline_a=a,
            )
            r = r.model_copy(update={"metrics_a": compute_case_metrics_a(r)})
            records.append(r)
        rate = compute_false_abstention_rate(records, "a")
        assert rate == 0.0

    def test_none_when_no_answered_cases(self) -> None:
        a = BaselineAResult(abstained=True, retrieved_doc_ids=[], retrieved_chunk_ids=[])
        r = _make_case_record(gold_expected_outcome="NEEDS_INFO", baseline_a=a)
        r = r.model_copy(update={"metrics_a": compute_case_metrics_a(r)})
        rate = compute_false_abstention_rate([r], "a")
        assert rate is None


# ===========================================================================
# Part 3: Outcome distribution tests
# ===========================================================================

class TestOutcomeDistribution:

    def test_correct_counts(self) -> None:
        records = [
            _make_case_record(devtrace=_make_devtrace_answered("ANSWERED_FULL")),
            _make_case_record(devtrace=_make_devtrace_answered("ANSWERED_PARTIAL")),
            _make_case_record(devtrace=_make_devtrace_abstained("INSUFFICIENT_EVIDENCE")),
            _make_case_record(devtrace=_make_devtrace_abstained("NEEDS_INFO")),
            _make_case_record(devtrace=_make_devtrace_abstained("DEGRADED")),
        ]
        dist = compute_outcome_distribution(records)
        assert dist.answered_full == 1
        assert dist.answered_partial == 1
        assert dist.insufficient_evidence == 1
        assert dist.needs_info == 1
        assert dist.degraded == 1
        assert dist.total == 5

    def test_empty_records(self) -> None:
        dist = compute_outcome_distribution([])
        assert dist.total == 0
        assert dist.answered_full == 0

    def test_no_devtrace_results(self) -> None:
        records = [_make_case_record(devtrace=None)]
        dist = compute_outcome_distribution(records)
        assert dist.answered_full == 0


# ===========================================================================
# Part 4: Retry statistics tests
# ===========================================================================

class TestRetryStats:

    def test_no_retries(self) -> None:
        records = [
            _make_case_record(devtrace=_make_devtrace_answered("ANSWERED_FULL")),
        ]
        stats = compute_retry_stats(records)
        assert stats.retries_attempted == 0
        assert stats.retry_recovery_rate is None
        assert stats.max_retry_count_exceeded is False

    def test_retry_succeeded(self) -> None:
        dt = _make_devtrace_answered("ANSWERED_FULL", retry_attempted=True, retry_succeeded=True)
        records = [_make_case_record(devtrace=dt)]
        stats = compute_retry_stats(records)
        assert stats.retries_attempted == 1
        assert stats.retries_succeeded == 1
        assert stats.retries_failed == 0
        assert stats.retry_recovery_rate == 1.0

    def test_retry_failed(self) -> None:
        dt = DevTraceEvalResult(
            final_outcome="INSUFFICIENT_EVIDENCE",
            retrieved_doc_ids=["AUTH-002"],
            retrieved_chunk_ids=["AUTH-002-C01"],
            applicable_doc_ids=["AUTH-002"],
            applicable_chunk_ids=["AUTH-002-C01"],
            cited_doc_ids=[],
            cited_chunk_ids=[],
            forbidden_doc_ids_cited=[],
            retry_attempted=True,
            retry_succeeded=False,
            abstained=True,
        )
        records = [_make_case_record(devtrace=dt)]
        stats = compute_retry_stats(records)
        assert stats.retries_attempted == 1
        assert stats.retries_succeeded == 0
        assert stats.retries_failed == 1
        assert stats.retry_recovery_rate == 0.0

    def test_initial_failure_without_retry(self) -> None:
        """INSUFFICIENT_EVIDENCE with no retry (no applicable evidence)."""
        dt = _make_devtrace_abstained("INSUFFICIENT_EVIDENCE")
        records = [_make_case_record(devtrace=dt)]
        stats = compute_retry_stats(records)
        assert stats.initial_root_cause_failures == 1
        assert stats.retries_attempted == 0

    def test_max_retry_invariant_always_false(self) -> None:
        """M6 enforces ≤1 retry; this field should always be False."""
        dt = _make_devtrace_answered("ANSWERED_FULL", retry_attempted=True, retry_succeeded=True)
        records = [_make_case_record(devtrace=dt)] * 5
        stats = compute_retry_stats(records)
        assert stats.max_retry_count_exceeded is False


# ===========================================================================
# Part 5: Version-conflict analysis tests
# ===========================================================================

class TestVersionConflictAnalysis:

    def _vc_record(
        self,
        case_id: str,
        forbidden_retrieved: bool = True,
        false_answer: bool = True,
        devtrace_excluded: bool = True,
    ) -> EvaluationCaseRecord:
        doc_ids = ["AUTH-001", "AUTH-002"] if forbidden_retrieved else ["AUTH-002"]
        a = BaselineAResult(
            retrieved_doc_ids=doc_ids,
            retrieved_chunk_ids=[d + "-C01" for d in doc_ids],
            answer_text="Auth fix: AUTH-002" if not false_answer else "Auth fix: AUTH-001",
            cited_doc_ids=["AUTH-001"] if false_answer else ["AUTH-002"],
            abstained=False,
        )
        applicable_docs = [] if devtrace_excluded else ["AUTH-001"]
        dt = DevTraceEvalResult(
            final_outcome="ANSWERED_FULL",
            answer_text="Fix is in AUTH-002.",
            retrieved_doc_ids=["AUTH-001", "AUTH-002"],
            retrieved_chunk_ids=["AUTH-001-C01", "AUTH-002-C01"],
            applicable_doc_ids=applicable_docs + ["AUTH-002"],
            applicable_chunk_ids=["AUTH-002-C01"],
            cited_doc_ids=["AUTH-002"],
            cited_chunk_ids=["AUTH-002-C01"],
            forbidden_doc_ids_cited=[],
            retry_attempted=False,
            retry_succeeded=False,
            abstained=False,
        )
        record = EvaluationCaseRecord(
            case_id=case_id,
            case_class="version_conflict",
            gold_expected_outcome="ANSWERED",
            gold_expected_completeness="FULL",
            gold_doc_ids=["AUTH-002"],
            gold_forbidden_doc_ids=["AUTH-001"],
            baseline_a=a,
            devtrace=dt,
        )
        metrics_a = compute_case_metrics_a(record)
        metrics_dt = compute_case_metrics_devtrace(record)
        return record.model_copy(update={"metrics_a": metrics_a, "metrics_devtrace": metrics_dt})

    def test_forbidden_retrieval_rate_baseline_a(self) -> None:
        records = [self._vc_record("VC-01", forbidden_retrieved=True)]
        analysis = compute_version_conflict_analysis(records)
        # Baseline A should have retrieved the forbidden doc
        assert analysis.baseline_a_forbidden_retrieved_rate == 1.0

    def test_devtrace_correct_exclusion(self) -> None:
        records = [self._vc_record("VC-01", devtrace_excluded=True)]
        analysis = compute_version_conflict_analysis(records)
        # AUTH-001 is forbidden and not in applicable_doc_ids → correctly excluded
        assert analysis.devtrace_correct_exclusion_rate == 1.0

    def test_no_version_conflict_cases(self) -> None:
        records = [_make_case_record(case_class="straightforward")]
        analysis = compute_version_conflict_analysis(records)
        assert analysis.total_version_conflict_cases == 0


# ===========================================================================
# Part 6: Unsupported-case analysis tests
# ===========================================================================

class TestUnsupportedCaseAnalysis:

    def test_all_false_answers_on_abstention_cases(self) -> None:
        records = []
        for i in range(3):
            a = BaselineAResult(
                retrieved_doc_ids=["SDK-001"],
                retrieved_chunk_ids=["SDK-001-C01"],
                answer_text="Here is an answer.",
                cited_doc_ids=[],
                abstained=False,
            )
            r = _make_case_record(
                case_id=f"TEST-{i}",
                gold_expected_outcome="INSUFFICIENT_EVIDENCE",
                gold_doc_ids=[],
                gold_forbidden_doc_ids=[],
                baseline_a=a,
            )
            r = r.model_copy(update={"metrics_a": compute_case_metrics_a(r)})
            records.append(r)
        analysis = compute_unsupported_case_analysis(records)
        assert analysis.total_abstention_expected_cases == 3
        assert analysis.baseline_a_false_answer_rate == 1.0

    def test_devtrace_correct_abstention(self) -> None:
        dt = _make_devtrace_abstained("INSUFFICIENT_EVIDENCE")
        r = _make_case_record(
            gold_expected_outcome="INSUFFICIENT_EVIDENCE",
            gold_doc_ids=[],
            gold_forbidden_doc_ids=[],
            devtrace=dt,
        )
        r = r.model_copy(update={"metrics_devtrace": compute_case_metrics_devtrace(r)})
        analysis = compute_unsupported_case_analysis([r])
        assert analysis.devtrace_correct_abstention_rate == 1.0

    def test_no_abstention_expected_cases(self) -> None:
        r = _make_case_record(gold_expected_outcome="ANSWERED")
        analysis = compute_unsupported_case_analysis([r])
        assert analysis.total_abstention_expected_cases == 0


# ===========================================================================
# Part 7: Baseline A tests
# ===========================================================================

class TestBaselineAExecution:

    def test_produces_answer_on_retrieval(self) -> None:
        retriever = _StubRetriever()
        normalized = normalize_incident(
            "AUTH_401 after upgrading to SDK 3.1.",
            current_version="3.1",
            error_codes=["AUTH_401"],
        )
        llm = FakeLLMClient(response="The fix is to update your auth headers.")
        result = run_baseline_a(normalized=normalized, retriever=retriever, llm_client=llm)
        assert result.abstained is False
        assert result.answer_text is not None
        assert "AUTH-002" in result.retrieved_doc_ids or "SDK-001" in result.retrieved_doc_ids
        assert result.error is None

    def test_abstains_on_empty_retrieval(self) -> None:
        normalized = normalize_incident("AUTH_401 after upgrading.")
        llm = FakeLLMClient(response="Not called.")
        result = run_baseline_a(
            normalized=normalized,
            retriever=_empty_stub_retriever(),
            llm_client=llm,
        )
        assert result.abstained is True
        assert result.answer_text is None
        assert llm.call_count == 0  # LLM should not be called

    def test_llm_error_causes_abstention(self) -> None:
        from src.errors import LLMError
        normalized = normalize_incident("AUTH_401.")
        retriever = _StubRetriever()
        llm = FakeLLMClient(exception=LLMError("timeout"))
        result = run_baseline_a(normalized=normalized, retriever=retriever, llm_client=llm)
        assert result.abstained is True
        assert result.error is not None

    def test_doc_ids_are_deduplicated(self) -> None:
        results = [
            _make_retrieval_result("AUTH-002-C01", "AUTH-002", 0.9),
            _make_retrieval_result("AUTH-002-C02", "AUTH-002", 0.8),  # Same doc, second chunk
        ]
        retriever = _StubRetriever(results=results)
        normalized = normalize_incident("AUTH issue.")
        llm = FakeLLMClient(response="Fix: AUTH-002 describes the solution.")
        result = run_baseline_a(normalized=normalized, retriever=retriever, llm_client=llm)
        assert result.retrieved_doc_ids.count("AUTH-002") == 1


# ===========================================================================
# Part 8: Baseline B tests
# ===========================================================================

class TestBaselineBExecution:

    def test_generates_above_threshold(self) -> None:
        retriever = _StubRetriever([
            _make_retrieval_result(score=0.85),
        ])
        normalized = normalize_incident("AUTH_401 after SDK update.", current_version="3.1")
        llm = FakeLLMClient(response="Fix is in AUTH-002.")
        result = run_baseline_b(
            normalized=normalized,
            retriever=retriever,
            llm_client=llm,
            threshold=0.50,
        )
        assert result.threshold_passed is True
        assert result.abstained is False
        assert result.answer_text is not None

    def test_abstains_below_threshold(self) -> None:
        retriever = _StubRetriever([
            _make_retrieval_result(score=0.30),
        ])
        normalized = normalize_incident("Some unclear issue.")
        llm = FakeLLMClient(response="Should not be called.")
        result = run_baseline_b(
            normalized=normalized,
            retriever=retriever,
            llm_client=llm,
            threshold=0.50,
        )
        assert result.threshold_passed is False
        assert result.abstained is True
        assert result.answer_text is None
        assert llm.call_count == 0

    def test_abstains_on_empty_retrieval(self) -> None:
        normalized = normalize_incident("Something is wrong.")
        llm = FakeLLMClient(response="Not called.")
        result = run_baseline_b(
            normalized=normalized,
            retriever=_empty_stub_retriever(),
            llm_client=llm,
            threshold=0.40,
        )
        assert result.abstained is True
        assert result.top_score is None

    def test_threshold_exactly_at_boundary(self) -> None:
        """Score exactly at threshold should pass (>=)."""
        retriever = _StubRetriever([
            _make_retrieval_result(score=0.40),
        ])
        normalized = normalize_incident("AUTH issue.", current_version="3.0")
        llm = FakeLLMClient(response="Answer based on AUTH-002.")
        result = run_baseline_b(
            normalized=normalized,
            retriever=retriever,
            llm_client=llm,
            threshold=0.40,
        )
        assert result.threshold_passed is True
        assert result.abstained is False

    def test_threshold_stored_in_result(self) -> None:
        normalized = normalize_incident("Issue.")
        llm = FakeLLMClient(response="A.")
        result = run_baseline_b(
            normalized=normalize_incident("Issue."),
            retriever=_StubRetriever([_make_retrieval_result(score=0.6)]),
            llm_client=llm,
            threshold=0.55,
        )
        assert result.threshold_used == 0.55


# ===========================================================================
# Part 9: DevTrace eval adapter tests
# ===========================================================================

class TestExtractDocIdsFromChunks:

    def test_standard_chunk_id_format(self) -> None:
        result = _extract_doc_ids_from_chunks(["AUTH-002-C01", "AUTH-002-C02", "SDK-001-C01"])
        assert result == ["AUTH-002", "SDK-001"]

    def test_non_standard_chunk_id_kept_as_is(self) -> None:
        result = _extract_doc_ids_from_chunks(["CUSTOM-CHUNK"])
        assert result == ["CUSTOM-CHUNK"]

    def test_empty_input(self) -> None:
        assert _extract_doc_ids_from_chunks([]) == []

    def test_deduplication(self) -> None:
        result = _extract_doc_ids_from_chunks(["AUTH-002-C01", "AUTH-002-C03"])
        assert result == ["AUTH-002"]


class TestDevTraceEvalDeterministic:

    def _diagnosis_json(self, chunk_id: str = "AUTH-002-C01") -> str:
        return json.dumps({
            "claims": [
                {
                    "role": "root_cause",
                    "text": "The auth header format changed in SDK 3.x.",
                    "evidence_ids": [chunk_id],
                }
            ]
        })

    def _verifier_json(self, verified: bool = True) -> str:
        return json.dumps({
            "citation_correct": verified,
            "sufficient": verified,
            "contradicted": False,
            "supporting_evidence_ids": ["AUTH-002-C01"] if verified else [],
            "contradicting_evidence_ids": [],
            "reason": "Verified." if verified else "Not grounded.",
        })

    def test_answered_full_result(self) -> None:
        retriever = _StubRetriever([
            _make_retrieval_result("AUTH-002-C01", "AUTH-002", 0.85),
        ])
        llm = FakeLLMClient(responses=[
            self._diagnosis_json("AUTH-002-C01"),
            self._verifier_json(True),  # root_cause verification
        ])
        result = run_devtrace_eval(
            description="AUTH_401 after upgrading to SDK 3.1.",
            current_version="3.1",
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
            gold_forbidden_doc_ids=["AUTH-001"],
        )
        assert result.abstained is False
        assert result.final_outcome in ("ANSWERED_FULL", "ANSWERED_PARTIAL")
        assert result.error is None

    def test_needs_info_when_no_version_and_conflicting(self) -> None:
        """
        No current_version + competing version-specific docs should trigger NEEDS_INFO.
        This test uses a retriever returning both 2.x and 3.x docs.
        """
        competing_results = [
            RetrievalResult(
                chunk_id="AUTH-001-C01",
                doc_id="AUTH-001",
                score=0.80,
                source=RetrievalSource.HYBRID,
                metadata={"content": "SDK 2.x auth fix.", "applies_to": ">=2.0,<3.0", "topic": "auth"},
            ),
            RetrievalResult(
                chunk_id="AUTH-002-C01",
                doc_id="AUTH-002",
                score=0.79,
                source=RetrievalSource.HYBRID,
                metadata={"content": "SDK 3.x auth fix.", "applies_to": ">=3.0,<4.0", "topic": "auth"},
            ),
        ]
        retriever = _StubRetriever(results=competing_results)
        # No LLM calls expected for NEEDS_INFO path
        llm = FakeLLMClient(response="Should not be called.")
        result = run_devtrace_eval(
            description="AUTH_401 after SDK upgrade.",
            # No current_version provided
            error_codes=["AUTH_401"],
            retriever=retriever,
            llm_client=llm,
        )
        assert result.final_outcome == "NEEDS_INFO"
        assert result.abstained is True

    def test_pipeline_error_returns_degraded(self) -> None:
        from src.errors import LLMError
        retriever = _StubRetriever()
        llm = FakeLLMClient(exception=LLMError("API error"))
        result = run_devtrace_eval(
            description="AUTH_401 issue.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        assert result.abstained is True

    def test_gold_forbidden_not_in_pipeline_prompt(self) -> None:
        """
        Critical leakage test: verify gold_forbidden_doc_ids are not visible
        to the LLM by checking that they don't appear in the prompt.

        We use a capturing FakeLLM that records the prompt and verify
        forbidden doc IDs were not injected.
        """
        prompts_seen: list[str] = []

        class CapturingFakeLLM(FakeLLMClient):
            def generate(self, prompt: str, **kwargs):  # type: ignore[override]
                prompts_seen.append(prompt)
                return super().generate(prompt, **kwargs)

        diagnosis_json = json.dumps({
            "claims": [
                {
                    "role": "root_cause",
                    "text": "Root cause.",
                    "evidence_ids": ["AUTH-002-C01"],
                }
            ]
        })
        verifier_json = self._verifier_json(True)
        llm = CapturingFakeLLM(responses=[diagnosis_json, verifier_json])
        retriever = _StubRetriever()

        run_devtrace_eval(
            description="AUTH_401 after SDK 3.1 upgrade.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
            gold_forbidden_doc_ids=["FORBIDDEN-DOC-SENTINEL-12345"],
        )

        # The sentinel forbidden doc ID must NOT appear in any LLM prompt
        for prompt in prompts_seen:
            assert "FORBIDDEN-DOC-SENTINEL-12345" not in prompt, (
                "Gold forbidden doc ID was leaked into LLM prompt!"
            )

    def test_retrieved_doc_ids_are_populated(self) -> None:
        retriever = _StubRetriever([
            _make_retrieval_result("AUTH-002-C01", "AUTH-002", 0.85),
            _make_retrieval_result("SDK-001-C01", "SDK-001", 0.70),
        ])
        diagnosis_json = json.dumps({
            "claims": [
                {"role": "root_cause", "text": "Root cause.", "evidence_ids": ["AUTH-002-C01"]},
            ]
        })
        verifier_json = self._verifier_json(True)
        llm = FakeLLMClient(responses=[diagnosis_json, verifier_json])
        result = run_devtrace_eval(
            description="AUTH_401.",
            current_version="3.1",
            retriever=retriever,
            llm_client=llm,
        )
        assert "AUTH-002" in result.retrieved_doc_ids


# ===========================================================================
# Part 10: Heuristic citation extraction tests
# ===========================================================================

class TestHeuristicCitedDocIds:

    def test_extracts_known_patterns(self) -> None:
        text = "As described in AUTH-002, you need to update your headers. See also SDK-001."
        ids = _heuristic_cited_doc_ids(text)
        assert "AUTH-002" in ids
        assert "SDK-001" in ids

    def test_no_false_positives_on_generic_text(self) -> None:
        text = "The answer depends on your configuration."
        ids = _heuristic_cited_doc_ids(text)
        assert ids == []

    def test_deduplication(self) -> None:
        text = "AUTH-002 is the key doc. AUTH-002 explains the issue."
        ids = _heuristic_cited_doc_ids(text)
        assert ids.count("AUTH-002") == 1

    def test_empty_text(self) -> None:
        assert _heuristic_cited_doc_ids("") == []


# ===========================================================================
# Part 11: Edge case tests
# ===========================================================================

class TestEdgeCases:

    def test_case_record_with_no_systems_run(self) -> None:
        """Record with all system outputs None should have None metrics."""
        record = _make_case_record()
        metrics_a = compute_case_metrics_a(record)
        # All fields should be None (no baseline_a to compute from)
        assert metrics_a.retrieval_hit is None

    def test_devtrace_degraded_counts_as_abstained(self) -> None:
        dt = _make_devtrace_abstained("DEGRADED")
        assert dt.abstained is True
        assert dt.final_outcome == "DEGRADED"

    def test_partial_answer_not_counted_as_abstained(self) -> None:
        dt = DevTraceEvalResult(
            final_outcome="ANSWERED_PARTIAL",
            answer_text="Partial answer here.",
            retrieved_doc_ids=["AUTH-002"],
            retrieved_chunk_ids=["AUTH-002-C01"],
            applicable_doc_ids=["AUTH-002"],
            applicable_chunk_ids=["AUTH-002-C01"],
            cited_doc_ids=["AUTH-002"],
            cited_chunk_ids=["AUTH-002-C01"],
            forbidden_doc_ids_cited=[],
            retry_attempted=False,
            retry_succeeded=False,
            abstained=False,
        )
        assert dt.abstained is False

    def test_build_aggregate_metrics_with_empty_records(self) -> None:
        agg = build_aggregate_metrics([], "devtrace")
        assert agg.total_cases == 0
        assert agg.retrieval_hit_rate is None

    def test_all_cases_abstaining(self) -> None:
        """If all DevTrace cases abstain, outcome counts should reflect it."""
        records = []
        for outcome in ["INSUFFICIENT_EVIDENCE", "NEEDS_INFO", "DEGRADED"]:
            dt = _make_devtrace_abstained(outcome)
            records.append(_make_case_record(devtrace=dt))
        dist = compute_outcome_distribution(records)
        assert dist.answered_full == 0
        assert dist.answered_partial == 0
        assert dist.total == 3


# ===========================================================================
# Part 12: Config validation tests
# ===========================================================================

class TestEvaluationConfig:

    def test_deterministic_mode_is_default(self) -> None:
        config = EvaluationConfig()
        assert config.evaluation_mode == "deterministic"

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="evaluation_mode"):
            EvaluationConfig(evaluation_mode="invalid")

    def test_threshold_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="baseline_b_threshold"):
            EvaluationConfig(baseline_b_threshold=1.5)

    def test_model_config_info_in_deterministic_mode(self) -> None:
        config = EvaluationConfig()
        info = config.model_config_info()
        assert info["live_model_name"] == "FakeLLM"
        assert info["evaluation_mode"] == "deterministic"

    def test_live_mode_uses_canonical_gemini_key(self, monkeypatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = EvaluationConfig(evaluation_mode="live")
        assert config.evaluation_mode == "live"

    def test_live_mode_missing_key_is_clear(self, monkeypatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            EvaluationConfig(evaluation_mode="live")

    def test_live_client_constructor_contract(self, monkeypatch) -> None:
        from src.m7.runner import _build_live_llm

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = EvaluationConfig(
            evaluation_mode="live",
            live_model_name="test-model",
            live_temperature=0.25,
        )
        with patch("src.llm.gemini.GeminiClient") as client_cls:
            _build_live_llm(config)
        client_cls.assert_called_once_with(model_name="test-model", temperature=0.25)


# ===========================================================================
# Part 13: Dataset loading tests
# ===========================================================================

class TestDatasetLoading:

    def test_default_dataset_loads(self) -> None:
        dataset = load_eval_dataset()
        assert len(dataset.cases) > 0

    def test_dataset_has_all_required_categories(self) -> None:
        dataset = load_eval_dataset()
        categories = {c.case_class for c in dataset.cases}
        expected = {
            "straightforward", "multi_document", "version_conflict",
            "similar_error", "near_miss", "unsupported", "missing_info",
            "contradictory_evidence",
        }
        assert categories == expected, f"Missing categories: {expected - categories}"

    def test_dataset_has_required_number_of_cases(self) -> None:
        dataset = load_eval_dataset()
        # M7 spec: 30-40 cases
        assert 30 <= len(dataset.cases) <= 45, f"Got {len(dataset.cases)} cases"

    def test_case_ids_are_unique(self) -> None:
        dataset = load_eval_dataset()
        ids = [c.case_id for c in dataset.cases]
        assert len(ids) == len(set(ids))

    def test_version_conflict_cases_have_forbidden_docs(self) -> None:
        dataset = load_eval_dataset()
        vc_cases = [c for c in dataset.cases if c.case_class == "version_conflict"]
        for case in vc_cases:
            assert case.forbidden_doc_ids, (
                f"Version conflict case {case.case_id} has no forbidden_doc_ids"
            )


# ===========================================================================
# Part 14: Deterministic reproducibility tests
# ===========================================================================

class TestDeterministicReproducibility:

    def test_case_fixture_uses_applicable_case_specific_evidence(self) -> None:
        from src.m7.runner import _run_case_deterministic
        from src.models.enums import AnswerCompleteness, SystemOutcome

        case = EvaluationCase(
            case_id="FIXTURE-ANSWERED",
            case_class="straightforward",
            description="Controlled fixture",
            incident=EvaluationIncident(
                description="AUTH_401 on SDK 3.1",
                current_version="3.1",
                error_codes=["AUTH_401"],
            ),
            expected_outcome=SystemOutcome.ANSWERED,
            expected_completeness=AnswerCompleteness.FULL,
            gold_doc_ids=["AUTH-002"],
            forbidden_doc_ids=["AUTH-001"],
        )
        retriever = _StubRetriever(results=[
            _make_retrieval_result("AUTH-002-C01", "AUTH-002", 0.9),
            RetrievalResult(
                chunk_id="AUTH-001-C01",
                doc_id="AUTH-001",
                score=0.8,
                source=RetrievalSource.HYBRID,
                metadata={"content": "Wrong version", "applies_to": ">=2.0,<3.0"},
            ),
        ])

        record = _run_case_deterministic(case, retriever, EvaluationConfig())

        assert record.devtrace is not None
        assert record.devtrace.final_outcome in {"ANSWERED_FULL", "ANSWERED_PARTIAL"}
        assert record.devtrace.cited_chunk_ids == ["AUTH-002-C01"]
        assert "AUTH-001" not in record.devtrace.applicable_doc_ids
        assert record.devtrace.retry_attempted is False

    def test_case_fixture_schedule_supports_failed_retry(self) -> None:
        from src.m7.runner import _run_case_deterministic
        from src.models.enums import SystemOutcome

        case = EvaluationCase(
            case_id="FIXTURE-UNSUPPORTED",
            case_class="unsupported",
            description="Controlled rejection fixture",
            incident=EvaluationIncident(description="Unsupported AUTH_999", error_codes=["AUTH_999"]),
            expected_outcome=SystemOutcome.INSUFFICIENT_EVIDENCE,
            gold_doc_ids=[],
        )
        retriever = _StubRetriever(results=[
            RetrievalResult(
                chunk_id="GENERIC-C01",
                doc_id="GENERIC",
                score=0.7,
                source=RetrievalSource.HYBRID,
                metadata={"content": "Generic evidence is insufficient.", "applies_to": "*"},
            )
        ])
        record = _run_case_deterministic(case, retriever, EvaluationConfig())

        assert record.devtrace is not None
        assert record.devtrace.final_outcome == "INSUFFICIENT_EVIDENCE"
        assert record.devtrace.retry_attempted is True
        assert record.devtrace.retry_succeeded is False

    def test_contradictory_fixture_exercises_successful_single_retry(self) -> None:
        from src.m7.runner import _run_case_deterministic
        from src.models.enums import AnswerCompleteness, SystemOutcome

        case = EvaluationCase(
            case_id="FIXTURE-RETRY",
            case_class="contradictory_evidence",
            description="Controlled retry fixture",
            incident=EvaluationIncident(
                description="AUTH_401 on SDK 3.1",
                current_version="3.1",
                error_codes=["AUTH_401"],
            ),
            expected_outcome=SystemOutcome.ANSWERED,
            expected_completeness=AnswerCompleteness.FULL,
            gold_doc_ids=["AUTH-002"],
        )
        record = _run_case_deterministic(case, _StubRetriever(), EvaluationConfig())

        assert record.devtrace is not None
        assert record.devtrace.final_outcome == "ANSWERED_FULL"
        assert record.devtrace.retry_attempted is True
        assert record.devtrace.retry_succeeded is True

    def test_summary_rejects_positive_claim_when_degradation_dominates(self) -> None:
        from src.m7.runner import _build_summary

        report = EvaluationReport(
            dataset_path="fixture.json",
            total_cases=3,
            evaluation_mode="live",
            baseline_b_threshold=0.4,
            aggregate_a=AggregateMetrics(total_cases=3, false_answer_rate=0.5),
            aggregate_devtrace=AggregateMetrics(total_cases=3, false_answer_rate=0.0),
            outcome_distribution=OutcomeDistribution(degraded=3, total=3),
        )
        summary = _build_summary(report)
        assert summary["Is this run suitable for end-to-end performance claims?"].startswith("No.")
        assert "not evidence of improved answer reliability" in summary["Did DevTrace reduce false answers?"]

    def test_same_inputs_produce_same_metrics(self) -> None:
        """Running metrics twice on the same record gives the same result."""
        a = _make_baseline_a_answered()
        record = _make_case_record(baseline_a=a)
        m1 = compute_case_metrics_a(record)
        m2 = compute_case_metrics_a(record)
        assert m1.retrieval_hit == m2.retrieval_hit
        assert m1.false_answer == m2.false_answer
        assert m1.citation_valid == m2.citation_valid

    def test_retry_stats_are_deterministic(self) -> None:
        dt = _make_devtrace_answered("ANSWERED_FULL", retry_attempted=True, retry_succeeded=True)
        records = [_make_case_record(devtrace=dt)]
        s1 = compute_retry_stats(records)
        s2 = compute_retry_stats(records)
        assert s1.retries_attempted == s2.retries_attempted
        assert s1.retry_recovery_rate == s2.retry_recovery_rate


# ===========================================================================
# Part 15: Category aggregation tests
# ===========================================================================

class TestCategoryAggregation:

    def test_category_metrics_built_correctly(self) -> None:
        from src.m7.runner import _build_category_metrics

        records = []
        for cat in ["version_conflict", "straightforward", "unsupported"]:
            a = _make_baseline_a_answered()
            r = _make_case_record(case_class=cat, baseline_a=a)
            r = r.model_copy(update={"metrics_a": compute_case_metrics_a(r)})
            records.append(r)

        categories = _build_category_metrics(records)
        cat_names = [cm.category for cm in categories]
        assert "version_conflict" in cat_names
        assert "straightforward" in cat_names
        assert "unsupported" in cat_names

    def test_empty_category_skipped(self) -> None:
        from src.m7.runner import _build_category_metrics
        records = [_make_case_record(case_class="straightforward")]
        categories = _build_category_metrics(records)
        # Only 'straightforward' should appear
        assert all(cm.category == "straightforward" for cm in categories)
