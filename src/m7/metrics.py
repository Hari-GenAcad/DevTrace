"""
DevTrace — Module 7: Core metric calculations.

All metric functions are pure functions that operate on EvaluationCaseRecord
collections. They never touch the production pipeline, LLM clients, or
evaluation dataset directly.

Metric definitions:

    retrieval_hit_rate:
        Fraction of cases where ≥1 gold_doc_id was present in
        retrieved_doc_ids.  Only meaningful for cases with gold_doc_ids.

    citation_validity:
        Fraction of cases where every cited_doc_id was in the set of
        retrieved_doc_ids (for baselines) or applicable_doc_ids (for DevTrace).
        N/A when no citations were produced.

    citation_correctness:
        Fraction of answered cases where ≥1 gold_doc_id was cited.
        Note: This is an offline benchmark proxy (gold-doc overlap).
        Runtime semantic verification (checking if evidence text semantically
        entails claim statement) is performed independently by M5 ClaimVerifier.

    false_answer_rate:
        Fraction of cases where the system gave a confident non-empty answer
        but ground truth says the system SHOULD abstain (NEEDS_INFO or
        INSUFFICIENT_EVIDENCE), OR where the system cited a forbidden document.

    false_abstention_rate:
        Fraction of expected-ANSWERED cases where the system abstained
        (produced no answer at all).

    outcome_distribution:
        For DevTrace: count of each FinalOutcome value.

    retry_stats:
        For DevTrace: retry attempt count, recovery count, and rate.
"""

from __future__ import annotations

from src.m7.models import (
    AggregateMetrics,
    CaseMetrics,
    EvaluationCaseRecord,
    OutcomeDistribution,
    RetryStats,
    UnsupportedCaseAnalysis,
    VersionConflictAnalysis,
)


# ---------------------------------------------------------------------------
# Helper: cases that are applicable to a given metric
# ---------------------------------------------------------------------------

def _cases_with_gold_docs(
    records: list[EvaluationCaseRecord],
) -> list[EvaluationCaseRecord]:
    """Cases that have at least one gold_doc_id (retrieval hit is meaningful)."""
    return [r for r in records if r.gold_doc_ids]


def _cases_expecting_answer(
    records: list[EvaluationCaseRecord],
) -> list[EvaluationCaseRecord]:
    """Cases where gold says the system should produce an answer."""
    return [r for r in records if r.gold_expected_outcome == "ANSWERED"]


def _cases_expecting_abstention(
    records: list[EvaluationCaseRecord],
) -> list[EvaluationCaseRecord]:
    """Cases where gold says the system should NOT produce an answer."""
    return [
        r
        for r in records
        if r.gold_expected_outcome in ("INSUFFICIENT_EVIDENCE", "NEEDS_INFO")
    ]


def _safe_rate(numerator: int, denominator: int) -> float | None:
    """Return numerator/denominator or None when denominator is zero."""
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


# ---------------------------------------------------------------------------
# Per-case metric calculators
# (Called during evaluation run to populate CaseMetrics)
# ---------------------------------------------------------------------------

def score_retrieval_hit(
    retrieved_doc_ids: list[str],
    gold_doc_ids: list[str],
) -> bool | None:
    """
    Return True if ≥1 gold doc was retrieved.
    Returns None when there are no gold docs (metric not applicable).
    """
    if not gold_doc_ids:
        return None
    return any(g in retrieved_doc_ids for g in gold_doc_ids)


def score_forbidden_retrieved(
    retrieved_doc_ids: list[str],
    forbidden_doc_ids: list[str],
) -> bool | None:
    """
    Return True if ≥1 forbidden doc was retrieved.
    Returns None when there are no forbidden docs.
    """
    if not forbidden_doc_ids:
        return None
    return any(f in retrieved_doc_ids for f in forbidden_doc_ids)


def score_citation_valid(
    cited_doc_ids: list[str],
    valid_pool_doc_ids: list[str],
) -> bool | None:
    """
    Return True if every cited doc ID is within valid_pool_doc_ids.

    For baselines: valid_pool = retrieved_doc_ids.
    For DevTrace: valid_pool = applicable_doc_ids.
    Returns None when no citations were produced.
    """
    if not cited_doc_ids:
        return None
    pool = set(valid_pool_doc_ids)
    return all(c in pool for c in cited_doc_ids)


def score_citation_correct(
    cited_doc_ids: list[str],
    gold_doc_ids: list[str],
) -> bool | None:
    """
    Return True if ≥1 gold doc was cited.
    Returns None when: no citations, or no gold docs.

    Note: Gold-document overlap is a deterministic offline benchmark proxy for
    citation correctness. Real runtime semantic verification (whether cited
    evidence actually supports the generated claim) is performed in M5.
    """
    if not cited_doc_ids or not gold_doc_ids:
        return None
    return any(g in cited_doc_ids for g in gold_doc_ids)


def score_false_answer(
    *,
    abstained: bool,
    answer_text: str | None,
    gold_expected_outcome: str,
    cited_doc_ids: list[str],
    forbidden_doc_ids: list[str],
) -> bool | None:
    """
    Return True if the system gave a confident wrong answer.

    A "false answer" is defined as:
      1. System produced a non-empty answer when gold says INSUFFICIENT_EVIDENCE
         or NEEDS_INFO, OR
      2. System cited a forbidden (wrong-version) document in its answer.

    Returns None for cases where gold expects ANSWERED (false answer metric
    is only meaningful for abstention-expected cases from the first condition,
    but forbidden citation is universal).

    Simplified: we compute this for all cases.
    """
    produced_answer = not abstained and bool(answer_text and answer_text.strip())

    # Condition 1: answered when should have abstained
    should_abstain = gold_expected_outcome in ("INSUFFICIENT_EVIDENCE", "NEEDS_INFO")
    condition_1 = produced_answer and should_abstain

    # Condition 2: cited a forbidden document
    condition_2 = bool(cited_doc_ids) and any(
        f in cited_doc_ids for f in forbidden_doc_ids
    )

    return condition_1 or condition_2


def score_false_abstention(
    *,
    abstained: bool,
    gold_expected_outcome: str,
) -> bool | None:
    """
    Return True if system abstained when gold expects ANSWERED.
    Returns None when gold does not expect ANSWERED.
    """
    if gold_expected_outcome != "ANSWERED":
        return None
    return abstained


# ---------------------------------------------------------------------------
# Per-case metric computation entry points (called by runner)
# ---------------------------------------------------------------------------

def compute_case_metrics_a(record: EvaluationCaseRecord) -> CaseMetrics:
    """Compute metrics for Baseline A on the given case record."""
    result = record.baseline_a
    if result is None:
        return CaseMetrics()

    retrieval_hit = score_retrieval_hit(result.retrieved_doc_ids, record.gold_doc_ids)
    forbidden_retrieved = score_forbidden_retrieved(
        result.retrieved_doc_ids, record.gold_forbidden_doc_ids
    )
    # Baseline A validity: cited within retrieved set
    citation_valid = score_citation_valid(
        result.cited_doc_ids, result.retrieved_doc_ids
    )
    citation_correct = score_citation_correct(
        result.cited_doc_ids, record.gold_doc_ids
    )
    false_answer = score_false_answer(
        abstained=result.abstained,
        answer_text=result.answer_text,
        gold_expected_outcome=record.gold_expected_outcome,
        cited_doc_ids=result.cited_doc_ids,
        forbidden_doc_ids=record.gold_forbidden_doc_ids,
    )
    false_abstention = score_false_abstention(
        abstained=result.abstained,
        gold_expected_outcome=record.gold_expected_outcome,
    )
    return CaseMetrics(
        retrieval_hit=retrieval_hit,
        forbidden_retrieved=forbidden_retrieved,
        citation_valid=citation_valid,
        citation_correct=citation_correct,
        false_answer=false_answer,
        false_abstention=false_abstention,
    )


def compute_case_metrics_b(record: EvaluationCaseRecord) -> CaseMetrics:
    """Compute metrics for Baseline B on the given case record."""
    result = record.baseline_b
    if result is None:
        return CaseMetrics()

    retrieval_hit = score_retrieval_hit(result.retrieved_doc_ids, record.gold_doc_ids)
    forbidden_retrieved = score_forbidden_retrieved(
        result.retrieved_doc_ids, record.gold_forbidden_doc_ids
    )
    citation_valid = score_citation_valid(
        result.cited_doc_ids, result.retrieved_doc_ids
    )
    citation_correct = score_citation_correct(
        result.cited_doc_ids, record.gold_doc_ids
    )
    false_answer = score_false_answer(
        abstained=result.abstained,
        answer_text=result.answer_text,
        gold_expected_outcome=record.gold_expected_outcome,
        cited_doc_ids=result.cited_doc_ids,
        forbidden_doc_ids=record.gold_forbidden_doc_ids,
    )
    false_abstention = score_false_abstention(
        abstained=result.abstained,
        gold_expected_outcome=record.gold_expected_outcome,
    )
    return CaseMetrics(
        retrieval_hit=retrieval_hit,
        forbidden_retrieved=forbidden_retrieved,
        citation_valid=citation_valid,
        citation_correct=citation_correct,
        false_answer=false_answer,
        false_abstention=false_abstention,
    )


def compute_case_metrics_devtrace(record: EvaluationCaseRecord) -> CaseMetrics:
    """Compute metrics for DevTrace on the given case record."""
    result = record.devtrace
    if result is None:
        return CaseMetrics()

    retrieval_hit = score_retrieval_hit(result.retrieved_doc_ids, record.gold_doc_ids)
    # For DevTrace: forbidden = any forbidden doc in retrieved set
    forbidden_retrieved = score_forbidden_retrieved(
        result.retrieved_doc_ids, record.gold_forbidden_doc_ids
    )
    # DevTrace citation validity: cited within applicable set
    citation_valid = score_citation_valid(
        result.cited_doc_ids, result.applicable_doc_ids
    )
    citation_correct = score_citation_correct(
        result.cited_doc_ids, record.gold_doc_ids
    )
    false_answer = score_false_answer(
        abstained=result.abstained,
        answer_text=result.answer_text,
        gold_expected_outcome=record.gold_expected_outcome,
        cited_doc_ids=result.cited_doc_ids,
        forbidden_doc_ids=record.gold_forbidden_doc_ids,
    )
    false_abstention = score_false_abstention(
        abstained=result.abstained,
        gold_expected_outcome=record.gold_expected_outcome,
    )
    return CaseMetrics(
        retrieval_hit=retrieval_hit,
        forbidden_retrieved=forbidden_retrieved,
        citation_valid=citation_valid,
        citation_correct=citation_correct,
        false_answer=false_answer,
        false_abstention=false_abstention,
    )


# ---------------------------------------------------------------------------
# Aggregate metric calculators (over full case list)
# ---------------------------------------------------------------------------

def compute_retrieval_hit_rate(
    records: list[EvaluationCaseRecord],
    system: str,  # "a", "b", or "devtrace"
) -> float | None:
    """
    Fraction of cases with gold_doc_ids where ≥1 gold doc was retrieved.
    """
    applicable = _cases_with_gold_docs(records)
    if not applicable:
        return None
    hits = 0
    for r in applicable:
        metrics = _get_metrics(r, system)
        if metrics and metrics.retrieval_hit is True:
            hits += 1
    return _safe_rate(hits, len(applicable))


def compute_citation_validity(
    records: list[EvaluationCaseRecord],
    system: str,
) -> float | None:
    """
    Fraction of cases where all citations were within the valid pool.
    Only counted for cases where citations were produced.
    """
    cited_cases = []
    valid_count = 0
    for r in records:
        metrics = _get_metrics(r, system)
        if metrics and metrics.citation_valid is not None:
            cited_cases.append(r)
            if metrics.citation_valid:
                valid_count += 1
    return _safe_rate(valid_count, len(cited_cases))


def compute_citation_correctness(
    records: list[EvaluationCaseRecord],
    system: str,
) -> float | None:
    """
    Fraction of cases where ≥1 gold doc was cited.
    Only counted for cases where citations were produced AND gold docs exist.
    """
    applicable = []
    correct = 0
    for r in records:
        metrics = _get_metrics(r, system)
        if metrics and metrics.citation_correct is not None:
            applicable.append(r)
            if metrics.citation_correct:
                correct += 1
    return _safe_rate(correct, len(applicable))


def compute_false_answer_rate(
    records: list[EvaluationCaseRecord],
    system: str,
) -> float | None:
    """
    Fraction of all cases where the system produced a false confident answer.
    """
    if not records:
        return None
    false_answers = 0
    for r in records:
        metrics = _get_metrics(r, system)
        if metrics and metrics.false_answer is True:
            false_answers += 1
    return _safe_rate(false_answers, len(records))


def compute_false_abstention_rate(
    records: list[EvaluationCaseRecord],
    system: str,
) -> float | None:
    """
    Fraction of expected-ANSWERED cases where the system abstained.
    """
    applicable = _cases_expecting_answer(records)
    if not applicable:
        return None
    false_abstentions = 0
    for r in applicable:
        metrics = _get_metrics(r, system)
        if metrics and metrics.false_abstention is True:
            false_abstentions += 1
    return _safe_rate(false_abstentions, len(applicable))


def _get_metrics(
    record: EvaluationCaseRecord,
    system: str,
) -> CaseMetrics | None:
    """Return the appropriate CaseMetrics for the given system label."""
    if system == "a":
        return record.metrics_a
    if system == "b":
        return record.metrics_b
    if system == "devtrace":
        return record.metrics_devtrace
    raise ValueError(f"Unknown system: {system!r}. Use 'a', 'b', or 'devtrace'.")


# ---------------------------------------------------------------------------
# Aggregate metrics builder
# ---------------------------------------------------------------------------

def build_aggregate_metrics(
    records: list[EvaluationCaseRecord],
    system: str,
) -> AggregateMetrics:
    """Build an AggregateMetrics record for the given system over all records."""
    total = len(records)

    # Forbidden retrieval rate
    forbidden_cases = [r for r in records if r.gold_forbidden_doc_ids]
    forbidden_count = sum(
        1
        for r in forbidden_cases
        if _get_metrics(r, system) and _get_metrics(r, system).forbidden_retrieved  # type: ignore[union-attr]
    )

    return AggregateMetrics(
        total_cases=total,
        retrieval_hit_rate=compute_retrieval_hit_rate(records, system),
        forbidden_retrieval_rate=_safe_rate(forbidden_count, len(forbidden_cases)),
        citation_validity_rate=compute_citation_validity(records, system),
        citation_correctness_rate=compute_citation_correctness(records, system),
        false_answer_rate=compute_false_answer_rate(records, system),
        false_abstention_rate=compute_false_abstention_rate(records, system),
    )


# ---------------------------------------------------------------------------
# Outcome distribution (DevTrace only)
# ---------------------------------------------------------------------------

def compute_outcome_distribution(
    records: list[EvaluationCaseRecord],
) -> OutcomeDistribution:
    """Compute the distribution of M6 FinalOutcome values."""
    counts: dict[str, int] = {
        "ANSWERED_FULL": 0,
        "ANSWERED_PARTIAL": 0,
        "INSUFFICIENT_EVIDENCE": 0,
        "NEEDS_INFO": 0,
        "DEGRADED": 0,
    }
    for r in records:
        if r.devtrace is not None:
            outcome = r.devtrace.final_outcome
            if outcome in counts:
                counts[outcome] += 1
    return OutcomeDistribution(
        answered_full=counts["ANSWERED_FULL"],
        answered_partial=counts["ANSWERED_PARTIAL"],
        insufficient_evidence=counts["INSUFFICIENT_EVIDENCE"],
        needs_info=counts["NEEDS_INFO"],
        degraded=counts["DEGRADED"],
        total=len(records),
    )


# ---------------------------------------------------------------------------
# Retry statistics (DevTrace only)
# ---------------------------------------------------------------------------

def compute_retry_stats(
    records: list[EvaluationCaseRecord],
) -> RetryStats:
    """Compute M6 retry behaviour statistics."""
    initial_failures = 0
    retries_attempted = 0
    retries_succeeded = 0

    for r in records:
        dt = r.devtrace
        if dt is None:
            continue
        # Count initial root cause failures
        # A retry is only triggered when root cause fails → so retries imply failure
        if dt.retry_attempted:
            initial_failures += 1
            retries_attempted += 1
            if dt.retry_succeeded:
                retries_succeeded += 1
        elif dt.final_outcome == "INSUFFICIENT_EVIDENCE":
            # Root cause failed but retry was not allowed (no applicable evidence)
            initial_failures += 1

    retries_failed = retries_attempted - retries_succeeded
    recovery_rate = _safe_rate(retries_succeeded, retries_attempted)

    return RetryStats(
        initial_root_cause_failures=initial_failures,
        retries_attempted=retries_attempted,
        retries_succeeded=retries_succeeded,
        retries_failed=retries_failed,
        retry_recovery_rate=recovery_rate,
        max_retry_count_exceeded=False,  # Invariant: M6 enforces ≤1 retry
    )


# ---------------------------------------------------------------------------
# Version-conflict analysis
# ---------------------------------------------------------------------------

def compute_version_conflict_analysis(
    records: list[EvaluationCaseRecord],
) -> VersionConflictAnalysis:
    """Dedicated analysis for version_conflict category cases."""
    vc_records = [r for r in records if r.case_class == "version_conflict"]
    total = len(vc_records)

    if total == 0:
        return VersionConflictAnalysis(total_version_conflict_cases=0)

    def _forbidden_rate(system: str) -> float | None:
        forbidden_cases = [r for r in vc_records if r.gold_forbidden_doc_ids]
        if not forbidden_cases:
            return None
        count = sum(
            1
            for r in forbidden_cases
            if _get_metrics(r, system) and _get_metrics(r, system).forbidden_retrieved  # type: ignore[union-attr]
        )
        return _safe_rate(count, len(forbidden_cases))

    def _false_answer_rate(system: str) -> float | None:
        count = sum(
            1
            for r in vc_records
            if _get_metrics(r, system) and _get_metrics(r, system).false_answer  # type: ignore[union-attr]
        )
        return _safe_rate(count, total)

    # DevTrace correct exclusion: forbidden docs NOT in applicable_doc_ids
    correct_exclusion_count = 0
    cases_with_forbidden = [r for r in vc_records if r.gold_forbidden_doc_ids]
    for r in cases_with_forbidden:
        if r.devtrace is not None:
            applicable_docs = set(r.devtrace.applicable_doc_ids)
            excluded = all(f not in applicable_docs for f in r.gold_forbidden_doc_ids)
            if excluded:
                correct_exclusion_count += 1
    correct_exclusion_rate = _safe_rate(correct_exclusion_count, len(cases_with_forbidden))

    return VersionConflictAnalysis(
        total_version_conflict_cases=total,
        baseline_a_forbidden_retrieved_rate=_forbidden_rate("a"),
        baseline_b_forbidden_retrieved_rate=_forbidden_rate("b"),
        devtrace_forbidden_retrieved_rate=_forbidden_rate("devtrace"),
        baseline_a_false_answer_rate=_false_answer_rate("a"),
        baseline_b_false_answer_rate=_false_answer_rate("b"),
        devtrace_false_answer_rate=_false_answer_rate("devtrace"),
        devtrace_correct_exclusion_rate=correct_exclusion_rate,
    )


# ---------------------------------------------------------------------------
# Unsupported-case analysis
# ---------------------------------------------------------------------------

def compute_unsupported_case_analysis(
    records: list[EvaluationCaseRecord],
) -> UnsupportedCaseAnalysis:
    """Analysis for cases where gold expects abstention."""
    abstention_expected = _cases_expecting_abstention(records)
    total = len(abstention_expected)

    if total == 0:
        return UnsupportedCaseAnalysis(total_abstention_expected_cases=0)

    def _false_answer_rate(system: str) -> float | None:
        count = sum(
            1
            for r in abstention_expected
            if _get_metrics(r, system) and _get_metrics(r, system).false_answer  # type: ignore[union-attr]
        )
        return _safe_rate(count, total)

    # DevTrace correct abstention
    correct_abstentions = 0
    for r in abstention_expected:
        if r.devtrace is not None and r.devtrace.abstained:
            correct_abstentions += 1
    correct_abstention_rate = _safe_rate(correct_abstentions, total)

    return UnsupportedCaseAnalysis(
        total_abstention_expected_cases=total,
        baseline_a_false_answer_rate=_false_answer_rate("a"),
        baseline_b_false_answer_rate=_false_answer_rate("b"),
        devtrace_false_answer_rate=_false_answer_rate("devtrace"),
        devtrace_correct_abstention_rate=correct_abstention_rate,
    )
