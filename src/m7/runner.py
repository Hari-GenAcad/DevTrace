"""
DevTrace — Module 7: Main evaluation runner.

Coordinates:
  1. Dataset loading (M2 EvaluationDataset)
  2. Retriever setup
  3. LLM client setup (FakeLLM in deterministic mode, Gemini in live mode)
  4. Per-case execution: Baseline A, Baseline B, DevTrace
  5. Metric computation
  6. Aggregate + category metrics
  7. Specialised analyses (version-conflict, unsupported, retry)
  8. Summary generation
  9. Report writing

Entry point: run_evaluation(config) or python -m src.m7.runner

Design rules:
  - Gold labels are passed to baselines ONLY for post-hoc metric extraction.
  - The production M6 pipeline is not modified.
  - FakeLLMClient is used in deterministic mode with a fixed scripted response
    that is consistent (not randomised) so the evaluation is reproducible.
  - In live mode, real Gemini responses are used (requires GOOGLE_API_KEY).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.evaluation.dataset import EvaluationCase, load_eval_dataset
from src.m7.baselines import run_baseline_a, run_baseline_b
from src.m7.config import EvaluationConfig
from src.m7.devtrace_eval import run_devtrace_eval
from src.m7.metrics import (
    build_aggregate_metrics,
    compute_case_metrics_a,
    compute_case_metrics_b,
    compute_case_metrics_devtrace,
    compute_outcome_distribution,
    compute_retry_stats,
    compute_unsupported_case_analysis,
    compute_version_conflict_analysis,
)
from src.m7.models import (
    CategoryMetrics,
    EvaluationCaseRecord,
    EvaluationReport,
)
from src.m7.reporter import write_json_report, write_markdown_report
from src.normalization.normalizer import normalize_incident
from src.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic FakeLLM response for evaluation mode
# ---------------------------------------------------------------------------

def _make_scripted_diagnosis_json(chunk_id: str = "CHUNK-001") -> str:
    """
    Return a minimal valid DiagnosisResult JSON for deterministic evaluation.

    In deterministic mode, the FakeLLMClient returns this for diagnosis calls.
    The chunk_id should match a real chunk in the applicable results when possible;
    in deterministic mode we use a placeholder since exact IDs differ per case.
    """
    return json.dumps({
        "claims": [
            {
                "role": "root_cause",
                "text": "Based on the provided evidence, the root cause has been identified.",
                "evidence_ids": [chunk_id],
            },
            {
                "role": "fix",
                "text": "Apply the fix described in the applicable documentation.",
                "evidence_ids": [chunk_id],
            },
        ]
    })


def _make_scripted_verifier_json(is_verified: bool = True) -> str:
    """
    Return a minimal valid verifier JSON response for deterministic evaluation.
    """
    return json.dumps({
        "citation_correct": is_verified,
        "sufficient": is_verified,
        "contradicted": False,
        "reasoning": "Deterministic evaluation mode.",
    })


def _make_scripted_naive_answer() -> str:
    """Return a scripted plain-text answer for baselines in deterministic mode."""
    return (
        "Based on the documentation provided, the issue is caused by a configuration "
        "mismatch. Please review your credentials and version-specific settings as "
        "described in the relevant documentation."
    )


# ---------------------------------------------------------------------------
# Retriever setup
# ---------------------------------------------------------------------------

def _build_retriever(config: EvaluationConfig) -> HybridRetriever | None:
    """
    Attempt to build and load a HybridRetriever from the existing Chroma index.

    Returns None if the retriever cannot be loaded (evaluation continues
    in retrieval-stub mode for deterministic tests).
    """
    try:
        from src.ingestion.loader import load_corpus  # noqa: PLC0415
        chunks = load_corpus()
        retriever = HybridRetriever(chunks)
        retriever.load()
        logger.info("HybridRetriever loaded with %d chunks.", len(chunks))
        return retriever
    except Exception as exc:
        logger.warning("Could not load HybridRetriever: %s. Using stub retriever.", exc)
        return None


# ---------------------------------------------------------------------------
# LLM client factories
# ---------------------------------------------------------------------------

def _build_deterministic_llm(
    scripted_responses: list[str] | None = None,
    fixed_response: str | None = None,
) -> Any:
    """Build a FakeLLMClient for deterministic evaluation."""
    from src.llm.fake import FakeLLMClient  # noqa: PLC0415
    if scripted_responses is not None:
        return FakeLLMClient(responses=scripted_responses)
    fixed = fixed_response or _make_scripted_naive_answer()
    return FakeLLMClient(response=fixed)


def _build_live_llm(config: EvaluationConfig) -> Any:
    """Build a GeminiClient for live evaluation."""
    from src.llm.gemini import GeminiClient  # noqa: PLC0415
    return GeminiClient(
        model_name=config.live_model_name,
        temperature=config.live_temperature,
    )


# ---------------------------------------------------------------------------
# Stub retriever for deterministic mode
# ---------------------------------------------------------------------------

class _StubRetriever:
    """
    Minimal stub retriever used when the real Chroma index is unavailable.

    Returns a fixed empty result list so the pipeline degrades gracefully.
    """

    def retrieve_as_contracts(self, normalized: Any) -> list:
        return []


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------

def _run_case_deterministic(
    case: EvaluationCase,
    retriever: Any,
    config: EvaluationConfig,
) -> EvaluationCaseRecord:
    """
    Run one evaluation case in deterministic (FakeLLM) mode.

    FakeLLM clients return scripted responses:
      - Baseline A/B: plain-text answer
      - DevTrace: diagnosis JSON + verifier JSON
    """
    incident = case.incident

    # Build normalised incident for baselines
    normalized = normalize_incident(
        description=incident.description,
        current_version=incident.current_version,
        previous_version=incident.previous_version,
        error_codes=list(incident.error_codes),
        product=incident.product,
        context=dict(incident.context),
    )

    # ------------------------------------------------------------------
    # Baseline A
    # ------------------------------------------------------------------
    baseline_a_result = None
    if config.run_baseline_a:
        llm_a = _build_deterministic_llm(fixed_response=_make_scripted_naive_answer())
        try:
            baseline_a_result = run_baseline_a(
                normalized=normalized,
                retriever=retriever,
                llm_client=llm_a,
            )
        except Exception as exc:
            logger.error("Baseline A failed for %s: %s", case.case_id, exc)
            from src.m7.models import BaselineAResult  # noqa: PLC0415
            baseline_a_result = BaselineAResult(abstained=True, error=str(exc))

    # ------------------------------------------------------------------
    # Baseline B
    # ------------------------------------------------------------------
    baseline_b_result = None
    if config.run_baseline_b:
        llm_b = _build_deterministic_llm(fixed_response=_make_scripted_naive_answer())
        try:
            baseline_b_result = run_baseline_b(
                normalized=normalized,
                retriever=retriever,
                llm_client=llm_b,
                threshold=config.baseline_b_threshold,
            )
        except Exception as exc:
            logger.error("Baseline B failed for %s: %s", case.case_id, exc)
            from src.m7.models import BaselineBResult  # noqa: PLC0415
            baseline_b_result = BaselineBResult(
                threshold_used=config.baseline_b_threshold,
                threshold_passed=False,
                abstained=True,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # DevTrace
    # ------------------------------------------------------------------
    devtrace_result = None
    if config.run_devtrace:
        # DevTrace requires: diagnosis LLM response + verifier LLM response
        # In deterministic mode, provide scripted multi-call sequence.
        # The FakeLLM responses are queued: [diagnosis, verifier_rc, verifier_fix, ...]
        # We queue enough responses for a two-claim diagnosis with one attempt.
        diagnosis_json = _make_scripted_diagnosis_json()
        verifier_json_pass = _make_scripted_verifier_json(is_verified=True)
        # Queue: diagnosis call + 2 verifier calls (root_cause, fix)
        llm_dt = _build_deterministic_llm(scripted_responses=[
            diagnosis_json,
            verifier_json_pass,
            verifier_json_pass,
        ])
        try:
            devtrace_result = run_devtrace_eval(
                description=incident.description,
                current_version=incident.current_version,
                previous_version=incident.previous_version,
                error_codes=list(incident.error_codes),
                product=incident.product,
                retriever=retriever,
                llm_client=llm_dt,
                verifier_llm_client=llm_dt,
                gold_forbidden_doc_ids=list(case.forbidden_doc_ids),
            )
        except Exception as exc:
            logger.error("DevTrace eval failed for %s: %s", case.case_id, exc)
            from src.m7.models import DevTraceEvalResult  # noqa: PLC0415
            devtrace_result = DevTraceEvalResult(
                final_outcome="DEGRADED",
                abstained=True,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Build initial record (without metrics)
    # ------------------------------------------------------------------
    record = EvaluationCaseRecord(
        case_id=case.case_id,
        case_class=case.case_class,
        gold_expected_outcome=case.expected_outcome.value,
        gold_expected_completeness=(
            case.expected_completeness.value if case.expected_completeness else None
        ),
        gold_doc_ids=list(case.gold_doc_ids),
        gold_forbidden_doc_ids=list(case.forbidden_doc_ids),
        baseline_a=baseline_a_result,
        baseline_b=baseline_b_result,
        devtrace=devtrace_result,
        notes=case.notes,
    )

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    metrics_a = compute_case_metrics_a(record) if baseline_a_result else None
    metrics_b = compute_case_metrics_b(record) if baseline_b_result else None
    metrics_dt = compute_case_metrics_devtrace(record) if devtrace_result else None

    # Return updated record with metrics
    return record.model_copy(update={
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "metrics_devtrace": metrics_dt,
    })


def _run_case_live(
    case: EvaluationCase,
    retriever: Any,
    llm_client: Any,
    config: EvaluationConfig,
) -> EvaluationCaseRecord:
    """
    Run one evaluation case in live (real Gemini) mode.

    A single shared llm_client is used for all systems for fair comparison.
    """
    incident = case.incident

    normalized = normalize_incident(
        description=incident.description,
        current_version=incident.current_version,
        previous_version=incident.previous_version,
        error_codes=list(incident.error_codes),
        product=incident.product,
        context=dict(incident.context),
    )

    baseline_a_result = None
    if config.run_baseline_a:
        try:
            baseline_a_result = run_baseline_a(
                normalized=normalized,
                retriever=retriever,
                llm_client=llm_client,
            )
        except Exception as exc:
            logger.error("Baseline A failed for %s: %s", case.case_id, exc)
            from src.m7.models import BaselineAResult  # noqa: PLC0415
            baseline_a_result = BaselineAResult(abstained=True, error=str(exc))

    baseline_b_result = None
    if config.run_baseline_b:
        try:
            baseline_b_result = run_baseline_b(
                normalized=normalized,
                retriever=retriever,
                llm_client=llm_client,
                threshold=config.baseline_b_threshold,
            )
        except Exception as exc:
            logger.error("Baseline B failed for %s: %s", case.case_id, exc)
            from src.m7.models import BaselineBResult  # noqa: PLC0415
            baseline_b_result = BaselineBResult(
                threshold_used=config.baseline_b_threshold,
                threshold_passed=False,
                abstained=True,
                error=str(exc),
            )

    devtrace_result = None
    if config.run_devtrace:
        try:
            devtrace_result = run_devtrace_eval(
                description=incident.description,
                current_version=incident.current_version,
                previous_version=incident.previous_version,
                error_codes=list(incident.error_codes),
                product=incident.product,
                retriever=retriever,
                llm_client=llm_client,
                verifier_llm_client=llm_client,
                gold_forbidden_doc_ids=list(case.forbidden_doc_ids),
            )
        except Exception as exc:
            logger.error("DevTrace eval failed for %s: %s", case.case_id, exc)
            from src.m7.models import DevTraceEvalResult  # noqa: PLC0415
            devtrace_result = DevTraceEvalResult(
                final_outcome="DEGRADED",
                abstained=True,
                error=str(exc),
            )

    record = EvaluationCaseRecord(
        case_id=case.case_id,
        case_class=case.case_class,
        gold_expected_outcome=case.expected_outcome.value,
        gold_expected_completeness=(
            case.expected_completeness.value if case.expected_completeness else None
        ),
        gold_doc_ids=list(case.gold_doc_ids),
        gold_forbidden_doc_ids=list(case.forbidden_doc_ids),
        baseline_a=baseline_a_result,
        baseline_b=baseline_b_result,
        devtrace=devtrace_result,
        notes=case.notes,
    )

    metrics_a = compute_case_metrics_a(record) if baseline_a_result else None
    metrics_b = compute_case_metrics_b(record) if baseline_b_result else None
    metrics_dt = compute_case_metrics_devtrace(record) if devtrace_result else None

    return record.model_copy(update={
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "metrics_devtrace": metrics_dt,
    })


# ---------------------------------------------------------------------------
# Category breakdown
# ---------------------------------------------------------------------------

def _build_category_metrics(
    records: list[EvaluationCaseRecord],
) -> list[CategoryMetrics]:
    """Compute per-category aggregate metrics."""
    from src.evaluation.dataset import VALID_CASE_CLASSES  # noqa: PLC0415
    result: list[CategoryMetrics] = []
    for cat in sorted(VALID_CASE_CLASSES):
        cat_records = [r for r in records if r.case_class == cat]
        if not cat_records:
            continue
        result.append(
            CategoryMetrics(
                category=cat,
                case_count=len(cat_records),
                metrics_a=build_aggregate_metrics(cat_records, "a"),
                metrics_b=build_aggregate_metrics(cat_records, "b"),
                metrics_devtrace=build_aggregate_metrics(cat_records, "devtrace"),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Summary generator
# ---------------------------------------------------------------------------

def _build_summary(report: EvaluationReport) -> dict[str, str]:
    """
    Generate human-readable thesis question answers from evaluation results.
    """
    summary: dict[str, str] = {}

    a = report.aggregate_a
    b = report.aggregate_b
    dt = report.aggregate_devtrace

    def _pct(v: float | None) -> str:
        return f"{v * 100:.1f}%" if v is not None else "N/A"

    # Q1: Did DevTrace reduce false answers?
    if a and dt:
        dt_fa = dt.false_answer_rate or 0.0
        a_fa = a.false_answer_rate or 0.0
        reduction = a_fa - dt_fa
        direction = "reduced" if reduction > 0 else "did not reduce"
        summary["Did DevTrace reduce false answers?"] = (
            f"DevTrace {direction} the false answer rate compared to Baseline A. "
            f"Baseline A: {_pct(a.false_answer_rate)}, "
            f"Baseline B: {_pct(b.false_answer_rate if b else None)}, "
            f"DevTrace: {_pct(dt.false_answer_rate)}. "
            f"Reduction vs Baseline A: {reduction * 100:.1f} percentage points."
        )

    # Q2: Did applicability help with version conflicts?
    if report.version_conflict_analysis:
        vc = report.version_conflict_analysis
        summary["Did applicability help with version conflicts?"] = (
            f"DevTrace correctly excluded wrong-version documents in "
            f"{_pct(vc.devtrace_correct_exclusion_rate)} of version-conflict cases. "
            f"Baseline A retrieved forbidden docs in {_pct(vc.baseline_a_forbidden_retrieved_rate)} "
            f"of cases; DevTrace in {_pct(vc.devtrace_forbidden_retrieved_rate)} of cases. "
            f"DevTrace version-conflict false answer rate: {_pct(vc.devtrace_false_answer_rate)}."
        )

    # Q3: Did verification reduce unsupported claims?
    if report.unsupported_case_analysis:
        uc = report.unsupported_case_analysis
        summary["Did evidence verification reduce unsupported claims?"] = (
            f"For cases where abstention was expected ({uc.total_abstention_expected_cases} cases), "
            f"Baseline A produced false answers {_pct(uc.baseline_a_false_answer_rate)} of the time. "
            f"DevTrace produced false answers {_pct(uc.devtrace_false_answer_rate)} of the time "
            f"and correctly abstained {_pct(uc.devtrace_correct_abstention_rate)} of the time."
        )

    # Q4: Retry behavior
    if report.retry_stats:
        rs = report.retry_stats
        summary["Did root-cause survival and retry improve reliability?"] = (
            f"DevTrace had {rs.initial_root_cause_failures} initial root-cause failures. "
            f"Retry was attempted {rs.retries_attempted} times. "
            f"Retry recovery rate: {_pct(rs.retry_recovery_rate)}. "
            f"Max retry invariant (≤1 retry) violated: {rs.max_retry_count_exceeded}."
        )

    # Q5: False abstentions
    if dt:
        summary["How often did DevTrace abstain unnecessarily?"] = (
            f"DevTrace false abstention rate: {_pct(dt.false_abstention_rate)}. "
            f"Baseline A: {_pct(a.false_abstention_rate if a else None)}. "
            f"Note: some abstentions are architecturally correct (NEEDS_INFO, INSUFFICIENT_EVIDENCE) "
            f"and are expected in ambiguous cases."
        )

    # Q6: Where did baselines outperform DevTrace?
    if a and dt:
        a_hr = a.retrieval_hit_rate or 0.0
        dt_hr = dt.retrieval_hit_rate or 0.0
        if a_hr > dt_hr:
            note = (
                f"Baseline A had a higher retrieval hit rate ({_pct(a.retrieval_hit_rate)}) "
                f"than DevTrace ({_pct(dt.retrieval_hit_rate)}), but this is expected "
                f"since DevTrace applies applicability filtering which can exclude gold docs "
                f"that happen to have wrong-version metadata."
            )
        else:
            note = (
                f"DevTrace retrieval hit rate ({_pct(dt.retrieval_hit_rate)}) was comparable "
                f"to Baseline A ({_pct(a.retrieval_hit_rate)}). Baselines may have higher "
                f"false-positive citation rates since they include all retrieved docs."
            )
        summary["Where did baselines outperform DevTrace?"] = note

    return summary


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def run_evaluation(config: EvaluationConfig | None = None) -> EvaluationReport:
    """
    Run the full M7 evaluation suite.

    Args:
        config: EvaluationConfig. If None, loads from environment via config_from_env().

    Returns:
        EvaluationReport with all results and analyses.
    """
    if config is None:
        from src.m7.config import config_from_env  # noqa: PLC0415
        config = config_from_env()

    logging.basicConfig(level=getattr(logging, config.log_level, logging.INFO))
    logger.info(
        "Starting M7 evaluation. mode=%s, dataset=%s",
        config.evaluation_mode,
        config.dataset_path,
    )

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    dataset = load_eval_dataset(config.dataset_path)
    logger.info("Loaded %d evaluation cases.", len(dataset.cases))

    # ------------------------------------------------------------------
    # Setup retriever
    # ------------------------------------------------------------------
    retriever_or_stub: Any
    if config.evaluation_mode == "deterministic":
        # Try loading real retriever; fall back to stub
        real_retriever = _build_retriever(config)
        retriever_or_stub = real_retriever if real_retriever is not None else _StubRetriever()
    else:
        real_retriever = _build_retriever(config)
        if real_retriever is None:
            raise RuntimeError("Live evaluation requires a working HybridRetriever.")
        retriever_or_stub = real_retriever

    # ------------------------------------------------------------------
    # Setup LLM client(s)
    # ------------------------------------------------------------------
    live_llm = None
    if config.evaluation_mode == "live":
        live_llm = _build_live_llm(config)

    # ------------------------------------------------------------------
    # Run cases
    # ------------------------------------------------------------------
    records: list[EvaluationCaseRecord] = []
    for i, case in enumerate(dataset.cases, 1):
        logger.info(
            "[%d/%d] Running case %s (%s)...",
            i,
            len(dataset.cases),
            case.case_id,
            case.case_class,
        )
        if config.evaluation_mode == "deterministic":
            record = _run_case_deterministic(case, retriever_or_stub, config)
        else:
            record = _run_case_live(case, retriever_or_stub, live_llm, config)
        records.append(record)
        logger.info("  Done: %s", case.case_id)

    # ------------------------------------------------------------------
    # Compute aggregates
    # ------------------------------------------------------------------
    logger.info("Computing aggregate metrics...")
    agg_a = build_aggregate_metrics(records, "a") if config.run_baseline_a else None
    agg_b = build_aggregate_metrics(records, "b") if config.run_baseline_b else None
    agg_dt = build_aggregate_metrics(records, "devtrace") if config.run_devtrace else None

    outcome_dist = compute_outcome_distribution(records) if config.run_devtrace else None
    retry_stats = compute_retry_stats(records) if config.run_devtrace else None

    # ------------------------------------------------------------------
    # Category breakdown
    # ------------------------------------------------------------------
    category_metrics = _build_category_metrics(records)

    # ------------------------------------------------------------------
    # Specialised analyses
    # ------------------------------------------------------------------
    vc_analysis = compute_version_conflict_analysis(records)
    uc_analysis = compute_unsupported_case_analysis(records)

    # ------------------------------------------------------------------
    # Assemble report
    # ------------------------------------------------------------------
    report = EvaluationReport(
        dataset_path=str(config.dataset_path),
        total_cases=len(records),
        evaluation_mode=config.evaluation_mode,
        model_config_info=config.model_config_info(),
        baseline_b_threshold=config.baseline_b_threshold,
        case_records=records,
        aggregate_a=agg_a,
        aggregate_b=agg_b,
        aggregate_devtrace=agg_dt,
        outcome_distribution=outcome_dist,
        retry_stats=retry_stats,
        category_metrics=category_metrics,
        version_conflict_analysis=vc_analysis,
        unsupported_case_analysis=uc_analysis,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    report.summary = _build_summary(report)

    # ------------------------------------------------------------------
    # Write reports
    # ------------------------------------------------------------------
    json_path = write_json_report(report, config.output_dir)
    md_path = write_markdown_report(report, config.output_dir)
    logger.info("JSON report written to: %s", json_path)
    logger.info("Markdown report written to: %s", md_path)

    logger.info("M7 evaluation complete. %d cases processed.", len(records))
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    from src.m7.config import config_from_env

    try:
        cfg = config_from_env()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    run_evaluation(cfg)
