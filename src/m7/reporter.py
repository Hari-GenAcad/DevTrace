"""
DevTrace — Module 7: Evaluation report writer.

Serialises an EvaluationReport to:
  1. Machine-readable JSON  (per-case records + aggregates)
  2. Human-readable Markdown summary

The Markdown answers the central thesis questions:
  - Did DevTrace reduce false answers?
  - Did applicability help with version conflicts?
  - Did verification reduce unsupported claims?
  - Did root-cause survival improve reliability?
  - Where did DevTrace fail?
  - Where did baselines outperform it?
  - How often did DevTrace abstain unnecessarily?
"""

from __future__ import annotations

import json
from pathlib import Path

from src.m7.models import EvaluationReport


def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _n(value: int | None) -> str:
    if value is None:
        return "N/A"
    return str(value)


def write_json_report(report: EvaluationReport, output_dir: Path) -> Path:
    """Write the machine-readable JSON report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "evaluation_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, default=str)
    return out_path


def write_markdown_report(report: EvaluationReport, output_dir: Path) -> Path:
    """Write the human-readable Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "evaluation_report.md"

    lines: list[str] = []

    lines.append("# DevTrace Module 7 — Evaluation Report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Dataset:** `{report.dataset_path}`")
    lines.append(f"- **Total cases:** {report.total_cases}")
    lines.append(f"- **Evaluation mode:** `{report.evaluation_mode}`")
    lines.append(f"- **Generated at (UTC):** {report.generated_at_utc or 'N/A'}")
    lines.append(f"- **Git commit:** `{report.git_commit or 'N/A'}`")
    lines.append(f"- **Python:** {report.python_version or 'N/A'}")
    lines.append(f"- **Baseline B threshold:** {report.baseline_b_threshold}")
    for k, v in report.model_config_info.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")
    lines.append(
        "> **Interpretation note:** Deterministic mode is a controlled fixture run that "
        "exercises production contracts. It is not an independent measurement of live-model quality."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Aggregate metrics comparison
    # ------------------------------------------------------------------
    lines.append("## Aggregate Metrics Comparison")
    lines.append("")
    lines.append(
        "| Metric | Baseline A (Naive RAG) | Baseline B (Threshold) | DevTrace (Full) |"
    )
    lines.append("|:---|:---:|:---:|:---:|")

    a = report.aggregate_a
    b = report.aggregate_b
    dt = report.aggregate_devtrace

    def row(label: str, field_name: str) -> str:
        av = _pct(getattr(a, field_name, None)) if a else "N/A"
        bv = _pct(getattr(b, field_name, None)) if b else "N/A"
        dtv = _pct(getattr(dt, field_name, None)) if dt else "N/A"
        return f"| {label} | {av} | {bv} | {dtv} |"

    lines.append(row("Retrieval Hit Rate", "retrieval_hit_rate"))
    lines.append(row("Forbidden Retrieval Rate ↓", "forbidden_retrieval_rate"))
    lines.append(row("Citation Validity Rate", "citation_validity_rate"))
    lines.append(row("Citation Correctness Rate", "citation_correctness_rate"))
    lines.append(row("False Answer Rate ↓ (lower=better)", "false_answer_rate"))
    lines.append(row("False Abstention Rate ↓ (lower=better)", "false_abstention_rate"))
    lines.append("")

    # ------------------------------------------------------------------
    # DevTrace outcome distribution
    # ------------------------------------------------------------------
    if report.outcome_distribution is not None:
        od = report.outcome_distribution
        lines.append("## DevTrace Outcome Distribution")
        lines.append("")
        lines.append("| Outcome | Count |")
        lines.append("|:---|:---:|")
        lines.append(f"| ANSWERED_FULL | {od.answered_full} |")
        lines.append(f"| ANSWERED_PARTIAL | {od.answered_partial} |")
        lines.append(f"| INSUFFICIENT_EVIDENCE | {od.insufficient_evidence} |")
        lines.append(f"| NEEDS_INFO | {od.needs_info} |")
        lines.append(f"| DEGRADED | {od.degraded} |")
        lines.append(f"| **Total** | **{od.total}** |")
        lines.append("")

    # ------------------------------------------------------------------
    # Retry statistics
    # ------------------------------------------------------------------
    if report.retry_stats is not None:
        rs = report.retry_stats
        lines.append("## DevTrace Retry Analysis")
        lines.append("")
        lines.append(f"- **Initial root-cause failures:** {rs.initial_root_cause_failures}")
        lines.append(f"- **Retries attempted:** {rs.retries_attempted}")
        lines.append(f"- **Retries succeeded:** {rs.retries_succeeded}")
        lines.append(f"- **Retries failed:** {rs.retries_failed}")
        lines.append(f"- **Retry recovery rate:** {_pct(rs.retry_recovery_rate)}")
        lines.append(f"- **Max retry invariant violated:** {rs.max_retry_count_exceeded}")
        lines.append("")

    # ------------------------------------------------------------------
    # Category-level breakdown
    # ------------------------------------------------------------------
    if report.category_metrics:
        lines.append("## Category-Level Results")
        lines.append("")
        lines.append(
            "| Category | Cases | A: False Answer ↓ | B: False Answer ↓ | DT: False Answer ↓ |"
            " A: Hit Rate | B: Hit Rate | DT: Hit Rate |"
        )
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for cm in report.category_metrics:
            fa_a = _pct(cm.metrics_a.false_answer_rate if cm.metrics_a else None)
            fa_b = _pct(cm.metrics_b.false_answer_rate if cm.metrics_b else None)
            fa_dt = _pct(cm.metrics_devtrace.false_answer_rate if cm.metrics_devtrace else None)
            hr_a = _pct(cm.metrics_a.retrieval_hit_rate if cm.metrics_a else None)
            hr_b = _pct(cm.metrics_b.retrieval_hit_rate if cm.metrics_b else None)
            hr_dt = _pct(cm.metrics_devtrace.retrieval_hit_rate if cm.metrics_devtrace else None)
            lines.append(
                f"| {cm.category} | {cm.case_count} | {fa_a} | {fa_b} | {fa_dt} |"
                f" {hr_a} | {hr_b} | {hr_dt} |"
            )
        lines.append("")

    # ------------------------------------------------------------------
    # Version-conflict analysis
    # ------------------------------------------------------------------
    if report.version_conflict_analysis is not None:
        vc = report.version_conflict_analysis
        lines.append("## Version-Conflict Analysis")
        lines.append("")
        lines.append(f"**Total version-conflict cases:** {vc.total_version_conflict_cases}")
        lines.append("")
        lines.append("| Metric | Baseline A | Baseline B | DevTrace |")
        lines.append("|:---|:---:|:---:|:---:|")
        lines.append(
            f"| Forbidden doc retrieved rate | "
            f"{_pct(vc.baseline_a_forbidden_retrieved_rate)} | "
            f"{_pct(vc.baseline_b_forbidden_retrieved_rate)} | "
            f"{_pct(vc.devtrace_forbidden_retrieved_rate)} |"
        )
        lines.append(
            f"| False answer rate | "
            f"{_pct(vc.baseline_a_false_answer_rate)} | "
            f"{_pct(vc.baseline_b_false_answer_rate)} | "
            f"{_pct(vc.devtrace_false_answer_rate)} |"
        )
        lines.append(
            f"| Correct applicability exclusion (DevTrace only) | — | — | "
            f"{_pct(vc.devtrace_correct_exclusion_rate)} |"
        )
        lines.append("")
        lines.append(
            "> **Interpretation:** Correct exclusion rate measures how often DevTrace "
            "successfully excluded forbidden (wrong-version) documents via its M4 "
            "applicability filter. This directly demonstrates the core architecture thesis."
        )
        lines.append("")

    # ------------------------------------------------------------------
    # Unsupported-case analysis
    # ------------------------------------------------------------------
    if report.unsupported_case_analysis is not None:
        uc = report.unsupported_case_analysis
        lines.append("## Unsupported-Case Analysis")
        lines.append("")
        lines.append(
            f"**Cases where abstention was expected:** {uc.total_abstention_expected_cases}"
        )
        lines.append("")
        lines.append("| Metric | Baseline A | Baseline B | DevTrace |")
        lines.append("|:---|:---:|:---:|:---:|")
        lines.append(
            f"| False answer rate | "
            f"{_pct(uc.baseline_a_false_answer_rate)} | "
            f"{_pct(uc.baseline_b_false_answer_rate)} | "
            f"{_pct(uc.devtrace_false_answer_rate)} |"
        )
        lines.append(
            f"| Correct abstention rate (DevTrace) | — | — | "
            f"{_pct(uc.devtrace_correct_abstention_rate)} |"
        )
        lines.append("")

    # ------------------------------------------------------------------
    # Summary answers to thesis questions
    # ------------------------------------------------------------------
    lines.append("## Thesis Question Answers")
    lines.append("")
    if report.summary:
        for question, answer in report.summary.items():
            lines.append(f"### {question}")
            lines.append("")
            lines.append(answer)
            lines.append("")
    else:
        lines.append("*(Summary not available for this evaluation run.)*")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "*This report was generated by DevTrace Module 7 evaluation. "
        "Results reflect the configured evaluation mode and dataset.*"
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path
