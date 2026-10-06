"""
DevTrace — Module 7: Evaluation, Baselines & Reliability Measurement.

Public API:

    Metrics:
        compute_retrieval_hit_rate
        compute_citation_validity
        compute_citation_correctness
        compute_false_answer_rate
        compute_false_abstention_rate
        compute_outcome_distribution
        compute_retry_stats

    Baselines:
        run_baseline_a        (Naive RAG: retrieve → generate, no controls)
        run_baseline_b        (Retrieve + score threshold)

    DevTrace evaluation:
        run_devtrace_eval     (Full M6 orchestration on an eval case)

    Case-level runner:
        run_evaluation_case   (Single case across all three systems)

    Report:
        run_evaluation        (Full evaluation suite entry point)
"""

from src.m7.metrics import (
    compute_citation_correctness,
    compute_citation_validity,
    compute_false_abstention_rate,
    compute_false_answer_rate,
    compute_outcome_distribution,
    compute_retrieval_hit_rate,
    compute_retry_stats,
)
from src.m7.models import (
    BaselineAResult,
    BaselineBResult,
    CategoryMetrics,
    EvaluationCaseRecord,
    EvaluationReport,
    SystemLabel,
)
from src.m7.runner import run_evaluation

__all__ = [
    # Metrics
    "compute_retrieval_hit_rate",
    "compute_citation_validity",
    "compute_citation_correctness",
    "compute_false_answer_rate",
    "compute_false_abstention_rate",
    "compute_outcome_distribution",
    "compute_retry_stats",
    # Models
    "BaselineAResult",
    "BaselineBResult",
    "CategoryMetrics",
    "EvaluationCaseRecord",
    "EvaluationReport",
    "SystemLabel",
    # Entry point
    "run_evaluation",
]
