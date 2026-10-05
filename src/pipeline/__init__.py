"""
DevTrace — Module 4: Pipeline layer public API.

Exposes:
    BaselinePipelineResult — structured result from run_baseline_diagnosis
    run_baseline_diagnosis — end-to-end M4 pipeline function
"""

from src.pipeline.baseline import BaselinePipelineResult, run_baseline_diagnosis

__all__ = [
    "BaselinePipelineResult",
    "run_baseline_diagnosis",
]
