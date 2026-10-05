"""
DevTrace — Module 4: Applicability layer public API.

Exposes:
    check_applicability   — single-chunk decision
    filter_applicable     — batch filter returning (applicable, not_applicable) pairs
    ApplicabilityDecision — string enum for APPLICABLE / NOT_APPLICABLE / UNKNOWN
"""

from src.applicability.checker import (
    ApplicabilityDecision,
    check_applicability,
    filter_applicable,
)

__all__ = [
    "ApplicabilityDecision",
    "check_applicability",
    "filter_applicable",
]
