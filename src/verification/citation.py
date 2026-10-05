"""
DevTrace — Module 5: Deterministic citation validity checker.

Responsibility
--------------
Before any semantic (LLM) verification, check whether the evidence IDs
cited by a diagnosis claim actually exist in the M4 applicable evidence bundle.

This check is:
    - Fully deterministic (no LLM involved)
    - O(1) per claim via a set lookup
    - The primary defence against hallucinated chunk IDs

Rules
-----
1. If a claim has no evidence_ids → CitationValidity.EMPTY.
2. If ALL cited IDs are present in the applicable bundle → CitationValidity.VALID.
3. If ANY cited ID is missing (hallucinated, NOT_APPLICABLE, or malformed)
   → CitationValidity.INVALID.

Only the M4 applicable evidence bundle is authoritative.
Retrieved-but-NOT_APPLICABLE chunks are NOT valid citations.
Nonexistent IDs are NOT valid citations.

This function intentionally knows nothing about claim semantics — it only
answers "is this ID in the bundle?".
"""

from __future__ import annotations

from src.models.contracts import DiagnosisClaim, RetrievalResult
from src.verification.models import CitationValidity


def build_applicable_id_set(applicable_results: list[RetrievalResult]) -> set[str]:
    """
    Build the authoritative set of valid evidence IDs from M4's applicable bundle.

    Args:
        applicable_results: The RetrievalResult list produced by M4's filter_applicable().

    Returns:
        A set of chunk_ids that are valid citation targets.
    """
    return {rr.chunk_id for rr in applicable_results}


def check_citation_validity(
    claim: DiagnosisClaim,
    applicable_id_set: set[str],
) -> tuple[CitationValidity, list[str]]:
    """
    Deterministically check whether a claim's citations are valid.

    Args:
        claim:              The DiagnosisClaim to evaluate.
        applicable_id_set:  Set of chunk_ids from build_applicable_id_set().

    Returns:
        (CitationValidity, invalid_ids) where invalid_ids is the list of
        cited IDs that were NOT found in the applicable bundle.
        invalid_ids is empty when CitationValidity is VALID or EMPTY.
    """
    if not claim.evidence_ids:
        return CitationValidity.EMPTY, []

    invalid_ids = [eid for eid in claim.evidence_ids if eid not in applicable_id_set]

    if invalid_ids:
        return CitationValidity.INVALID, invalid_ids

    return CitationValidity.VALID, []
