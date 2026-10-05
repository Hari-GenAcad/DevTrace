"""
DevTrace — Module 5: Evidence Verifier.

Responsibility
--------------
Independently verify each claim in a DiagnosisResult against the applicable
evidence bundle from M4.

Verification pipeline per claim
---------------------------------
Step 1 — Deterministic citation validity (citation.py)
    Are the cited evidence IDs actually present in the applicable bundle?
    No LLM involved. Pure set membership check.
    Protects against hallucinated chunk IDs.

Step 2 — Semantic verification via LLM (prompts.py + LLMClient)
    Only runs when citation_validity == VALID.
    Asks an independent LLM:
        - Does the cited evidence actually support the claim? (citation_correct)
        - Is the evidence sufficient? (sufficient)
        - Does any applicable evidence contradict the claim? (contradicted)
    The verifier LLM is called independently from the diagnosis generator.

Step 3 — Verdict determination
    VERIFIED only when:
        citation_validity == VALID
        AND citation_correct == True
        AND sufficient == True
        AND contradicted == False
    Otherwise REJECTED.

Short-circuit behaviour
------------------------
If citation_validity is INVALID or EMPTY:
    → REJECTED immediately (no LLM call needed).
    → citation_correct, sufficient = False by definition.
    → contradiction check is skipped (not enough information).

Design decisions
----------------
- Uses the M1 LLMClient abstraction (never calls Gemini SDK directly).
- Receives only APPLICABLE evidence (M4's boundary is not re-implemented here).
- Does NOT retry on failure (M6's responsibility).
- Does NOT make the final outcome decision (M6's responsibility).
- Raises SchemaValidationError if verifier LLM returns unparseable output.
- Raises LLMError if the LLM provider fails.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.errors import LLMError, SchemaValidationError
from src.llm.base import LLMClient
from src.models.contracts import DiagnosisClaim, DiagnosisResult, RetrievalResult
from src.normalization.normalizer import NormalizedIncident
from src.verification.citation import build_applicable_id_set, check_citation_validity
from src.verification.models import (
    CitationValidity,
    ClaimVerification,
    DiagnosisVerification,
    VerificationVerdict,
)
from src.verification.prompts import build_verification_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verifier response parsing
# ---------------------------------------------------------------------------

def _parse_verifier_response(raw_text: str) -> dict[str, Any]:
    """
    Parse raw LLM verifier response into a dict.

    Strips markdown fences if present.

    Raises:
        SchemaValidationError: If the response is not valid JSON or is missing
                               required keys.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # strip opening fence line
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # strip closing fence
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(
            f"Verifier LLM response is not valid JSON: {exc}\n"
            f"Raw response (first 500 chars): {raw_text[:500]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise SchemaValidationError(
            f"Verifier response must be a JSON object, got {type(data).__name__}."
        )

    required_keys = {
        "citation_correct",
        "sufficient",
        "contradicted",
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
        "reason",
    }
    missing = required_keys - data.keys()
    if missing:
        raise SchemaValidationError(
            f"Verifier response is missing required keys: {sorted(missing)}. "
            f"Got keys: {sorted(data.keys())}"
        )

    # Type coerce booleans (LLMs sometimes return strings)
    for bool_key in ("citation_correct", "sufficient", "contradicted"):
        val = data[bool_key]
        if isinstance(val, str):
            data[bool_key] = val.strip().lower() in ("true", "yes", "1")
        elif not isinstance(val, bool):
            data[bool_key] = bool(val)

    # Ensure list fields
    for list_key in ("supporting_evidence_ids", "contradicting_evidence_ids"):
        if not isinstance(data[list_key], list):
            data[list_key] = []

    # Ensure reason is a string
    if not isinstance(data.get("reason"), str):
        data["reason"] = str(data.get("reason", ""))

    return data


# ---------------------------------------------------------------------------
# Per-claim verifier
# ---------------------------------------------------------------------------

def _verify_claim(
    *,
    claim: DiagnosisClaim,
    normalized: NormalizedIncident,
    applicable_results: list[RetrievalResult],
    applicable_id_set: set[str],
    llm_client: LLMClient,
) -> ClaimVerification:
    """
    Verify a single DiagnosisClaim against the applicable evidence bundle.

    Args:
        claim:              The claim to verify.
        normalized:         Normalized incident from M2.
        applicable_results: M4's applicable evidence (only these may be cited).
        applicable_id_set:  Pre-built set of applicable chunk IDs (for O(1) lookup).
        llm_client:         LLM verifier client (independent of diagnosis generator).

    Returns:
        ClaimVerification with full detail of all four checks and final verdict.
    """
    # ------------------------------------------------------------------
    # Step 1: Deterministic citation validity — no LLM.
    # ------------------------------------------------------------------
    citation_validity, invalid_ids = check_citation_validity(claim, applicable_id_set)

    if citation_validity in (CitationValidity.INVALID, CitationValidity.EMPTY):
        # Short-circuit: if citations are bad, skip the LLM call entirely.
        if citation_validity == CitationValidity.EMPTY:
            reason = (
                f"Claim '{claim.claim_id}' has no evidence citations. "
                "An unsupported claim cannot be verified."
            )
        else:
            reason = (
                f"Claim '{claim.claim_id}' cites evidence IDs not present in the "
                f"applicable bundle: {invalid_ids}. These may be hallucinated IDs or "
                "references to evidence excluded by M4 applicability filtering."
            )

        logger.debug(
            "Claim %s: citation %s — short-circuit REJECTED. Invalid IDs: %s",
            claim.claim_id,
            citation_validity.value,
            invalid_ids,
        )
        return ClaimVerification(
            claim_id=claim.claim_id,
            verdict=VerificationVerdict.REJECTED,
            citation_validity=citation_validity,
            citation_correct=False,
            sufficient=False,
            contradicted=False,
            reason=reason,
            supporting_evidence_ids=[],
            contradicting_evidence_ids=[],
        )

    # ------------------------------------------------------------------
    # Step 2: Semantic verification via independent LLM call.
    # ------------------------------------------------------------------
    logger.debug(
        "Claim %s: citation VALID — proceeding to semantic verification (%d applicable chunks).",
        claim.claim_id,
        len(applicable_results),
    )

    prompt = build_verification_prompt(
        normalized=normalized,
        claim=claim,
        applicable_results=applicable_results,
    )

    try:
        response = llm_client.generate(prompt)
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(
            f"Unexpected error during semantic verification of claim {claim.claim_id}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not response.success or not response.text:
        raise LLMError(
            f"Verifier LLM returned empty/unsuccessful response for claim {claim.claim_id}."
        )

    # Parse and validate the verifier response.
    parsed = _parse_verifier_response(response.text)

    citation_correct: bool = parsed["citation_correct"]
    sufficient: bool = parsed["sufficient"]
    contradicted: bool = parsed["contradicted"]
    supporting_ids: list[str] = parsed["supporting_evidence_ids"]
    contradicting_ids: list[str] = parsed["contradicting_evidence_ids"]
    reason: str = parsed["reason"]

    # ------------------------------------------------------------------
    # Step 3: Compute verdict.
    # VERIFIED only if all four checks pass.
    # ------------------------------------------------------------------
    verdict = (
        VerificationVerdict.VERIFIED
        if (citation_correct and sufficient and not contradicted)
        else VerificationVerdict.REJECTED
    )

    logger.debug(
        "Claim %s: citation_correct=%s, sufficient=%s, contradicted=%s → %s",
        claim.claim_id,
        citation_correct,
        sufficient,
        contradicted,
        verdict.value,
    )

    return ClaimVerification(
        claim_id=claim.claim_id,
        verdict=verdict,
        citation_validity=citation_validity,
        citation_correct=citation_correct,
        sufficient=sufficient,
        contradicted=contradicted,
        reason=reason,
        supporting_evidence_ids=supporting_ids,
        contradicting_evidence_ids=contradicting_ids,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def verify_diagnosis(
    *,
    diagnosis: DiagnosisResult,
    applicable_results: list[RetrievalResult],
    normalized: NormalizedIncident,
    llm_client: LLMClient,
) -> DiagnosisVerification:
    """
    Verify all claims in a DiagnosisResult against the applicable evidence bundle.

    This is the main M5 entry point. It verifies each claim independently.

    Args:
        diagnosis:          The M4 DiagnosisResult containing claims to verify.
        applicable_results: M4's applicable evidence bundle.
                            MUST contain only APPLICABLE chunks (M4's filter_applicable
                            output). NOT_APPLICABLE chunks must never be passed here.
        normalized:         Normalized incident from M2.
        llm_client:         An LLMClient implementation. Must be independent from
                            the diagnosis generator (a separate instance or the same
                            type but called independently).

    Returns:
        DiagnosisVerification with per-claim ClaimVerification results.

    Raises:
        LLMError:             If the verifier LLM fails during semantic verification.
        SchemaValidationError: If the verifier LLM returns unparseable output.

    Notes:
        - Citation validity is checked deterministically before any LLM call.
        - Claims with invalid/empty citations are rejected without LLM overhead.
        - Claims are verified independently — one claim's failure does not affect others.
        - Does NOT implement retry (M6 responsibility).
        - Does NOT determine final outcome (M6 responsibility).
    """
    applicable_id_set = build_applicable_id_set(applicable_results)

    logger.info(
        "Starting M5 verification: %d claims, %d applicable chunks.",
        len(diagnosis.claims),
        len(applicable_results),
    )

    claim_verifications: list[ClaimVerification] = []

    for claim in diagnosis.claims:
        cv = _verify_claim(
            claim=claim,
            normalized=normalized,
            applicable_results=applicable_results,
            applicable_id_set=applicable_id_set,
            llm_client=llm_client,
        )
        claim_verifications.append(cv)

    verified_count = sum(
        1 for cv in claim_verifications if cv.verdict == VerificationVerdict.VERIFIED
    )
    logger.info(
        "M5 verification complete: %d/%d claims verified.",
        verified_count,
        len(claim_verifications),
    )

    return DiagnosisVerification(claim_verifications=claim_verifications)
