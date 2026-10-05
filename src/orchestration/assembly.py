"""
DevTrace — Module 6: Final answer assembly.

Responsibility
--------------
Given a set of verified claims (from initial or retry verification),
assemble the user-facing VerifiedAnswer.

Key rules (from spec)
---------------------
1. Only VERIFIED claims may appear in the final answer.
2. If root_cause is not verified → INSUFFICIENT_EVIDENCE (no answer text).
3. If root_cause + fix are verified → at minimum ANSWERED_PARTIAL.
4. If root_cause + fix + explanation are all verified → ANSWERED_FULL.
5. NEEDS_INFO and DEGRADED are handled before reaching this module.

Answer text is assembled in role order: root_cause → fix → explanation.
Rejected claims are silently excluded from the text; they remain visible
in the TroubleshootingResult for audit/evaluation purposes.

This module does NOT:
- Perform verification (M5's job).
- Determine whether to retry (orchestrator's job).
- Classify NEEDS_INFO or DEGRADED (orchestrator handles these before calling assembly).
"""

from __future__ import annotations

from src.models.contracts import DiagnosisClaim, DiagnosisResult
from src.models.enums import ClaimRole
from src.orchestration.models import FinalOutcome, VerifiedAnswer
from src.verification.models import DiagnosisVerification, VerificationVerdict


# ---------------------------------------------------------------------------
# Root-cause survival check
# ---------------------------------------------------------------------------

def root_cause_survived(
    diagnosis: DiagnosisResult,
    verification: DiagnosisVerification,
) -> bool:
    """
    Return True only when the root_cause claim is VERIFIED.

    Presence of a root_cause claim is NOT sufficient — it must be VERIFIED.

    Args:
        diagnosis:     The DiagnosisResult (guaranteed to have exactly one ROOT_CAUSE).
        verification:  The DiagnosisVerification with per-claim verdicts.

    Returns:
        True if and only if the single root_cause claim was VERIFIED.
    """
    # Find the root_cause claim from the diagnosis
    root_cause_claim = next(
        (c for c in diagnosis.claims if c.role == ClaimRole.ROOT_CAUSE), None
    )
    if root_cause_claim is None:
        # DiagnosisResult invariant guarantees this cannot happen, but be safe.
        return False

    # Find the corresponding ClaimVerification
    root_verification = next(
        (cv for cv in verification.claim_verifications
         if cv.claim_id == root_cause_claim.claim_id),
        None,
    )
    if root_verification is None:
        return False

    return root_verification.verdict == VerificationVerdict.VERIFIED


# ---------------------------------------------------------------------------
# Retrieve root-cause rejection reason for retry
# ---------------------------------------------------------------------------

def get_root_cause_rejection_reason(
    diagnosis: DiagnosisResult,
    verification: DiagnosisVerification,
) -> str:
    """
    Return the rejection reason for the root_cause claim.

    Used to populate retry_reason in TroubleshootingResult for observability.

    Returns:
        The ClaimVerification.reason string, or a default message if missing.
    """
    root_cause_claim = next(
        (c for c in diagnosis.claims if c.role == ClaimRole.ROOT_CAUSE), None
    )
    if root_cause_claim is None:
        return "No root_cause claim found in diagnosis."

    root_cv = next(
        (cv for cv in verification.claim_verifications
         if cv.claim_id == root_cause_claim.claim_id),
        None,
    )
    if root_cv is None:
        return "Root cause claim not found in verification results."

    return root_cv.reason or "Root cause claim failed verification (no reason provided)."


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------

def classify_outcome(
    diagnosis: DiagnosisResult,
    verification: DiagnosisVerification,
) -> FinalOutcome:
    """
    Classify the final outcome given a diagnosis whose root_cause has survived.

    Precondition: root_cause_survived(diagnosis, verification) is True.

    Rules (from spec §3):
        CASE A — ANSWERED_FULL:    root_cause + fix + explanation all VERIFIED.
        CASE B — ANSWERED_PARTIAL: root_cause + fix VERIFIED, explanation rejected/missing.
        CASE C — ANSWERED_PARTIAL: root_cause VERIFIED, fix rejected/missing.

    Args:
        diagnosis:     DiagnosisResult (root_cause must have survived).
        verification:  DiagnosisVerification.

    Returns:
        ANSWERED_FULL or ANSWERED_PARTIAL.
    """
    # Build a lookup: claim_id → verdict
    verdict_map: dict[str, VerificationVerdict] = {
        cv.claim_id: cv.verdict
        for cv in verification.claim_verifications
    }

    # Build a lookup: role → claim_id (the first of each role)
    role_claim_id: dict[ClaimRole, str] = {}
    for claim in diagnosis.claims:
        if claim.role not in role_claim_id:
            role_claim_id[claim.role] = claim.claim_id

    def is_verified(role: ClaimRole) -> bool:
        claim_id = role_claim_id.get(role)
        if claim_id is None:
            return False
        return verdict_map.get(claim_id) == VerificationVerdict.VERIFIED

    root_ok = is_verified(ClaimRole.ROOT_CAUSE)      # guaranteed True (precondition)
    fix_ok = is_verified(ClaimRole.FIX)
    explanation_ok = is_verified(ClaimRole.EXPLANATION)

    if root_ok and fix_ok and explanation_ok:
        return FinalOutcome.ANSWERED_FULL

    # Root + fix verified but no explanation → PARTIAL
    # Root verified but fix rejected/missing → PARTIAL
    return FinalOutcome.ANSWERED_PARTIAL


# ---------------------------------------------------------------------------
# Answer text assembly
# ---------------------------------------------------------------------------

_ROLE_ORDER = [ClaimRole.ROOT_CAUSE, ClaimRole.FIX, ClaimRole.EXPLANATION]

_ROLE_LABELS = {
    ClaimRole.ROOT_CAUSE: "Root cause",
    ClaimRole.FIX: "Fix",
    ClaimRole.EXPLANATION: "Explanation",
}


def assemble_answer_text(
    diagnosis: DiagnosisResult,
    verification: DiagnosisVerification,
) -> str:
    """
    Assemble human-readable answer text from VERIFIED claims only.

    Rejected claims are excluded silently.
    Claims are presented in role order: root_cause → fix → explanation.

    Args:
        diagnosis:     The DiagnosisResult.
        verification:  The DiagnosisVerification (used to filter to VERIFIED only).

    Returns:
        A formatted string containing only verified claims.
        Empty string if no claims passed verification (should not happen if
        root_cause survived, but handled defensively).
    """
    verified_ids = set(verification.verified_claim_ids)

    # Collect verified claims in role order
    verified_by_role: dict[ClaimRole, list[DiagnosisClaim]] = {
        role: [] for role in _ROLE_ORDER
    }
    for claim in diagnosis.claims:
        if claim.claim_id in verified_ids and claim.role in verified_by_role:
            verified_by_role[claim.role].append(claim)

    sections: list[str] = []
    for role in _ROLE_ORDER:
        for claim in verified_by_role[role]:
            label = _ROLE_LABELS[role]
            sections.append(f"{label}: {claim.text}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Collect verified claims (for VerifiedAnswer.verified_claims)
# ---------------------------------------------------------------------------

def collect_verified_claims(
    diagnosis: DiagnosisResult,
    verification: DiagnosisVerification,
) -> list[DiagnosisClaim]:
    """
    Return only the DiagnosisClaim objects that passed verification.

    Returned in role order: root_cause → fix → explanation.

    Args:
        diagnosis:    The DiagnosisResult.
        verification: The DiagnosisVerification.

    Returns:
        List of VERIFIED DiagnosisClaims (may be empty).
    """
    verified_ids = set(verification.verified_claim_ids)

    verified_by_role: dict[ClaimRole, list[DiagnosisClaim]] = {
        role: [] for role in _ROLE_ORDER
    }
    for claim in diagnosis.claims:
        if claim.claim_id in verified_ids and claim.role in verified_by_role:
            verified_by_role[claim.role].append(claim)

    result: list[DiagnosisClaim] = []
    for role in _ROLE_ORDER:
        result.extend(verified_by_role[role])
    return result


# ---------------------------------------------------------------------------
# High-level answer builders
# ---------------------------------------------------------------------------

def build_answered_answer(
    diagnosis: DiagnosisResult,
    verification: DiagnosisVerification,
) -> VerifiedAnswer:
    """
    Build a VerifiedAnswer for cases where root_cause survived.

    Classifies ANSWERED_FULL vs ANSWERED_PARTIAL and assembles text.

    Precondition: root_cause_survived(diagnosis, verification) is True.
    """
    outcome = classify_outcome(diagnosis, verification)
    answer_text = assemble_answer_text(diagnosis, verification)
    verified_claims = collect_verified_claims(diagnosis, verification)

    return VerifiedAnswer(
        outcome=outcome,
        answer_text=answer_text if answer_text else None,
        verified_claims=verified_claims,
    )


def build_insufficient_answer() -> VerifiedAnswer:
    """
    Build a VerifiedAnswer for INSUFFICIENT_EVIDENCE.

    Used when root_cause failed verification in both the initial attempt
    and the retry (or retry was not possible).
    """
    return VerifiedAnswer(
        outcome=FinalOutcome.INSUFFICIENT_EVIDENCE,
        answer_text=None,
        verified_claims=[],
    )


def build_needs_info_answer(reason: str | None) -> VerifiedAnswer:
    """Build a VerifiedAnswer for NEEDS_INFO early exit."""
    return VerifiedAnswer(
        outcome=FinalOutcome.NEEDS_INFO,
        answer_text=None,
        verified_claims=[],
        needs_info_reason=reason,
    )


def build_degraded_answer(error_detail: str | None) -> VerifiedAnswer:
    """Build a VerifiedAnswer for DEGRADED system failure."""
    return VerifiedAnswer(
        outcome=FinalOutcome.DEGRADED,
        answer_text=None,
        verified_claims=[],
        error_detail=error_detail,
    )
