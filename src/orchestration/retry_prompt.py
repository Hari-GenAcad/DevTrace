"""
DevTrace — Module 6: Retry diagnosis prompt builder.

Responsibility
--------------
Build a targeted retry prompt that incorporates:
  - The original incident information.
  - The applicable evidence bundle (UNCHANGED — no new evidence).
  - The first diagnosis claims.
  - Verification failure feedback (why root_cause was rejected).

This is deliberately separate from M4's diagnosis prompt so that:
  1. The retry receives explicit feedback about what failed.
  2. The retry instruction is unambiguous about the goal.
  3. The separation is visible for auditing and M7 evaluation.

What the retry prompt does NOT do
----------------------------------
- It does NOT inject new evidence beyond the applicable bundle.
- It does NOT relax version applicability constraints.
- It does NOT hint at a "correct" answer.
- It expects the same DiagnosisResult JSON format as M4.

Design
------
The retry prompt is a superset of the M4 diagnosis prompt:
  [incident context] + [applicable evidence] + [first diagnosis] +
  [verification feedback] + [revised output instruction]
"""

from __future__ import annotations

from src.models.contracts import DiagnosisClaim, DiagnosisResult, RetrievalResult
from src.verification.models import ClaimVerification, DiagnosisVerification, VerificationVerdict


# ---------------------------------------------------------------------------
# Evidence block (reuse pattern from M4 prompts.py)
# ---------------------------------------------------------------------------

def _build_evidence_block(applicable_results: list[RetrievalResult]) -> str:
    """Format applicable evidence chunks for the retry prompt."""
    if not applicable_results:
        return "(No applicable evidence was found.)"

    lines: list[str] = []
    for rr in applicable_results:
        chunk_id = rr.chunk_id
        content: str = rr.metadata.get("content", "(content not available)")
        topic: str = rr.metadata.get("topic", "")
        applies_to: str = rr.metadata.get("applies_to", "*")

        lines.append(f"--- EVIDENCE CHUNK: {chunk_id} ---")
        if topic:
            lines.append(f"Topic: {topic}")
        lines.append(f"Version range: {applies_to}")
        lines.append(f"Content:\n{content}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verification feedback block
# ---------------------------------------------------------------------------

def _build_verification_feedback(
    diagnosis: DiagnosisResult,
    verification: DiagnosisVerification,
) -> str:
    """
    Build a concise summary of what failed in the first verification pass.

    This is injected into the retry prompt so the model understands exactly
    what went wrong and what to fix.
    """
    # Map claim_id → DiagnosisClaim for lookup
    claim_map: dict[str, DiagnosisClaim] = {c.claim_id: c for c in diagnosis.claims}

    lines: list[str] = ["FIRST DIAGNOSIS VERIFICATION RESULTS:"]
    lines.append("")

    for cv in verification.claim_verifications:
        claim = claim_map.get(cv.claim_id)
        claim_role = claim.role.value if claim else "unknown"
        claim_text = claim.text[:120] if claim else "(unknown claim)"
        verdict_label = cv.verdict.value

        lines.append(f"  Claim [{claim_role}]: \"{claim_text}\"")
        lines.append(f"  Verdict: {verdict_label}")
        if cv.verdict == VerificationVerdict.REJECTED and cv.reason:
            lines.append(f"  Rejection reason: {cv.reason}")
        lines.append("")

    lines.append(
        "RETRY INSTRUCTION: The root cause claim FAILED verification. "
        "Re-examine the applicable evidence below and produce a revised diagnosis. "
        "Focus especially on what the evidence directly and explicitly states "
        "about the root cause of the incident."
    )
    lines.append(
        "Do NOT invent facts or cite chunk IDs not listed in the evidence. "
        "If the evidence is insufficient, your root_cause claim must honestly state this "
        "and cite whatever relevant evidence you found."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output schema description
# ---------------------------------------------------------------------------

_SCHEMA_DESCRIPTION = """\
Return a JSON object with the following structure:

{
  "claims": [
    {
      "role": "<root_cause | fix | explanation>",
      "text": "<claim text>",
      "evidence_ids": ["<chunk_id_1>", "<chunk_id_2>"]
    }
  ]
}

Rules:
- "claims" must be a non-empty list.
- Exactly one claim must have role "root_cause".
- Additional claims with role "fix" or "explanation" are encouraged when supported.
- "text" must be a non-empty string.
- "evidence_ids" must contain chunk IDs from the provided evidence only.
  Do NOT cite chunk IDs that were not listed in the evidence below.
- Do NOT add extra keys outside the schema.
- Do NOT wrap the JSON in markdown code fences.
- Return raw JSON only.
"""


# ---------------------------------------------------------------------------
# Main retry prompt builder
# ---------------------------------------------------------------------------

def build_retry_prompt(
    *,
    incident_description: str,
    current_version: str | None,
    previous_version: str | None,
    error_codes: list[str],
    applicable_results: list[RetrievalResult],
    first_diagnosis: DiagnosisResult,
    first_verification: DiagnosisVerification,
) -> str:
    """
    Build the targeted retry diagnosis prompt.

    This prompt is used when the initial diagnosis's root_cause claim failed
    M5 verification. It gives the model:
      1. The same incident context.
      2. The SAME applicable evidence (no new evidence allowed).
      3. The first diagnosis claims (for context).
      4. A clear summary of what failed verification and why.
      5. An explicit instruction to revise the root_cause specifically.

    Args:
        incident_description:  Free-text incident description.
        current_version:       Incident's current SDK/API version.
        previous_version:      Previous version (may be None).
        error_codes:           Extracted error codes.
        applicable_results:    SAME applicable evidence as first attempt.
        first_diagnosis:       The first DiagnosisResult (pre-retry).
        first_verification:    The first DiagnosisVerification (failure context).

    Returns:
        A complete prompt string ready to send to the LLM client.
    """
    # Build incident context block.
    incident_lines: list[str] = [
        f"Description: {incident_description}",
    ]
    if current_version:
        incident_lines.append(f"Current version: {current_version}")
    if previous_version:
        incident_lines.append(f"Previous version: {previous_version}")
    if error_codes:
        incident_lines.append(f"Error codes: {', '.join(error_codes)}")
    incident_block = "\n".join(incident_lines)

    evidence_block = _build_evidence_block(applicable_results)
    feedback_block = _build_verification_feedback(first_diagnosis, first_verification)

    prompt = f"""\
You are a senior developer support engineer diagnosing a software troubleshooting incident.

A FIRST DIAGNOSIS ATTEMPT was made but its root cause claim FAILED evidence verification.
Your task is to produce a REVISED, IMPROVED diagnosis based ONLY on the applicable evidence.

CRITICAL INSTRUCTIONS:
1. Use ONLY the applicable evidence provided below. Do not use any knowledge outside these documents.
2. Do not invent facts, error codes, API names, or version numbers not present in the evidence.
3. Every claim you make MUST cite the chunk ID(s) from the evidence that support it.
4. Distinguish clearly between root cause (why the error occurs) and fix (what the developer must do).
5. Do NOT hallucinate chunk IDs. Only cite IDs that appear in the evidence block below.
6. Focus on what the evidence EXPLICITLY says about the root cause of this incident.
7. If the evidence is insufficient to identify a root cause, state this honestly but cite evidence.

INCIDENT:
{incident_block}

{feedback_block}

APPLICABLE EVIDENCE (SAME AS FIRST ATTEMPT — no new evidence):
{evidence_block}

OUTPUT FORMAT:
{_SCHEMA_DESCRIPTION}
"""
    return prompt
