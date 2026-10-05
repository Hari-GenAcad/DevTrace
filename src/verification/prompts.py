"""
DevTrace — Module 5: Verification prompt builder.

Responsibility
--------------
Build the LLM prompt that asks an independent verifier to evaluate whether
a diagnosis claim is supported by the applicable evidence.

Design constraints
------------------
- The verifier must reason ONLY from the supplied evidence.
- It must NOT use pretrained knowledge as evidence.
- It must NOT reference any evidence excluded by M4 applicability.
- It must return structured JSON matching the verifier output schema.

Prompt structure
----------------
1. System context: role and strict constraints.
2. Incident summary: what problem is being diagnosed.
3. Claim under review: the specific claim text and cited IDs.
4. Evidence block: applicable chunks the verifier may use.
5. Output specification: strict JSON schema.

The verifier answers four distinct questions:
    Q1. Does the cited evidence DIRECTLY support this claim? (citation_correct)
    Q2. Is the evidence SUFFICIENT to justify the claim? (sufficient)
    Q3. Does any applicable evidence CONTRADICT the claim? (contradicted)
    Q4. Which evidence IDs support the claim? (supporting_evidence_ids)
    Q5. Which evidence IDs contradict the claim? (contradicting_evidence_ids)
    Q6. Short reason for the verdict. (reason)
"""

from __future__ import annotations

from src.models.contracts import DiagnosisClaim, RetrievalResult
from src.normalization.normalizer import NormalizedIncident


# ---------------------------------------------------------------------------
# Evidence block formatting
# ---------------------------------------------------------------------------

def _format_evidence_block(applicable_results: list[RetrievalResult]) -> str:
    """
    Format applicable evidence chunks into a readable evidence block for the prompt.

    Each chunk is prefixed with its chunk_id so the verifier can reference
    specific IDs in its output.
    """
    if not applicable_results:
        return "  [No applicable evidence provided.]\n"

    lines: list[str] = []
    for rr in applicable_results:
        content = rr.metadata.get("content", "[No content available]")
        applies_to = rr.metadata.get("applies_to", "*")
        lines.append(
            f"  [CHUNK {rr.chunk_id} | doc={rr.doc_id} | applies_to={applies_to}]\n"
            f"  {content}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_verification_prompt(
    *,
    normalized: NormalizedIncident,
    claim: DiagnosisClaim,
    applicable_results: list[RetrievalResult],
) -> str:
    """
    Build the structured verification prompt for a single claim.

    Args:
        normalized:         Normalized incident context (M2 output).
        claim:              The DiagnosisClaim to verify.
        applicable_results: Applicable evidence from M4 (only these may be used).

    Returns:
        A complete prompt string to send to the LLM verifier.
    """
    incident = normalized.incident
    signals = normalized.signals

    # Incident context summary
    version_line = (
        f"Current version: {incident.current_version}"
        if incident.current_version
        else "Current version: not specified"
    )
    prev_version_line = (
        f"Previous version: {incident.previous_version}"
        if incident.previous_version
        else ""
    )
    error_codes_line = (
        f"Error codes: {', '.join(signals.error_codes)}"
        if signals.error_codes
        else ""
    )

    incident_summary_parts = [
        f"Description: {incident.description}",
        version_line,
    ]
    if prev_version_line:
        incident_summary_parts.append(prev_version_line)
    if error_codes_line:
        incident_summary_parts.append(error_codes_line)
    incident_summary = "\n".join(f"  {p}" for p in incident_summary_parts)

    # Claim details
    cited_ids_str = (
        ", ".join(claim.evidence_ids) if claim.evidence_ids else "[none cited]"
    )

    # Evidence block
    evidence_block = _format_evidence_block(applicable_results)

    # Available evidence IDs (so verifier knows what's valid)
    available_ids = [rr.chunk_id for rr in applicable_results]
    available_ids_str = ", ".join(available_ids) if available_ids else "[none]"

    prompt = f"""You are an independent evidence verifier for a software troubleshooting system.

Your task is to evaluate whether a specific diagnosis claim is supported by the provided applicable evidence.

STRICT RULES:
- You must reason ONLY from the evidence provided below. Do NOT use your training knowledge as evidence.
- You must NOT reference any documents not listed in the evidence block.
- You must answer all questions based solely on what the evidence says.
- Return ONLY a valid JSON object — no markdown fences, no prose.

==================================================
INCIDENT CONTEXT
==================================================
{incident_summary}

==================================================
CLAIM UNDER REVIEW (role: {claim.role.value})
==================================================
  Claim text: "{claim.text}"
  Cited evidence IDs: {cited_ids_str}

==================================================
AVAILABLE APPLICABLE EVIDENCE
==================================================
Available IDs: {available_ids_str}

{evidence_block}
==================================================
VERIFICATION QUESTIONS
==================================================

Answer each of the following questions based ONLY on the evidence above:

Q1. Does the cited evidence (IDs: {cited_ids_str}) DIRECTLY support the claim text?
    - "true" means the cited evidence explicitly states or clearly implies the claim.
    - "false" means the cited evidence is absent, unrelated, or only tangentially related.

Q2. Is the available applicable evidence SUFFICIENT to justify the claim?
    - "true" means the claim is well-grounded in the evidence.
    - "false" means the evidence is too weak, vague, or indirect to establish the claim.

Q3. Does any of the applicable evidence CONTRADICT the claim?
    - "true" means some evidence explicitly conflicts with the claim.
    - "false" means no conflict exists.

Q4. Which evidence IDs (from the available list: {available_ids_str}) SUPPORT the claim?
    List only IDs from the available list above. Empty list if none.

Q5. Which evidence IDs (from the available list: {available_ids_str}) CONTRADICT the claim?
    List only IDs from the available list above. Empty list if none.

Q6. Provide a concise reason (1–2 sentences) explaining your overall verdict.

==================================================
REQUIRED JSON OUTPUT FORMAT
==================================================

Return EXACTLY this JSON structure and nothing else:

{{
  "citation_correct": <true|false>,
  "sufficient": <true|false>,
  "contradicted": <true|false>,
  "supporting_evidence_ids": [<list of chunk ID strings>],
  "contradicting_evidence_ids": [<list of chunk ID strings>],
  "reason": "<concise explanation>"
}}
"""
    return prompt
