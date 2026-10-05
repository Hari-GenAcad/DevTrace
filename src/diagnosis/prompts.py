"""
DevTrace — Module 4: Diagnosis prompt templates.

Responsibility
--------------
Centralise all prompt construction for the diagnosis generator.

Prompts are plain strings — no template engines.
They are kept here so they can be reviewed and tuned independently of
the generator logic.

Design principles
-----------------
- The model is told to use ONLY the supplied evidence.
- The model is told NOT to invent facts.
- Every claim must cite evidence IDs.
- The output must be valid JSON conforming to the DiagnosisResult schema.
- The model is reminded that evidence grounding is mandatory.
- The model is reminded that citation verification is NOT its job here
  (M5 will do that) — but it must still cite accurately from what it can see.
"""

from __future__ import annotations

from src.models.contracts import RetrievalResult


# ---------------------------------------------------------------------------
# Schema description embedded in prompt
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
# Evidence block builder
# ---------------------------------------------------------------------------

def _build_evidence_block(applicable_results: list[RetrievalResult]) -> str:
    """
    Format applicable evidence chunks into the prompt.

    Each chunk is presented with its ID and content so the model can
    cite specific evidence.
    """
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
# Main prompt builder
# ---------------------------------------------------------------------------

def build_diagnosis_prompt(
    incident_description: str,
    current_version: str | None,
    previous_version: str | None,
    error_codes: list[str],
    applicable_results: list[RetrievalResult],
) -> str:
    """
    Build the full diagnosis prompt for the Gemini model.

    Args:
        incident_description: Free-text description of the incident.
        current_version:      Incident's current SDK/API version (may be None).
        previous_version:     Version before the change that triggered the issue (may be None).
        error_codes:          Extracted error codes from the incident.
        applicable_results:   Evidence chunks that passed M4 applicability filtering.

    Returns:
        A complete prompt string ready to send to the LLM client.
    """
    # Build incident context block.
    incident_block_lines: list[str] = [
        f"Description: {incident_description}",
    ]
    if current_version:
        incident_block_lines.append(f"Current version: {current_version}")
    if previous_version:
        incident_block_lines.append(f"Previous version: {previous_version}")
    if error_codes:
        incident_block_lines.append(f"Error codes: {', '.join(error_codes)}")

    incident_block = "\n".join(incident_block_lines)

    evidence_block = _build_evidence_block(applicable_results)

    prompt = f"""\
You are a senior developer support engineer diagnosing a software troubleshooting incident.

Your task is to produce a structured diagnosis of the incident based ONLY on the provided evidence.

CRITICAL INSTRUCTIONS:
1. Use ONLY the evidence provided below. Do not use any knowledge outside these documents.
2. Do not invent facts, error codes, API names, or version numbers not present in the evidence.
3. Every claim you make MUST cite the chunk ID(s) from the evidence that support it.
4. Distinguish clearly between root cause (why the error occurs) and fix (what the developer must do).
5. If the evidence is insufficient to identify a root cause, still produce a root_cause claim that
   honestly states the evidence is insufficient, citing whatever relevant evidence you found.
6. Do NOT hallucinate chunk IDs. Only cite IDs that appear in the evidence block below.
7. Prefer precise technical language. Avoid vague statements like "something may be wrong".

INCIDENT:
{incident_block}

APPLICABLE EVIDENCE:
{evidence_block}

OUTPUT FORMAT:
{_SCHEMA_DESCRIPTION}
"""
    return prompt
