"""
DevTrace — Module 4: Baseline diagnosis generator.

Responsibility
--------------
Call the LLM (via the M1 abstraction) with a diagnosis prompt and parse
the structured JSON response into a DiagnosisResult.

Key design points
-----------------
- Uses only the M1 LLMClient abstraction — never calls Gemini SDK directly.
- Receives only APPLICABLE evidence (NOT_APPLICABLE evidence is excluded
  by the M4 pipeline before calling this generator).
- Parses and validates the LLM response against the M1 DiagnosisResult schema.
- Raises typed errors (LLMError, SchemaValidationError) for clean failure paths.
- Does NOT verify whether citations are actually supported by evidence.
  That is M5's job.

What this module does NOT do
------------------------------
- No evidence verification
- No retry logic (M6)
- No citation correctness checking (M5)
- No contradiction detection (M5)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from src.errors import LLMError, SchemaValidationError
from src.llm.base import LLMClient
from src.models.contracts import DiagnosisClaim, DiagnosisResult, RetrievalResult
from src.models.enums import ClaimRole
from src.normalization.normalizer import NormalizedIncident
from src.diagnosis.prompts import build_diagnosis_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_claim_role(role_str: str) -> ClaimRole:
    """Parse a role string into a ClaimRole enum, with helpful error."""
    try:
        return ClaimRole(role_str.lower().strip())
    except ValueError:
        valid = [r.value for r in ClaimRole]
        raise SchemaValidationError(
            f"Invalid claim role {role_str!r}. Must be one of: {valid}"
        )


def parse_diagnosis(raw_text: str) -> DiagnosisResult:
    """
    Parse raw LLM JSON output into a validated DiagnosisResult.

    Args:
        raw_text: The raw text returned by the LLM client.

    Returns:
        A validated DiagnosisResult.

    Raises:
        SchemaValidationError: If the text is not parseable JSON, missing
                               required keys, or fails DiagnosisResult validation.
    """
    # Strip leading/trailing whitespace and any accidental markdown fences.
    text = raw_text.strip()
    if text.startswith("```"):
        # Strip first and last fence lines.
        lines = text.splitlines()
        # Remove first line (```json or ```) and last line (```)
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Parse JSON.
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(
            f"LLM response is not valid JSON: {exc}\n"
            f"Raw response (first 500 chars): {raw_text[:500]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise SchemaValidationError(
            f"LLM response JSON must be an object, got {type(data).__name__}."
        )

    if "claims" not in data:
        raise SchemaValidationError(
            "LLM response JSON is missing the required 'claims' key."
        )

    claims_data = data["claims"]
    if not isinstance(claims_data, list) or not claims_data:
        raise SchemaValidationError(
            "'claims' must be a non-empty list."
        )

    # Parse each claim.
    parsed_claims: list[DiagnosisClaim] = []
    for i, raw_claim in enumerate(claims_data):
        if not isinstance(raw_claim, dict):
            raise SchemaValidationError(
                f"Claim at index {i} is not a JSON object."
            )
        role_str = raw_claim.get("role", "")
        if not role_str:
            raise SchemaValidationError(
                f"Claim at index {i} is missing 'role'."
            )
        text_str = raw_claim.get("text", "")
        if not text_str or not text_str.strip():
            raise SchemaValidationError(
                f"Claim at index {i} is missing or has empty 'text'."
            )
        evidence_ids = raw_claim.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            raise SchemaValidationError(
                f"Claim at index {i}: 'evidence_ids' must be a list."
            )

        role = _parse_claim_role(role_str)
        parsed_claims.append(
            DiagnosisClaim(
                role=role,
                text=text_str.strip(),
                evidence_ids=[str(eid) for eid in evidence_ids],
            )
        )

    # Validate via DiagnosisResult (enforces exactly-one root_cause rule).
    try:
        result = DiagnosisResult(claims=parsed_claims)
    except ValidationError as exc:
        raise SchemaValidationError(
            f"DiagnosisResult schema validation failed: {exc}"
        ) from exc

    return result


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class DiagnosisGenerator:
    """
    Baseline Gemini-backed diagnosis generator.

    Accepts only applicable evidence (pre-filtered by M4 applicability layer).
    Returns a structured DiagnosisResult.

    Does NOT verify whether cited evidence actually supports each claim.
    That is M5's responsibility.

    Usage:
        generator = DiagnosisGenerator(llm_client)
        result = generator.generate(normalized_incident, applicable_results)
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """
        Args:
            llm_client: An LLMClient implementation (GeminiClient in production,
                        FakeLLMClient in tests).
        """
        self._client = llm_client

    def generate(
        self,
        normalized: NormalizedIncident,
        applicable_results: list[RetrievalResult],
    ) -> DiagnosisResult:
        """
        Generate a structured diagnosis from applicable evidence.

        Args:
            normalized:         Normalized incident from M2.
            applicable_results: Evidence that passed M4 applicability filtering.
                                Must NOT include NOT_APPLICABLE evidence.

        Returns:
            DiagnosisResult with structured claims.

        Raises:
            LLMError:             If the LLM provider returns an error.
            SchemaValidationError: If the LLM response cannot be parsed.
        """
        incident = normalized.incident
        signals = normalized.signals

        # Build the prompt.
        prompt = build_diagnosis_prompt(
            incident_description=incident.description,
            current_version=incident.current_version,
            previous_version=incident.previous_version,
            error_codes=signals.error_codes,
            applicable_results=applicable_results,
        )

        logger.debug(
            "Sending diagnosis prompt (%d chars, %d applicable chunks)",
            len(prompt),
            len(applicable_results),
        )

        # Call the LLM (errors propagate up as LLMError / SchemaValidationError).
        try:
            response = self._client.generate(prompt)
        except LLMError:
            raise
        except Exception as exc:
            # Wrap unexpected errors in LLMError so callers don't need to
            # import provider-specific exceptions.
            raise LLMError(
                f"Unexpected error calling LLM: {type(exc).__name__}: {exc}"
            ) from exc

        if not response.success or not response.text:
            raise LLMError(
                "LLM returned an empty or unsuccessful response."
            )

        logger.debug("LLM response received (%d chars)", len(response.text))

        # Parse and validate the structured response.
        return parse_diagnosis(response.text)
