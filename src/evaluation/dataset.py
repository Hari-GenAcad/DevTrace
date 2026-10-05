"""
Evaluation dataset contracts and loader for M2.

Defines the EvaluationCase Pydantic model and loads the evaluation dataset
from data/eval/eval_dataset.json.

These contracts are consumed by M7 evaluation logic.
M2 only defines and validates them — no metrics are computed here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from src.models.enums import AnswerCompleteness, SystemOutcome

_DEFAULT_EVAL_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "eval" / "eval_dataset.json"
)

# Valid case class identifiers.
VALID_CASE_CLASSES: frozenset[str] = frozenset({
    "straightforward",
    "multi_document",
    "version_conflict",
    "similar_error",
    "near_miss",
    "unsupported",
    "missing_info",
    "contradictory_evidence",
})


class EvaluationIncident(BaseModel):
    """Incident specification embedded in an evaluation case."""

    description: str = Field(..., min_length=1)
    current_version: str | None = None
    previous_version: str | None = None
    error_codes: list[str] = Field(default_factory=list)
    product: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class EvaluationCase(BaseModel):
    """
    A single evaluation case in the DevTrace evaluation dataset.

    Provides the ground truth needed for M7 metric computation:
      - expected_outcome + expected_completeness
      - gold_doc_ids (documents that legitimately support the answer)
      - forbidden_doc_ids (inapplicable/wrong-version docs that must not be cited)
      - case_class (for stratified analysis)
    """

    case_id: str = Field(..., description="Stable unique case identifier.")
    case_class: str = Field(..., description="Evaluation class category.")
    description: str = Field(..., description="Human-readable case description.")
    incident: EvaluationIncident = Field(..., description="The incident to troubleshoot.")
    expected_outcome: SystemOutcome = Field(...)
    expected_completeness: AnswerCompleteness | None = Field(
        default=None,
        description="Set only when expected_outcome is ANSWERED.",
    )
    gold_doc_ids: list[str] = Field(
        default_factory=list,
        description="Document IDs that legitimately support the expected diagnosis.",
    )
    forbidden_doc_ids: list[str] = Field(
        default_factory=list,
        description="Inapplicable/wrong-version documents that must not be cited.",
    )
    notes: str = Field(default="", description="Explanation of the case design.")

    @field_validator("case_class")
    @classmethod
    def _validate_case_class(cls, v: str) -> str:
        if v not in VALID_CASE_CLASSES:
            raise ValueError(
                f"Unknown case_class {v!r}. Valid values: {sorted(VALID_CASE_CLASSES)}"
            )
        return v

    @model_validator(mode="after")
    def _completeness_requires_answered(self) -> EvaluationCase:
        if (
            self.expected_completeness is not None
            and self.expected_outcome != SystemOutcome.ANSWERED
        ):
            raise ValueError(
                "expected_completeness may only be set when expected_outcome is ANSWERED."
            )
        return self

    model_config = {"frozen": True}


class EvaluationDataset(BaseModel):
    """The full evaluation dataset, validated as a collection."""

    cases: list[EvaluationCase]

    @model_validator(mode="after")
    def _unique_case_ids(self) -> EvaluationDataset:
        ids = [c.case_id for c in self.cases]
        if len(ids) != len(set(ids)):
            from collections import Counter  # noqa: PLC0415
            duplicates = [k for k, v in Counter(ids).items() if v > 1]
            raise ValueError(f"Duplicate case_ids in dataset: {duplicates}")
        return self

    model_config = {"frozen": True}


class EvalLoadError(Exception):
    """Raised when the evaluation dataset cannot be loaded."""


def load_eval_dataset(
    eval_path: Path | str | None = None,
    *,
    valid_doc_ids: set[str] | None = None,
) -> EvaluationDataset:
    """
    Load and validate the evaluation dataset from JSON.

    Args:
        eval_path: Path to eval_dataset.json. Defaults to data/eval/eval_dataset.json.
        valid_doc_ids: If provided, checks that all gold/forbidden doc IDs reference
                       documents that exist in the corpus. Raises EvalLoadError if not.

    Returns:
        Validated EvaluationDataset.

    Raises:
        EvalLoadError: If the file cannot be read, is malformed, or doc IDs are invalid.
    """
    path = Path(eval_path) if eval_path is not None else _DEFAULT_EVAL_PATH

    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise EvalLoadError(f"Evaluation dataset not found: {path}") from exc
    except OSError as exc:
        raise EvalLoadError(f"Cannot read evaluation dataset: {path}: {exc}") from exc

    try:
        raw_list = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise EvalLoadError(f"Evaluation dataset is not valid JSON: {exc}") from exc

    if not isinstance(raw_list, list):
        raise EvalLoadError("Evaluation dataset must be a JSON array.")

    try:
        from pydantic import ValidationError  # noqa: PLC0415
        dataset = EvaluationDataset(cases=[EvaluationCase(**item) for item in raw_list])
    except (TypeError, Exception) as exc:
        raise EvalLoadError(f"Evaluation dataset validation failed: {exc}") from exc

    # Optional corpus integrity check.
    if valid_doc_ids is not None:
        for case in dataset.cases:
            for doc_id in case.gold_doc_ids + case.forbidden_doc_ids:
                if doc_id not in valid_doc_ids:
                    raise EvalLoadError(
                        f"Case {case.case_id!r} references unknown doc_id {doc_id!r}. "
                        f"Ensure the corpus contains this document."
                    )

    return dataset
