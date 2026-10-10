"""
DevTrace — Module 7: Evaluation configuration.

Centralises all evaluation-level configuration so that:
  - No secrets are hardcoded.
  - The evaluation can be run from CLI args or environment variables.
  - All relevant settings are documented.

Default values represent the deterministic (FakeLLM) evaluation mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EVAL_DATASET = _REPO_ROOT / "data" / "eval" / "eval_dataset.json"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "eval_results"


@dataclass
class EvaluationConfig:
    """
    Configuration for a single M7 evaluation run.

    Attributes
    ----------
    evaluation_mode:
        'deterministic' — use FakeLLMClient with scripted responses.
        'live'          — use real Gemini API (requires GEMINI_API_KEY env var).

    dataset_path:
        Path to the evaluation dataset JSON file.

    output_dir:
        Directory where evaluation results and reports are written.

    baseline_b_threshold:
        Score threshold for Baseline B (0.0–1.0).
        Do NOT tune this on evaluation cases; set it here before running.

    retrieval_top_k:
        Number of candidates to retrieve.

    live_model_name:
        Gemini model to use in live mode (e.g. 'gemini-3.5-flash-lite').

    live_temperature:
        Temperature for live generation (lower = more deterministic).

    run_baseline_a:
        Whether to run Baseline A evaluation (set False to skip).

    run_baseline_b:
        Whether to run Baseline B evaluation.

    run_devtrace:
        Whether to run the full DevTrace evaluation.

    log_level:
        Python logging level string (e.g. 'INFO', 'DEBUG', 'WARNING').
    """

    evaluation_mode: str = "deterministic"
    dataset_path: Path = field(default_factory=lambda: _DEFAULT_EVAL_DATASET)
    output_dir: Path = field(default_factory=lambda: _DEFAULT_OUTPUT_DIR)
    baseline_b_threshold: float = 0.40
    retrieval_top_k: int = 10
    live_model_name: str = "gemini-3.5-flash-lite"
    live_temperature: float = 0.0
    run_baseline_a: bool = True
    run_baseline_b: bool = True
    run_devtrace: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.evaluation_mode not in ("deterministic", "live"):
            raise ValueError(
                f"evaluation_mode must be 'deterministic' or 'live', "
                f"got {self.evaluation_mode!r}."
            )
        if not 0.0 <= self.baseline_b_threshold <= 1.0:
            raise ValueError(
                f"baseline_b_threshold must be in [0.0, 1.0], "
                f"got {self.baseline_b_threshold}."
            )
        if self.evaluation_mode == "live":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "evaluation_mode='live' requires GEMINI_API_KEY environment variable."
                )

    def model_config_info(self) -> dict[str, object]:
        """Return a dict describing the model configuration for reproducibility."""
        return {
            "evaluation_mode": self.evaluation_mode,
            "live_model_name": self.live_model_name if self.evaluation_mode == "live" else "FakeLLM",
            "live_temperature": self.live_temperature if self.evaluation_mode == "live" else None,
            "baseline_b_threshold": self.baseline_b_threshold,
            "retrieval_top_k": self.retrieval_top_k,
        }


def config_from_env() -> EvaluationConfig:
    """
    Build an EvaluationConfig from environment variables.

    Environment variables:
        DEVTRACE_EVAL_MODE:        'deterministic' or 'live'
        DEVTRACE_EVAL_DATASET:     Path to eval dataset JSON
        DEVTRACE_EVAL_OUTPUT_DIR:  Output directory path
        DEVTRACE_EVAL_THRESHOLD:   Baseline B score threshold (float)
        DEVTRACE_EVAL_TOP_K:       Retrieval top-k (int)
        DEVTRACE_EVAL_MODEL:       Gemini model name (live mode only)
        DEVTRACE_EVAL_TEMPERATURE: Generation temperature (live mode, float)
        GEMINI_API_KEY:            Required for live mode
        DEVTRACE_LOG_LEVEL:        Logging level
    """
    mode = os.environ.get("DEVTRACE_EVAL_MODE", "deterministic")
    dataset_path_str = os.environ.get("DEVTRACE_EVAL_DATASET")
    output_dir_str = os.environ.get("DEVTRACE_EVAL_OUTPUT_DIR")
    threshold_str = os.environ.get("DEVTRACE_EVAL_THRESHOLD")
    top_k_str = os.environ.get("DEVTRACE_EVAL_TOP_K")
    model = os.environ.get("DEVTRACE_EVAL_MODEL", "gemini-3.5-flash-lite")
    temperature_str = os.environ.get("DEVTRACE_EVAL_TEMPERATURE")
    log_level = os.environ.get("DEVTRACE_LOG_LEVEL", "INFO")

    return EvaluationConfig(
        evaluation_mode=mode,
        dataset_path=Path(dataset_path_str) if dataset_path_str else _DEFAULT_EVAL_DATASET,
        output_dir=Path(output_dir_str) if output_dir_str else _DEFAULT_OUTPUT_DIR,
        baseline_b_threshold=float(threshold_str) if threshold_str else 0.40,
        retrieval_top_k=int(top_k_str) if top_k_str else 10,
        live_model_name=model,
        live_temperature=float(temperature_str) if temperature_str else 0.0,
        log_level=log_level,
    )
