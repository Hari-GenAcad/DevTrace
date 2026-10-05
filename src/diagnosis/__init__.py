"""
DevTrace — Module 4: Diagnosis layer public API.

Exposes:
    NeedsInfoResult      — structured NEEDS_INFO response
    needs_info_check     — deterministic gate function
    DiagnosisGenerator   — Gemini-backed structured diagnosis generator
    parse_diagnosis      — parse raw LLM JSON text into DiagnosisResult
"""

from src.diagnosis.generator import DiagnosisGenerator, parse_diagnosis
from src.diagnosis.needs_info import NeedsInfoResult, needs_info_check

__all__ = [
    "NeedsInfoResult",
    "needs_info_check",
    "DiagnosisGenerator",
    "parse_diagnosis",
]
