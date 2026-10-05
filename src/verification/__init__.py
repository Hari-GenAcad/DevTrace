"""
DevTrace — Module 5: Evidence Verification.

Public API
----------
from src.verification import verify_diagnosis, ClaimVerification, DiagnosisVerification

verify_diagnosis(
    diagnosis=diagnosis_result,
    applicable_results=applicable_chunks,
    normalized=normalized_incident,
    llm_client=llm_client,
) -> DiagnosisVerification
"""

from src.verification.models import (
    CitationValidity,
    ClaimVerification,
    DiagnosisVerification,
    VerificationVerdict,
)
from src.verification.verifier import verify_diagnosis

__all__ = [
    "CitationValidity",
    "ClaimVerification",
    "DiagnosisVerification",
    "VerificationVerdict",
    "verify_diagnosis",
]
