"""
src/models package — re-exports for convenient imports.
"""

from src.models.contracts import (
    ApplicabilityResult,
    DiagnosisClaim,
    DiagnosisResult,
    Document,
    EvidenceChunk,
    FinalAnswer,
    RetrievalResult,
    Trace,
    TroubleshootingIncident,
    VerificationVerdict,
)
from src.models.enums import (
    AnswerCompleteness,
    ClaimRole,
    ContradictionDecision,
    RetrievalSource,
    SupportDecision,
    SystemOutcome,
    VerificationStatus,
)

__all__ = [
    # contracts
    "TroubleshootingIncident",
    "Document",
    "EvidenceChunk",
    "RetrievalResult",
    "ApplicabilityResult",
    "DiagnosisClaim",
    "DiagnosisResult",
    "VerificationVerdict",
    "FinalAnswer",
    "Trace",
    # enums
    "SystemOutcome",
    "AnswerCompleteness",
    "ClaimRole",
    "RetrievalSource",
    "VerificationStatus",
    "SupportDecision",
    "ContradictionDecision",
]
