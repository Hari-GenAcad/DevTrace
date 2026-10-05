"""
DevTrace — Module 3: Retrieval configuration.

Extends the M1 DevTraceConfig cleanly rather than creating a parallel
configuration object.  All retrieval parameters live here as a dataclass
so they can be constructed in tests without touching environment variables.

Scoring formula (documented here so it never drifts from the implementation):

    raw_dense_score  ∈ [0, 1]   (cosine similarity from Chroma)
    raw_bm25_score   ∈ [0, ∞)   (raw BM25 score; normalised per-query below)

    norm_dense  = raw_dense_score                              (already in [0,1])
    norm_bm25   = bm25_score / max_bm25_in_batch              (normalise per query)

    signal_boost = sum of applicable boost values from RetrievalBoosts
                   (capped at max_signal_boost)

    fused = dense_weight * norm_dense
          + bm25_weight  * norm_bm25
          + signal_boost

    final_score = min(fused, 1.0)   (clamp to [0, 1] contract)

Weights should sum to 1.0 for intuitive interpretation, but this is not
enforced — evaluations in M7 may find non-unit weights useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Boost configuration
# ---------------------------------------------------------------------------

@dataclass
class RetrievalBoosts:
    """
    Additive signal boost values.

    Each boost is added to the fused score when the retrieved chunk's
    content or metadata matches the corresponding signal from the incident.
    These are intentionally modest — they tilt the ranking, not dominate it.
    """

    error_code_match: float = 0.20
    """Chunk content/metadata contains an exact incident error code."""

    technical_term_match: float = 0.05
    """Chunk content contains a known technical term from the incident."""

    version_match: float = 0.08
    """Chunk's applies_to range overlaps the incident's current_version."""

    max_total: float = 0.30
    """Signal boosts are capped at this value (prevents over-boosting)."""


# ---------------------------------------------------------------------------
# Main retrieval configuration
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    """
    All retrieval parameters in one place.

    Instances are passed explicitly to HybridRetriever so the same retriever
    class can be used with different configurations in tests.
    """

    # ---- Embedding model ----
    embedding_model: str = "all-MiniLM-L6-v2"
    """
    Sentence-transformers model name.  Small and fast; produces 384-dim vectors.
    Configurable so M7 can experiment with larger models without code changes.
    """

    # ---- Chroma ----
    chroma_persist_dir: Path = field(
        default_factory=lambda: Path("data/chroma_index")
    )
    """Directory where Chroma persists its on-disk index."""

    chroma_collection_name: str = "devcore_chunks"
    """Name of the Chroma collection that stores DevCore evidence chunks."""

    # ---- Top-K parameters ----
    dense_top_k: int = 10
    """How many results to retrieve from dense (Chroma) search."""

    bm25_top_k: int = 10
    """How many results to retrieve from BM25 sparse search."""

    final_top_k: int = 8
    """Final ranked list length after fusion and deduplication."""

    # ---- Fusion weights ----
    dense_weight: float = 0.6
    """Weight applied to the normalised dense score during fusion."""

    bm25_weight: float = 0.4
    """Weight applied to the normalised BM25 score during fusion."""

    # ---- Signal boosts ----
    boosts: RetrievalBoosts = field(default_factory=RetrievalBoosts)
    """Per-signal boost configuration."""
