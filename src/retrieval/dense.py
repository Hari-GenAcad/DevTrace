"""
DevTrace — Module 3: Dense (semantic) retrieval via Chroma.

Responsibilities:
  - Load/hold the SentenceTransformer embedding model.
  - Build a query embedding from the normalised incident representation.
  - Query the Chroma collection and return DenseHit objects with raw scores.

DenseHit is an internal dataclass — it is NOT the final RetrievalResult.
Hybrid fusion (hybrid.py) converts DenseHits into RetrievalResult contracts.

Scoring note:
  Chroma uses cosine *distance* (lower = more similar).
  We convert to cosine *similarity*: score = 1.0 − distance.
  Scores are therefore in [0, 1] where 1.0 = identical.

IMPORTANT: No applicability filtering happens here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.retrieval.config import RetrievalConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal result type
# ---------------------------------------------------------------------------

@dataclass
class DenseHit:
    """Raw result from Chroma dense retrieval (pre-fusion)."""

    chunk_id: str
    doc_id: str
    content: str
    applies_to: str
    topic: str
    extra_metadata: dict[str, Any]
    raw_score: float   # cosine similarity in [0, 1]


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def build_dense_query(
    description: str,
    *,
    error_codes: list[str] | None = None,
    technical_terms: list[str] | None = None,
    current_version: str | None = None,
    previous_version: str | None = None,
    product: str | None = None,
) -> str:
    """
    Construct a single query string for dense embedding from the normalised
    incident and its signals.

    Design rationale:
    - Concatenation is simple, transparent, and deterministic.
    - Error codes and technical terms are appended verbatim so the embedding
      captures their token-level semantics.
    - Version strings give the model a hint about the software context.
    - No LLM is used here.

    Args:
        description:      Normalised incident description.
        error_codes:      Error codes extracted by M2.
        technical_terms:  Technical terms extracted by M2.
        current_version:  Current SDK/API version from incident.
        previous_version: Previous version (upgrade context).
        product:          Product or component name.

    Returns:
        A single concatenated query string ready for embedding.
    """
    parts: list[str] = [description.strip()]

    if error_codes:
        parts.append("Error codes: " + " ".join(error_codes))

    if technical_terms:
        parts.append("Keywords: " + " ".join(technical_terms))

    if current_version:
        parts.append(f"SDK version {current_version}")

    if previous_version:
        parts.append(f"Upgrading from version {previous_version}")

    if product:
        parts.append(product)

    return ". ".join(parts)


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Wraps the SentenceTransformer model and a Chroma collection for
    semantic (dense) retrieval.

    Lifecycle:
        retriever = DenseRetriever(config)
        retriever.load()          # loads model + opens Chroma
        hits = retriever.query(query_text, top_k=10)
    """

    def __init__(self, config: RetrievalConfig) -> None:
        self._config = config
        self._model: Any = None          # SentenceTransformer
        self._collection: Any = None     # chromadb.Collection

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the embedding model and open the Chroma collection."""
        from sentence_transformers import SentenceTransformer
        from src.retrieval.index import get_or_create_collection

        if self._model is None:
            logger.info("Loading embedding model '%s' …", self._config.embedding_model)
            self._model = SentenceTransformer(self._config.embedding_model)

        _client, self._collection = get_or_create_collection(self._config)
        logger.debug("DenseRetriever ready (collection: %s).", self._config.chroma_collection_name)

    @property
    def model(self) -> Any:
        """Return the loaded SentenceTransformer model (for shared use in index.py)."""
        if self._model is None:
            raise RuntimeError("DenseRetriever.load() must be called first.")
        return self._model

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query_text: str, *, top_k: int | None = None) -> list[DenseHit]:
        """
        Embed the query and retrieve the closest evidence chunks from Chroma.

        Args:
            query_text: The query string (built by build_dense_query).
            top_k:      Override the configured dense_top_k.

        Returns:
            List of DenseHit objects sorted by descending similarity score.
        """
        if self._model is None or self._collection is None:
            raise RuntimeError("DenseRetriever.load() must be called before query().")

        k = top_k if top_k is not None else self._config.dense_top_k
        query_embedding = self._model.encode([query_text], show_progress_bar=False)

        results = self._collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[DenseHit] = []
        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            # Chroma cosine distance ∈ [0, 2]; similarity = 1 − (distance / 2)
            # For unit vectors (sentence transformers normalises by default):
            #   cosine_distance = 1 − cosine_similarity  → similarity = 1 − distance
            # We clamp to [0, 1] defensively.
            raw_score = max(0.0, min(1.0, 1.0 - distance))
            metadata = results["metadatas"][0][i]

            import json
            extra = {}
            try:
                extra = json.loads(metadata.get("extra_metadata", "{}"))
            except (json.JSONDecodeError, TypeError):
                pass

            hit = DenseHit(
                chunk_id=chunk_id,
                doc_id=metadata.get("doc_id", ""),
                content=results["documents"][0][i],
                applies_to=metadata.get("applies_to", "*"),
                topic=metadata.get("topic", ""),
                extra_metadata=extra,
                raw_score=raw_score,
            )
            hits.append(hit)

        # Sort descending by score (Chroma usually returns sorted, but be explicit).
        hits.sort(key=lambda h: (-h.raw_score, h.chunk_id))
        return hits
