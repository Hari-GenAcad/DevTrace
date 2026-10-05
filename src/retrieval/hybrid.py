"""
DevTrace — Module 3: Hybrid retrieval fusion, deduplication, and ranking.

This is the top-level M3 orchestrator.

Flow:
    1. Build dense query from normalised incident + signals.
    2. Retrieve dense candidates (DenseHit list) from Chroma.
    3. Retrieve BM25 candidates (BM25Hit list).
    4. Merge by chunk_id — preserve both channel scores.
    5. Compute signal boost per candidate.
    6. Fuse scores into a single value.
    7. Sort descending, stable tie-break on chunk_id.
    8. Slice to final_top_k.
    9. Return list[RichRetrievalResult] — an extended RetrievalResult.

RichRetrievalResult:
    M3 extends the M1 RetrievalResult contract with score components so
    the trace can show exactly why a chunk ranked where it did:

        Dense: 0.82
        BM25:  4.71  → normalised 0.61
        Error-code boost: +0.20
        Keyword boost:    +0.05
        Final:            0.79

    The M1 RetrievalResult.score field holds the final fused score.
    The extra fields are additive detail — M4 only needs RetrievalResult.

CRITICAL: No applicability filtering happens here.
    Both 2.x and 3.x chunks for the same topic may appear in the result.
    M4 will decide which are applicable.

CRITICAL: No reranker.
    Dense + BM25 + signal boosts is the complete M3 ranking system.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.models.contracts import EvidenceChunk, RetrievalResult
from src.models.enums import RetrievalSource
from src.normalization.normalizer import NormalizedIncident
from src.normalization.signals import ExtractedSignals
from src.retrieval.boosting import BoostResult, compute_boost
from src.retrieval.config import RetrievalConfig
from src.retrieval.dense import DenseHit, DenseRetriever, build_dense_query
from src.retrieval.sparse import BM25Hit, BM25Retriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extended result type
# ---------------------------------------------------------------------------

@dataclass
class RichRetrievalResult:
    """
    M3 retrieval result with full score transparency.

    The M1 contract RetrievalResult can be derived from this via to_contract().
    M4 consumes the contract form; tests and tracing use the rich form.
    """

    chunk_id: str
    doc_id: str
    content: str
    applies_to: str
    topic: str
    extra_metadata: dict[str, Any]

    # --- Score components ---
    dense_score: float | None       # normalised cosine similarity ∈ [0, 1]
    bm25_raw_score: float | None    # raw BM25 score
    bm25_norm_score: float | None   # normalised to [0, 1] per-query
    boost: BoostResult
    fused_score: float              # final fused score (clamped to [0, 1])

    # --- Retrieval source ---
    source: RetrievalSource

    def to_contract(self) -> RetrievalResult:
        """
        Convert to the M1 RetrievalResult contract.

        The contract score is the fused_score.
        The metadata dict includes all score components for traceability.
        """
        return RetrievalResult(
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            score=self.fused_score,
            source=self.source,
            metadata={
                "content": self.content,
                "applies_to": self.applies_to,
                "topic": self.topic,
                "dense_score": self.dense_score,
                "bm25_raw_score": self.bm25_raw_score,
                "bm25_norm_score": self.bm25_norm_score,
                "error_code_boost": self.boost.error_code_boost,
                "technical_term_boost": self.boost.technical_term_boost,
                "version_boost": self.boost.version_boost,
                "signal_boost_total": self.boost.total,
                **self.extra_metadata,
            },
        )


# ---------------------------------------------------------------------------
# Fusion helpers
# ---------------------------------------------------------------------------

def _normalise_bm25(hits: list[BM25Hit]) -> dict[str, float]:
    """
    Normalise BM25 scores to [0, 1] within this query batch.

    Strategy: linear normalisation by the maximum score in the batch.
    If all scores are 0, returns 0 for all.
    """
    if not hits:
        return {}
    max_score = max(h.raw_score for h in hits)
    if max_score <= 0.0:
        return {h.chunk_id: 0.0 for h in hits}
    return {h.chunk_id: h.raw_score / max_score for h in hits}


def _determine_source(
    in_dense: bool,
    in_bm25: bool,
) -> RetrievalSource:
    if in_dense and in_bm25:
        return RetrievalSource.HYBRID
    if in_dense:
        return RetrievalSource.SEMANTIC
    return RetrievalSource.KEYWORD


def _get_metadata_error_codes(extra: dict[str, Any]) -> list[str]:
    """Extract error_codes from the extra_metadata dict (may be a list)."""
    codes = extra.get("error_codes", [])
    if isinstance(codes, list):
        return [str(c) for c in codes]
    return []


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    M3 top-level retriever.  Combines Chroma dense + BM25 sparse retrieval
    with deterministic signal boosting and stable ranking.

    Usage:
        retriever = HybridRetriever(chunks, config)
        retriever.load()
        results = retriever.retrieve(normalized_incident)
    """

    def __init__(
        self,
        chunks: list[EvidenceChunk],
        config: RetrievalConfig | None = None,
    ) -> None:
        self._chunks = chunks
        self._config = config or RetrievalConfig()
        self._dense: DenseRetriever | None = None
        self._sparse: BM25Retriever | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the embedding model, open the Chroma collection, and build
        the BM25 index.

        This should be called once.  Subsequent calls are safe (idempotent).
        """
        if self._dense is None:
            self._dense = DenseRetriever(self._config)
            self._dense.load()

        if self._sparse is None:
            self._sparse = BM25Retriever(self._chunks, self._config)

    def build_vector_index(self) -> None:
        """
        (Re)build the Chroma vector index from the current chunk set.

        Call this the first time, or whenever the corpus changes.
        Subsequent calls are idempotent (Chroma uses upsert).
        """
        if self._dense is None:
            raise RuntimeError("Call load() before build_vector_index().")
        from src.retrieval.index import build_index
        build_index(self._chunks, self._config, embedder=self._dense.model)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        normalized: NormalizedIncident,
    ) -> list[RichRetrievalResult]:
        """
        Retrieve ranked evidence candidates for the given normalised incident.

        Args:
            normalized: Output of M2 normalize_incident().

        Returns:
            List of RichRetrievalResult, sorted descending by fused_score.
            Length ≤ final_top_k.  Each chunk_id appears at most once.

        CRITICAL: Does NOT filter by version applicability.
            Both 2.x and 3.x chunks for the same topic may appear.
        """
        if self._dense is None or self._sparse is None:
            raise RuntimeError("Call load() before retrieve().")

        incident = normalized.incident
        signals = normalized.signals

        # --- 1. Build query ---
        query_text = build_dense_query(
            incident.description,
            error_codes=signals.error_codes or None,
            technical_terms=signals.technical_terms or None,
            current_version=signals.current_version,
            previous_version=signals.previous_version,
            product=incident.product,
        )
        logger.debug("Dense query: %r", query_text[:120])

        # --- 2. Dense retrieval ---
        dense_hits: list[DenseHit] = self._dense.query(
            query_text, top_k=self._config.dense_top_k
        )

        # --- 3. BM25 retrieval ---
        bm25_hits: list[BM25Hit] = self._sparse.query(
            query_text, top_k=self._config.bm25_top_k
        )

        # --- 4. Merge by chunk_id (deduplication) ---
        dense_map: dict[str, DenseHit] = {h.chunk_id: h for h in dense_hits}
        bm25_map: dict[str, BM25Hit] = {h.chunk_id: h for h in bm25_hits}
        bm25_norms: dict[str, float] = _normalise_bm25(bm25_hits)

        # All unique chunk_ids from either channel.
        all_ids: set[str] = set(dense_map) | set(bm25_map)

        # --- 5. Build a lookup from chunk_id to chunk (for metadata) ---
        chunk_lookup: dict[str, EvidenceChunk] = {c.chunk_id: c for c in self._chunks}

        # --- 6. Compute fused scores ---
        rich_results: list[RichRetrievalResult] = []

        for chunk_id in all_ids:
            in_dense = chunk_id in dense_map
            in_bm25 = chunk_id in bm25_map

            # Scores.
            dense_score: float | None = dense_map[chunk_id].raw_score if in_dense else None
            bm25_raw: float | None = bm25_map[chunk_id].raw_score if in_bm25 else None
            bm25_norm: float | None = bm25_norms.get(chunk_id) if in_bm25 else None

            # Retrieve chunk metadata.  Prefer the DenseHit (has full metadata),
            # fall back to BM25Hit, then chunk_lookup.
            if in_dense:
                dh = dense_map[chunk_id]
                content = dh.content
                applies_to = dh.applies_to
                topic = dh.topic
                extra = dh.extra_metadata
                doc_id = dh.doc_id
            elif in_bm25:
                bh = bm25_map[chunk_id]
                content = bh.content
                applies_to = bh.applies_to
                topic = bh.topic
                extra = {}
                doc_id = bh.doc_id
            else:
                # Fallback: look up from original chunks.
                chunk = chunk_lookup.get(chunk_id)
                if chunk is None:
                    continue
                content = chunk.content
                applies_to = chunk.applies_to
                topic = chunk.topic or ""
                extra = chunk.metadata
                doc_id = chunk.doc_id

            # Signal boost.
            boost = compute_boost(
                content=content,
                applies_to=applies_to,
                metadata_error_codes=_get_metadata_error_codes(extra),
                signals=signals,
                boosts=self._config.boosts,
            )

            # Fuse.
            d_component = self._config.dense_weight * (dense_score or 0.0)
            b_component = self._config.bm25_weight * (bm25_norm or 0.0)
            fused = min(d_component + b_component + boost.total, 1.0)

            source = _determine_source(in_dense, in_bm25)

            rich_results.append(RichRetrievalResult(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                applies_to=applies_to,
                topic=topic,
                extra_metadata=extra,
                dense_score=dense_score,
                bm25_raw_score=bm25_raw,
                bm25_norm_score=bm25_norm,
                boost=boost,
                fused_score=fused,
                source=source,
            ))

        # --- 7. Sort: descending fused_score, stable tie-break on chunk_id ---
        rich_results.sort(key=lambda r: (-r.fused_score, r.chunk_id))

        # --- 8. Slice to final_top_k ---
        return rich_results[: self._config.final_top_k]

    def retrieve_as_contracts(
        self,
        normalized: NormalizedIncident,
    ) -> list[RetrievalResult]:
        """
        Convenience wrapper — returns M1 RetrievalResult contracts directly.
        This is the primary interface M4 will call.
        """
        return [r.to_contract() for r in self.retrieve(normalized)]
