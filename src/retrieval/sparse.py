"""
DevTrace — Module 3: Sparse (BM25) retrieval.

Responsibilities:
  - Build a BM25 index from M2 EvidenceChunks.
  - Tokenize corpus text in a way that preserves technical identifiers.
  - Query the BM25 index and return BM25Hit objects with raw scores.

Tokenisation design:
  Technical identifiers like AUTH_401, ERR_TIMEOUT, X-DevCore-Signature-256
  must NOT be split on underscores or hyphens. Our tokeniser:
    1. Preserves tokens that match [A-Z][A-Z0-9_-]+ (all-caps identifiers).
    2. Preserves tokens that look like error codes: WORD_NUMBER patterns.
    3. Falls back to simple whitespace+punctuation splitting for normal words.

This gives BM25 an advantage for exact signal matching without breaking
normal word retrieval.

BM25Hit is an internal dataclass — hybrid.py converts it into RetrievalResult.

IMPORTANT: No applicability filtering happens here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.models.contracts import EvidenceChunk
from src.retrieval.config import RetrievalConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Pattern: all-caps identifier including underscores/digits — treat as one token.
# Matches: AUTH_401, RATE_429, ERR_TIMEOUT, X_DEVCORE_SIGNATURE, HTTP_429.
_CAPS_IDENTIFIER = re.compile(r"\b[A-Z][A-Z0-9]*(?:[_-][A-Z0-9]+)+\b")

# Pattern: mixed-case technical terms we want to keep intact.
# Matches: OAuth, Bearer, HMAC-SHA256, X-DevCore-Signature-256.
_TECH_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)+\b")

# Fallback split: split on whitespace and common punctuation, keep alphanumerics.
_WORD_SPLIT = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """
    Tokenize text while preserving technical identifiers.

    Strategy:
    1. Extract all-caps identifiers (AUTH_401, RATE_429) as atomic tokens.
    2. Extract mixed-case tech tokens (OAuth, HMAC-SHA256) as atomic tokens.
    3. Split remainder on non-alphanumeric characters.
    4. Lowercase all tokens, but keep the caps-identifier form too (both added).

    The result has intentional duplication for caps identifiers — they appear
    once in original form and once lowercase — improving recall for queries
    that use either case.

    Returns:
        Flat list of string tokens, all lowercase.
    """
    tokens: list[str] = []
    used_spans: list[tuple[int, int]] = []

    # Pass 1: all-caps identifiers (highest priority — AUTH_401 etc.)
    for m in _CAPS_IDENTIFIER.finditer(text):
        tokens.append(m.group().lower())
        used_spans.append((m.start(), m.end()))

    # Pass 2: mixed-case tech tokens (OAuth, HMAC, etc.)
    for m in _TECH_TOKEN.finditer(text):
        # Skip if already covered by a caps identifier.
        if any(s <= m.start() and m.end() <= e for s, e in used_spans):
            continue
        tokens.append(m.group().lower())
        used_spans.append((m.start(), m.end()))

    # Pass 3: fallback — plain words from spans not already matched.
    for m in _WORD_SPLIT.finditer(text):
        if any(s <= m.start() and m.end() <= e for s, e in used_spans):
            continue
        tokens.append(m.group().lower())

    return tokens


# ---------------------------------------------------------------------------
# Internal result type
# ---------------------------------------------------------------------------

@dataclass
class BM25Hit:
    """Raw result from BM25 retrieval (pre-fusion)."""

    chunk_id: str
    doc_id: str
    content: str
    applies_to: str
    topic: str
    raw_score: float   # BM25 score ∈ [0, ∞)


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    BM25 sparse retrieval over the DevCore evidence chunks.

    Lifecycle:
        retriever = BM25Retriever(chunks, config)
        hits = retriever.query(query_text, top_k=10)

    The index is built at construction time (in memory; fast for 28 chunks).
    """

    def __init__(
        self,
        chunks: list[EvidenceChunk],
        config: RetrievalConfig,
    ) -> None:
        self._chunks = chunks
        self._config = config

        # Build tokenised corpus.
        self._corpus_tokens: list[list[str]] = [
            tokenize(chunk.content) for chunk in chunks
        ]

        # Construct BM25 index.
        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.debug("BM25 index built over %d chunks.", len(chunks))

    def query(self, query_text: str, *, top_k: int | None = None) -> list[BM25Hit]:
        """
        Score all chunks against the query and return the top-k results.

        Args:
            query_text: Raw query string (the same query used for dense search).
            top_k:      Override the configured bm25_top_k.

        Returns:
            List of BM25Hit objects sorted by descending score.
            Chunks with score 0.0 are excluded (no lexical overlap).
        """
        k = top_k if top_k is not None else self._config.bm25_top_k

        query_tokens = tokenize(query_text)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Build (index, score) pairs and sort descending.
        indexed: list[tuple[int, float]] = [
            (i, float(scores[i])) for i in range(len(scores))
        ]
        indexed.sort(
            key=lambda x: (
                -x[1],
                getattr(self._chunks[x[0]], "chunk_id", getattr(self._chunks[x[0]], "doc_id", str(x[0]))),
            )
        )

        hits: list[BM25Hit] = []
        for idx, score in indexed[:k]:
            if score <= 0.0:
                break   # No point returning zero-score results.
            chunk = self._chunks[idx]
            c_id = getattr(chunk, "chunk_id", getattr(chunk, "doc_id", f"doc_{idx}"))
            d_id = getattr(chunk, "doc_id", getattr(chunk, "chunk_id", f"doc_{idx}"))
            applies = getattr(chunk, "applies_to", "*")
            top = getattr(chunk, "topic", "") or ""
            hits.append(BM25Hit(
                chunk_id=c_id,
                doc_id=d_id,
                content=chunk.content,
                applies_to=applies,
                topic=top,
                raw_score=score,
            ))

        return hits
