"""
DevTrace — Module 3: Corpus indexing.

Responsibilities:
  - Build the Chroma vector index from M2 EvidenceChunks.
  - Ensure idempotent indexing (no duplicates on repeated calls).
  - Store chunk metadata alongside embeddings so retrieval results are
    self-contained.
  - Expose a clear build_index() / get_or_create_collection() interface
    that separates index construction from query time.

IMPORTANT: This module does NOT query; querying is in dense.py.
IMPORTANT: This module does NOT filter by applicability; that is M4.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb import Collection

from src.models.contracts import EvidenceChunk
from src.retrieval.config import RetrievalConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_to_metadata(chunk: EvidenceChunk) -> dict[str, Any]:
    """
    Flatten EvidenceChunk fields into a Chroma-compatible metadata dict.

    Chroma metadata values must be str, int, float, or bool.
    The nested `metadata` dict is JSON-serialised as a string.
    """
    import json

    return {
        "doc_id": chunk.doc_id,
        "chunk_id": chunk.chunk_id,
        "applies_to": chunk.applies_to,
        "topic": chunk.topic or "",
        "extra_metadata": json.dumps(chunk.metadata),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_create_collection(
    config: RetrievalConfig,
) -> tuple[chromadb.PersistentClient, Collection]:
    """
    Open (or create) the persistent Chroma client and collection.

    Returns:
        (client, collection) — both are ready for upsert or query operations.
    """
    persist_dir = Path(config.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=config.chroma_collection_name,
        # cosine distance → similarity = 1 − distance.
        # Chroma returns distances, so dense.py converts: score = 1 − distance.
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def build_index(
    chunks: list[EvidenceChunk],
    config: RetrievalConfig,
    *,
    embedder: Any,  # sentence_transformers.SentenceTransformer, typed loosely to avoid import at module level
) -> Collection:
    """
    Index all EvidenceChunks into Chroma using the provided embedder.

    Idempotency: Chroma's upsert() replaces any existing document with the
    same ID. Repeated calls with the same corpus are therefore safe and
    produce the same final index state.

    Args:
        chunks:   Evidence chunks produced by M2 load_corpus_and_chunks().
        config:   Retrieval configuration (collection name, persist dir, etc.).
        embedder: A SentenceTransformer instance (caller constructs it so the
                  same model object can be shared with dense retrieval).

    Returns:
        The populated Chroma collection.
    """
    if not chunks:
        raise ValueError("Cannot build index: no chunks provided.")

    _client, collection = get_or_create_collection(config)

    # Batch-encode all chunk texts for efficiency.
    texts = [chunk.content for chunk in chunks]
    logger.info("Encoding %d chunks with model %s …", len(chunks), config.embedding_model)
    embeddings = embedder.encode(texts, show_progress_bar=False)

    ids = [chunk.chunk_id for chunk in chunks]
    metadatas = [_chunk_to_metadata(chunk) for chunk in chunks]

    # Upsert is idempotent: existing IDs are overwritten, new IDs are added.
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    logger.info(
        "Indexed %d chunks into collection '%s' at '%s'.",
        len(chunks),
        config.chroma_collection_name,
        config.chroma_persist_dir,
    )
    return collection
