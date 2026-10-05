"""
DevCore corpus loader and evidence chunk builder.

Responsibilities:
  - Load the DevCore JSON corpus from disk.
  - Validate required fields using the M1 Document contract.
  - Produce deterministic Document and EvidenceChunk objects.
  - Reject malformed corpus entries with clear errors.

Stable IDs are fundamental: the same corpus file always produces the same
doc_id and chunk_id values. Retrieval, evaluation, and trace objects all
refer to these IDs.

Chunking strategy (M2):
  Documents in the DevCore corpus are already appropriately sized for
  single-pass retrieval (200–600 words each). Each document is therefore
  treated as a single chunk with chunk_id = "{doc_id}-C01". This keeps
  IDs predictable and the implementation simple. Semantic or token-aware
  chunking is DEFERRED TO M3/M4 if evaluation proves it is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.models.contracts import Document, EvidenceChunk


# Default corpus location relative to the project root.
_DEFAULT_CORPUS_PATH = Path(__file__).resolve().parents[2] / "data" / "corpus" / "devcore_corpus.json"

# Required fields for every corpus entry.
_REQUIRED_FIELDS: frozenset[str] = frozenset({"doc_id", "title", "content"})


class CorpusLoadError(Exception):
    """Raised when the corpus file cannot be loaded or is structurally invalid."""


class DocumentValidationError(CorpusLoadError):
    """Raised when an individual corpus entry fails validation."""

    def __init__(self, doc_id: str | None, message: str) -> None:
        self.doc_id = doc_id
        super().__init__(f"Document {doc_id!r}: {message}")


def _validate_raw_entry(raw: Any, index: int) -> None:
    """
    Check that a raw corpus entry has the minimum required fields
    before Pydantic validation.

    Args:
        raw: The raw object read from JSON.
        index: Position in the JSON array (for error messages).

    Raises:
        DocumentValidationError: If a required field is missing.
    """
    if not isinstance(raw, dict):
        raise DocumentValidationError(None, f"Entry at index {index} is not a JSON object.")

    doc_id = raw.get("doc_id")
    missing = _REQUIRED_FIELDS - raw.keys()
    if missing:
        raise DocumentValidationError(
            doc_id,
            f"Missing required fields: {sorted(missing)}",
        )

    if not isinstance(raw.get("content"), str) or not raw["content"].strip():
        raise DocumentValidationError(doc_id, "Field 'content' must be a non-empty string.")

    if not isinstance(raw.get("title"), str) or not raw["title"].strip():
        raise DocumentValidationError(doc_id, "Field 'title' must be a non-empty string.")


def load_corpus(
    corpus_path: Path | str | None = None,
    *,
    strict: bool = True,
) -> list[Document]:
    """
    Load the DevCore corpus from a JSON file and return validated Documents.

    Args:
        corpus_path: Path to the JSON corpus file. Defaults to the project
                     corpus at data/corpus/devcore_corpus.json.
        strict: If True (default), raises on any malformed entry. If False,
                skips malformed entries and continues loading.

    Returns:
        List of validated, immutable Document objects.

    Raises:
        CorpusLoadError: If the file cannot be read or is not valid JSON.
        DocumentValidationError: If an entry fails validation (strict mode).
    """
    path = Path(corpus_path) if corpus_path is not None else _DEFAULT_CORPUS_PATH

    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CorpusLoadError(f"Corpus file not found: {path}") from exc
    except OSError as exc:
        raise CorpusLoadError(f"Cannot read corpus file: {path}: {exc}") from exc

    try:
        raw_list = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise CorpusLoadError(f"Corpus file is not valid JSON: {exc}") from exc

    if not isinstance(raw_list, list):
        raise CorpusLoadError("Corpus file must contain a JSON array at the top level.")

    documents: list[Document] = []
    seen_ids: set[str] = set()

    for i, raw in enumerate(raw_list):
        try:
            _validate_raw_entry(raw, i)

            doc_id = raw["doc_id"]
            if doc_id in seen_ids:
                raise DocumentValidationError(doc_id, "Duplicate doc_id in corpus.")
            seen_ids.add(doc_id)

            # Build the Document using M1 contract.
            # applies_to defaults to "*" if absent.
            doc = Document(
                doc_id=doc_id,
                title=raw["title"],
                content=raw["content"],
                applies_to=raw.get("applies_to", "*"),
                topic=raw.get("topic"),
                metadata=raw.get("metadata", {}),
            )
            documents.append(doc)

        except (DocumentValidationError, ValidationError) as exc:
            if strict:
                raise
            # Non-strict: log and skip.
            import warnings  # noqa: PLC0415
            warnings.warn(f"Skipping malformed corpus entry at index {i}: {exc}", stacklevel=2)

    return documents


def build_chunks(documents: list[Document]) -> list[EvidenceChunk]:
    """
    Convert a list of Documents into retrievable EvidenceChunk objects.

    Chunking strategy (M2): one chunk per document.
    chunk_id = "{doc_id}-C01"

    The "C01" suffix is intentional — it leaves room for M3/M4 to introduce
    finer-grained chunking without breaking the ID convention.

    Args:
        documents: Validated Document objects from load_corpus().

    Returns:
        List of EvidenceChunk objects with stable, traceable IDs.
    """
    chunks: list[EvidenceChunk] = []
    for doc in documents:
        chunk = EvidenceChunk(
            chunk_id=f"{doc.doc_id}-C01",
            doc_id=doc.doc_id,
            content=doc.content,
            applies_to=doc.applies_to,
            topic=doc.topic,
            metadata=doc.metadata,
        )
        chunks.append(chunk)
    return chunks


def load_corpus_and_chunks(
    corpus_path: Path | str | None = None,
    *,
    strict: bool = True,
) -> tuple[list[Document], list[EvidenceChunk]]:
    """
    Convenience function: load corpus and build chunks in one call.

    Returns:
        (documents, chunks) — both are stable and traceable.
    """
    documents = load_corpus(corpus_path, strict=strict)
    chunks = build_chunks(documents)
    return documents, chunks
