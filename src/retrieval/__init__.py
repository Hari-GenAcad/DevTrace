"""
DevTrace — Module 3: Hybrid Retrieval.

Public surface:
    HybridRetriever   — the main entry point for M4 to consume.
    RetrievalConfig   — configurable parameters.
    build_index       — one-time corpus indexing.
    retrieve          — query the built index.
"""

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.index import build_index

__all__ = ["HybridRetriever", "build_index"]
