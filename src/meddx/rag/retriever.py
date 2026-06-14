"""Hybrid retriever: BGE-M3 query embedding → Qdrant RRF (dense + sparse).

Designed with dependency injection (client, embedder passed explicitly) so
agent tests can mock retrieval without needing GPU or a running Qdrant.

The public factory `get_retriever()` is cached — the heavy BGE-M3 model loads
once and is reused across all agent invocations in a session.
"""

from __future__ import annotations

import functools
from typing import Any

from meddx.config import settings


def _qdrant_search_imports() -> tuple:
    try:
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            Fusion,
            FusionQuery,
            Prefetch,
            Range,
            SparseVector,
        )
        return Filter, FieldCondition, Range, Prefetch, FusionQuery, Fusion, SparseVector
    except ImportError as exc:
        raise ImportError(
            "qdrant-client>=1.12 is required for retrieval. "
            "Install it: pip install 'qdrant-client>=1.12'"
        ) from exc


class HybridRetriever:
    """RRF fusion over named dense and sparse Qdrant vectors.

    Args:
        client:     QdrantClient instance (deferred type to avoid import at load)
        collection: Qdrant collection name
        embedder:   BGE_M3Embedder instance (deferred to avoid GPU import at load)
    """

    def __init__(self, client: Any, collection: str, embedder: Any) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        max_evidence_level: int | None = None,
    ) -> list[dict]:
        """Embed *query*, run hybrid RRF search, return payload dicts.

        Args:
            query:              English search query
            k:                  Number of results to return
            max_evidence_level: When set, pre-filter to evidence_level ≤ value
                                (1=meta-analysis … 8=other; see ADR-0002).
                                Falls back to no filter if the filtered search
                                returns fewer than *k* results.
        """
        Filter, FieldCondition, Range, Prefetch, FusionQuery, Fusion, SparseVector = (
            _qdrant_search_imports()
        )

        dense, sparse = self._embedder.encode([query])[0]

        ev_filter = None
        if max_evidence_level is not None:
            ev_filter = Filter(
                must=[
                    FieldCondition(
                        key="evidence_level",
                        range=Range(lte=max_evidence_level),
                    )
                ]
            )

        def _run(limit: int, filt) -> list[dict]:
            prefetch = [
                Prefetch(
                    query=dense,
                    using="dense",
                    limit=limit * 3,
                    filter=filt,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=list(sparse.keys()),
                        values=list(sparse.values()),
                    ),
                    using="sparse",
                    limit=limit * 3,
                    filter=filt,
                ),
            ]
            response = self._client.query_points(
                collection_name=self._collection,
                prefetch=prefetch,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
            return [p.payload for p in response.points if p.payload]

        results = _run(k, ev_filter)

        # Fallback: relax evidence filter if it produced too few results
        if ev_filter is not None and len(results) < max(1, k // 2):
            results = _run(k, None)

        return results

    def is_ready(self) -> bool:
        """True when the collection exists and contains at least one point."""
        try:
            return self._client.count(collection_name=self._collection).count > 0
        except Exception:
            return False


@functools.lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Shared retriever singleton — BGE-M3 loads on first call (GPU, ~2 GB)."""
    from qdrant_client import QdrantClient  # type: ignore[import-not-found]

    from meddx.rag.embedder import BGE_M3Embedder

    embedder = BGE_M3Embedder()
    client = QdrantClient(url=settings.qdrant_url)
    return HybridRetriever(
        client=client,
        collection=settings.qdrant_collection,
        embedder=embedder,
    )
