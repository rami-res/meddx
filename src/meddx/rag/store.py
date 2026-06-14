"""Qdrant store: collection setup and chunk upsert (ADR-0002).

Named vectors:
  "dense"   — BGE-M3 1024-d cosine vectors (semantic search)
  "sparse"  — SPLADE-style token weights with IDF modifier (keyword search)

Qdrant RRF (Reciprocal Rank Fusion) combines both at query time.
Payload fields that drive evidence-level filtering:
  year, study_type, evidence_level, pmid, doi, section, text, title, journal
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meddx.ingestion.chunker import ArticleChunk

# Deferred import — do not import qdrant at module top so that graph/agents
# can be imported without qdrant-client installed.
DENSE_DIM = 1024


def _qdrant_imports():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance,
            Modifier,
            PointStruct,
            SparseVector,
            SparseVectorParams,
            VectorParams,
        )
        return QdrantClient, Distance, Modifier, PointStruct, SparseVector, SparseVectorParams, VectorParams
    except ImportError as exc:
        raise ImportError(
            "qdrant-client is required for vector storage. "
            "Install it: pip install 'qdrant-client>=1.12'"
        ) from exc


class QdrantStore:
    """Thin wrapper around qdrant-client for the MedDx literature corpus."""

    def __init__(self, url: str, collection: str):
        QdrantClient, *_ = _qdrant_imports()
        self._client = QdrantClient(url=url)
        self._collection = collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        QdrantClient, Distance, Modifier, _, _, SparseVectorParams, VectorParams = _qdrant_imports()
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=Modifier.IDF),
            },
        )

    def upsert(
        self,
        chunks: list[ArticleChunk],
        embeddings: list[tuple[list[float], dict[int, float]]],
    ) -> int:
        """Upsert chunks with their embeddings. Returns number of points upserted."""
        _, _, _, PointStruct, SparseVector, *_ = _qdrant_imports()

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )

        points = []
        for chunk, (dense, sparse) in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=chunk.point_id(),
                    vector={
                        "dense": dense,
                        "sparse": SparseVector(
                            indices=list(sparse.keys()),
                            values=list(sparse.values()),
                        ),
                    },
                    payload=chunk.payload(),
                )
            )

        # Qdrant upsert is idempotent: re-running with the same point_id
        # (deterministic from pmid/doi + chunk_index) overwrites safely.
        self._client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count

    def close(self) -> None:
        self._client.close()
