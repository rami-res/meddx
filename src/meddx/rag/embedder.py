"""BGE-M3 dense+sparse embedder running locally on GPU (ADR-0004).

BGE-M3 produces both dense (1024-d) and sparse (SPLADE-style) vectors in a
single forward pass, making it ideal for Qdrant hybrid search with RRF fusion.
The model is multilingual (100+ languages) — student queries in any language
are embedded into the same space as the English-language corpus.

Import is deferred so that the rest of the codebase (graph, agents) can be
imported and tested without requiring torch + FlagEmbedding to be installed.
Only the ingestion pipeline and the RAG retriever need this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type hint only — never imported at module level
    from FlagEmbedding import BGEM3FlagModel as _BGEM3FlagModel

MODEL_NAME = "BAAI/bge-m3"
DENSE_DIM = 1024  # BGE-M3 output dimension


class BGE_M3Embedder:
    """Wraps FlagEmbedding BGEM3FlagModel for dense+sparse encoding.

    Usage:
        embedder = BGE_M3Embedder()
        pairs = embedder.encode(["text one", "text two"])
        dense_vec, sparse_weights = pairs[0]
    """

    def __init__(self, use_fp16: bool = True):
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is required for embeddings. "
                "Install it: pip install FlagEmbedding  "
                "(pulls PyTorch; RTX 4080 used automatically if CUDA is available)."
            ) from exc
        self._model: _BGEM3FlagModel = BGEM3FlagModel(MODEL_NAME, use_fp16=use_fp16)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 12,
    ) -> list[tuple[list[float], dict[int, float]]]:
        """Return one (dense_vec, sparse_weights) tuple per input text.

        dense_vec       — list[float] of length DENSE_DIM (1024)
        sparse_weights  — dict[token_id, weight] (SPLADE-style; may be empty)
        """
        outputs = self._model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs = outputs["dense_vecs"]
        sparse_vecs = outputs["lexical_weights"]  # list[dict[int|str, float]]

        result: list[tuple[list[float], dict[int, float]]] = []
        for dense, sparse in zip(dense_vecs, sparse_vecs):
            result.append((
                dense.tolist(),
                {int(k): float(v) for k, v in sparse.items()},
            ))
        return result
