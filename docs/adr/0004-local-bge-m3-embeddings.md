# 0004. Local BGE-M3 embeddings for multilingual hybrid search

Date: 2026-06-11
Status: Accepted

## Context

- The corpus is English-language medical literature; **students query the
  system in many languages** (an explicit requirement), so cross-lingual
  retrieval quality matters.
- Hybrid search in Qdrant (ADR-0002) needs both dense and sparse vectors.
- Course-project budget favors free local inference; the dev machine has an
  RTX 4080 (12 GB VRAM), far more than embedding inference requires.

## Decision

Use **BGE-M3** (BAAI) running locally on the GPU for both ingestion and
query-time embedding:

- **Multilingual** (100+ languages) — supports cross-lingual retrieval for
  queries in Ukrainian or other languages against English literature.
- Produces **dense and sparse (lexical) vectors in a single forward pass** —
  maps directly onto Qdrant named vectors and RRF hybrid fusion.
- Free, offline, reproducible; ~2 GB VRAM at inference.

Defense in depth for cross-linguality: the Evidence Agent additionally
normalizes literature search queries to English (medical literature is
English), so retrieval does not rely on cross-lingual embeddings alone.

A local reranker (`bge-reranker-v2-m3`) is a planned follow-up on top of
hybrid retrieval (tracked in the architecture overview's open questions).

### Alternatives considered

- **Voyage AI (`voyage-3-large`)** — slightly stronger retrieval quality and
  a managed reranker, but a paid API, another key to manage, and a network
  dependency for every query.
- **OpenAI `text-embedding-3-large`** — good multilingual quality, same key
  as the default LLM provider, but paid per call and dense-only (sparse side
  of hybrid search would need a separate solution).
- **PubMedBERT-based embeddings** — domain-tuned for biomedicine but not
  multilingual; fails the cross-lingual requirement.
- **multilingual-e5-large** — solid multilingual dense model, but dense-only;
  BGE-M3 covers the same need plus sparse vectors.

## Consequences

- Zero embedding cost; ingestion of large corpora is bounded by local GPU
  time, not API budget.
- `FlagEmbedding` (and a CUDA-enabled PyTorch) becomes a dependency; first
  run downloads ~2 GB of weights.
- Embedding model change requires re-indexing the Qdrant collection — raw
  corpus is kept in `data/` to make that cheap.
- Quality is slightly below the best paid APIs; acceptable for the project,
  and the reranker follow-up narrows the gap where it matters (top-k
  precision before the LLM context).
