# 0002. Qdrant as the vector database

Date: 2026-06-11
Status: Accepted

## Context

The RAG layer stores chunks of open-access medical literature and has
specific requirements:

1. **Hybrid search (dense + sparse).** Medical terminology contains exact
   terms (drug names, nosologies, MeSH terms) where lexical match is
   critical; pure semantic search dilutes them. Our embedding model (BGE-M3,
   see ADR-0004) produces both dense and sparse vectors, so the store must
   support both natively.
2. **Rich metadata filtering.** Publication year, study type (meta-analysis,
   systematic review, RCT, case report), journal, evidence level. This is
   the foundation of the "evidence level over national protocols" mechanism.
3. **Free local deployment** (course project) with a path to scale later.
4. A good Python client and LangChain integration.

## Decision

Use **Qdrant** (open source, run locally via Docker in `docker-compose.yml`).

- Named vectors: dense + sparse in a single collection; server-side fusion
  (Query API, RRF).
- Chunk metadata in payload with indexes for filtering
  (`pmid`, `doi`, `journal`, `year`, `study_type`, `evidence_level`).
- The same code works against local Docker and Qdrant Cloud (free tier
  exists) if a cloud demo is ever needed.

### Alternatives considered

| Option | Why rejected |
|---|---|
| **Chroma** | Simplest to start, but weaker filtering and no native sparse/hybrid — BM25 would have to be bolted on separately |
| **pgvector** | Reasonable if PostgreSQL were already required; here it adds infrastructure and hybrid search must be hand-rolled |
| **Weaviate** | Comparable features but heavier to operate locally; Qdrant is lighter and simpler |
| **Pinecone** | Managed-only, vendor lock-in, limited free tier — undesirable for a course project |
| **FAISS** | Index library only: no payload filters, persistence, or CRUD out of the box |
| **Milvus** | Powerful but operationally heavy (multiple components) for a single-host project |

## Consequences

- Docker is required for local development (`docker compose up -d qdrant`) —
  already available on the dev machine.
- Hybrid search and evidence-level filtering are implemented with built-in
  database features rather than custom code.
- Free for development; migration path to Qdrant Cloud without code changes.
- Changing the embedding model requires re-creating the collection
  (vector dimensions are fixed per collection) — raw corpus is kept in
  `data/` for cheap re-indexing.
