"""Corpus ingestion entry point (stub).

Planned flow (see docs/architecture/system-architecture-overview.md §4):
  Europe PMC REST (+ NCBI E-utilities) -> section-aware chunking with
  evidence metadata -> BGE-M3 dense+sparse embeddings -> Qdrant collection.

Run: python scripts/ingest.py --query "<MeSH or keywords>"
"""

if __name__ == "__main__":
    raise SystemExit("Not implemented yet — see docs/architecture/ §4.")
