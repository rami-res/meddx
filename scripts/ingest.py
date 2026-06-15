"""Corpus ingestion: literature source → chunking → BGE-M3 → Qdrant.

Run:
    python scripts/ingest.py --query "chest pain differential diagnosis" --limit 100
    python scripts/ingest.py --query "fever of unknown origin" --limit 50 --min-year 2015
    python scripts/ingest.py --query "dyspnea causes" --source ncbi    # force NCBI if EBI is down

Source options:
  auto         Try Europe PMC first; fall back to NCBI on any HTTP error (default)
  europe-pmc   Europe PMC only (EBI host — may be blocked on some networks)
  ncbi         PubMed E-utilities only (NIH host — reliable worldwide)

Prerequisites:
    docker compose up -d          # Qdrant must be running
    pip install FlagEmbedding     # BGE-M3 (downloads ~2 GB model on first run)
    OPENAI_API_KEY not required — this script uses no LLM.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running directly without 'python -m' or editable install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meddx.config import settings
from meddx.ingestion.chunker import chunk_article
from meddx.ingestion.europe_pmc import ArticleRecord, EuropePMCClient
from meddx.ingestion.ncbi import NCBIClient
from meddx.rag.embedder import BGE_M3Embedder
from meddx.rag.store import QdrantStore


def _progress(current: int, total: int, width: int = 40) -> str:
    filled = int(width * current / max(total, 1))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total}"


def _fetch(
    source: str,
    query: str,
    limit: int,
    min_year: int | None,
    max_year: int | None,
    ncbi_api_key: str | None,
) -> list[ArticleRecord]:
    """Fetch articles using the requested source strategy.

    'auto': try Europe PMC; on any HTTP/connection error, fall back to NCBI.
    'europe-pmc': Europe PMC only (raises on error).
    'ncbi': NCBI E-utilities only.
    """
    fetch_kwargs = dict(query=query, limit=limit, min_year=min_year, max_year=max_year)

    if source == "ncbi":
        print(f"  source: NCBI E-utilities (PubMed)")
        with NCBIClient(api_key=ncbi_api_key) as c:
            return c.search(**fetch_kwargs)

    if source == "europe-pmc":
        print(f"  source: Europe PMC (EBI)")
        with EuropePMCClient() as c:
            return c.search(**fetch_kwargs)

    # auto: try Europe PMC, fall back to NCBI on any error
    print(f"  source: auto (Europe PMC → NCBI fallback)")
    try:
        with EuropePMCClient() as c:
            articles = c.search(**fetch_kwargs)
        print(f"  backend: Europe PMC ✓")
        return articles
    except Exception as epmc_err:
        print(f"  Europe PMC unavailable ({type(epmc_err).__name__}: {epmc_err})")
        print(f"  Falling back to NCBI E-utilities…")
        with NCBIClient(api_key=ncbi_api_key) as c:
            return c.search(**fetch_kwargs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ingest medical literature into Qdrant"
    )
    parser.add_argument(
        "--query", required=True,
        help="Search query in English (MeSH terms or free text)",
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Maximum number of articles to fetch (default: 100)",
    )
    parser.add_argument(
        "--min-year", type=int, default=None,
        help="Only include articles published from this year onwards",
    )
    parser.add_argument(
        "--max-year", type=int, default=None,
        help="Only include articles published up to this year",
    )
    parser.add_argument(
        "--batch-size", type=int, default=12,
        help="BGE-M3 embedding batch size (default: 12, reduce if OOM)",
    )
    parser.add_argument(
        "--qdrant-url", default=settings.qdrant_url,
        help=f"Qdrant URL (default: {settings.qdrant_url})",
    )
    parser.add_argument(
        "--collection", default=settings.qdrant_collection,
        help=f"Qdrant collection name (default: {settings.qdrant_collection})",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "europe-pmc", "ncbi"],
        default="auto",
        help="Article source: 'auto' tries Europe PMC then falls back to NCBI (default: auto)",
    )
    args = parser.parse_args(argv)

    ncbi_api_key: str | None = os.getenv("NCBI_API_KEY")

    # ── 1. Fetch articles ──────────────────────────────────────────────────
    print(f"\n[1/4] Fetching  query={args.query!r}  limit={args.limit}")
    articles = _fetch(
        source=args.source,
        query=args.query,
        limit=args.limit,
        min_year=args.min_year,
        max_year=args.max_year,
        ncbi_api_key=ncbi_api_key,
    )

    usable = [a for a in articles if a.has_usable_text]
    skipped = len(articles) - len(usable)
    print(f"  fetched={len(articles)}  usable={len(usable)}  skipped (no abstract)={skipped}")

    if not usable:
        print("No usable articles — try a different query, higher --limit, or --source ncbi.")
        return

    # Study-type breakdown
    from collections import Counter
    type_counts = Counter(a.study_type for a in usable)
    print("  study types: " + "  ".join(f"{t}:{n}" for t, n in type_counts.most_common()))

    # ── 2. Chunk ───────────────────────────────────────────────────────────
    print(f"\n[2/4] Chunking {len(usable)} articles …")
    chunks = []
    for article in usable:
        chunks.extend(chunk_article(article))
    print(f"  → {len(chunks)} chunks")

    section_counts = Counter(c.section for c in chunks)
    print("  sections: " + "  ".join(f"{s}:{n}" for s, n in section_counts.most_common()))

    # ── 3. Embed ───────────────────────────────────────────────────────────
    print(f"\n[3/4] Embedding with BGE-M3 (batch_size={args.batch_size}) …")
    print("  (first run downloads BAAI/bge-m3 ~2 GB to ~/.cache/huggingface)")
    embedder = BGE_M3Embedder()

    texts = [c.text for c in chunks]
    embeddings: list[tuple[list[float], dict[int, float]]] = []
    total = len(texts)
    bs = args.batch_size

    for start in range(0, total, bs):
        batch = texts[start: start + bs]
        embeddings.extend(embedder.encode(batch, batch_size=bs))
        print(f"  {_progress(min(start + bs, total), total)}", end="\r", flush=True)
    print()  # newline after progress bar

    # ── 4. Upsert ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Upserting to Qdrant  url={args.qdrant_url}  collection={args.collection}")
    store = QdrantStore(url=args.qdrant_url, collection=args.collection)
    n = store.upsert(chunks, embeddings)
    total_in_collection = store.count()
    store.close()

    print(f"  upserted={n}  total in collection={total_in_collection}")
    print("\nDone.")


if __name__ == "__main__":
    main()
