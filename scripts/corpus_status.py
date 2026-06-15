"""Print Qdrant collection stats: document count, vector config, status."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("qdrant-client not installed — run: pip install qdrant-client")
        sys.exit(0)

    try:
        from meddx.config import settings
        url = settings.qdrant_url
        col = settings.qdrant_collection
    except Exception:
        url, col = "http://localhost:6333", "med_literature"

    try:
        client = QdrantClient(url=url, timeout=5)
    except Exception as e:
        print(f"Cannot connect to Qdrant at {url}: {e}")
        print("Is Qdrant running?  →  make infra")
        sys.exit(1)

    if not client.collection_exists(col):
        print(f"\nCollection '{col}' does not exist yet.")
        print("Run 'make ingest-demo' to populate it.")
        client.close()
        return

    info  = client.get_collection(col)
    count = client.count(col).count
    cfg   = info.config.params.vectors

    print(f"\n{'─' * 40}")
    print(f"  Collection : {col}")
    print(f"  URL        : {url}")
    print(f"  Points     : {count:,}")
    print(f"  Status     : {info.status.value}")

    if isinstance(cfg, dict):
        for name, vcfg in cfg.items():
            print(f"  Vector [{name}] : dim={vcfg.size}  distance={vcfg.distance.value}")
    print(f"{'─' * 40}\n")

    client.close()


if __name__ == "__main__":
    main()
