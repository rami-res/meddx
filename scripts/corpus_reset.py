"""Delete and recreate the Qdrant collection (destroys all ingested data).

Called by 'make corpus-reset'. Requires interactive confirmation unless
--yes is passed (useful for CI/scripted resets).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete the Qdrant corpus collection")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("qdrant-client not installed — run: pip install qdrant-client")
        sys.exit(1)

    from meddx.config import settings
    url = settings.qdrant_url
    col = settings.qdrant_collection

    if not args.yes:
        print(f"WARNING: This will permanently delete all data in collection '{col}' at {url}.")
        answer = input("Type YES to confirm: ").strip()
        if answer != "YES":
            print("Aborted.")
            sys.exit(0)

    client = QdrantClient(url=url, timeout=10)

    if client.collection_exists(col):
        client.delete_collection(col)
        print(f"Deleted collection '{col}'.")
    else:
        print(f"Collection '{col}' does not exist — nothing to delete.")

    client.close()
    print("Done. Run 'make ingest-corpus' to repopulate.")


if __name__ == "__main__":
    main()
