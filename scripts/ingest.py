"""CLI entrypoint for the ingestion pipeline."""
from __future__ import annotations

from civicai.rag.ingest import ingest_all_docs


def main() -> None:
    ingest_all_docs()


if __name__ == "__main__":
    main()
