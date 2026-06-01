"""ChromaDB client + collection accessors."""
from __future__ import annotations

from functools import lru_cache

import chromadb

from civicai.config import SETTINGS


@lru_cache(maxsize=1)
def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(SETTINGS.db_dir))


def get_collection():
    """Return the live collection (must already exist via ingest)."""
    return get_client().get_collection(SETTINGS.collection_name)


def reset_collection():
    """Drop and recreate the collection. Used by the ingest pipeline.

    The collection is created with cosine distance so `score = 1 - distance`
    sits in roughly [0, 1] for relevant matches — directly comparable across
    queries and easy to threshold on.
    """
    client = get_client()
    try:
        client.delete_collection(SETTINGS.collection_name)
    except Exception:
        pass
    return client.create_collection(
        SETTINGS.collection_name,
        metadata={"hnsw:space": "cosine"},
    )
