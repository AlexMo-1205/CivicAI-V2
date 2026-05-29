"""ChromaDB client + collection accessors."""
from __future__ import annotations

from functools import lru_cache

import chromadb

from civicai.config import SETTINGS


@lru_cache(maxsize=1)
def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(SETTINGS.db_dir))


def get_collection():
    """Return the live `civicai` collection (must already exist via ingest)."""
    return get_client().get_collection(SETTINGS.collection_name)


def reset_collection():
    """Drop and recreate the collection. Used by the ingest pipeline."""
    client = get_client()
    try:
        client.delete_collection(SETTINGS.collection_name)
    except Exception:
        pass
    return client.create_collection(SETTINGS.collection_name)
