"""Ingestion pipeline: read .txt docs, chunk, embed, store in ChromaDB."""
from __future__ import annotations

from pathlib import Path

from civicai.config import SETTINGS
from civicai.rag.embeddings import get_embedder
from civicai.rag.vectorstore import reset_collection


def chunk_text(text: str, source: str) -> list[dict]:
    """Slide a fixed window across `text` with overlap."""
    chunks: list[dict] = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + SETTINGS.chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append({"text": chunk, "source": source, "chunk_id": idx})
            idx += 1

        start += SETTINGS.chunk_size - SETTINGS.chunk_overlap

    return chunks


def ingest_all_docs(docs_dir: Path | None = None) -> int:
    """Wipe + repopulate the vector store from every .txt in `docs_dir`.

    Returns the number of chunks stored.
    """
    docs_dir = docs_dir or SETTINGS.docs_dir
    txt_files = list(docs_dir.glob("*.txt"))

    if not txt_files:
        print(f"No documents found in {docs_dir}/")
        return 0

    print(f"{len(txt_files)} documents found: {[f.name for f in txt_files]}\n")

    all_chunks: list[dict] = []
    for txt_path in txt_files:
        print(f"Reading {txt_path.name}...")
        text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(text, source=txt_path.name)
        print(f"  -> {len(text)} chars, {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal: {len(all_chunks)} chunks to embed...")

    embedder = get_embedder()
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    collection = reset_collection()
    collection.add(
        ids=[f"{c['source']}_{c['chunk_id']}" for c in all_chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"source": c["source"], "chunk_id": c["chunk_id"]}
            for c in all_chunks
        ],
    )

    stored = collection.count()
    print(f"\nVector store ready at {SETTINGS.db_dir}/")
    print(f"  Collection: '{SETTINGS.collection_name}'")
    print(f"  {stored} chunks stored")
    return stored
