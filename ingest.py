"""
CivicAI — Ingest
Lit les documents txt du dossier docs/, les découpe en chunks,
les convertit en embeddings et les stocke dans ChromaDB.
Script one-shot : à relancer uniquement si tu ajoutes des documents.
"""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR      = Path("docs")
DB_DIR        = Path("chroma_db")
COLLECTION    = "civicai"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# ── Modèle d'embedding ────────────────────────────────────────────────────────
print("Chargement du modèle d'embedding...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
client = chromadb.PersistentClient(path=str(DB_DIR))

try:
    client.delete_collection(COLLECTION)
except:
    pass
collection = client.create_collection(COLLECTION)


# ── Fonctions ─────────────────────────────────────────────────────────────────
def chunk_text(text: str, source: str) -> list[dict]:
    chunks = []
    start  = 0
    idx    = 0

    while start < len(text):
        end   = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "text":     chunk,
                "source":   source,
                "chunk_id": idx
            })
            idx += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def ingest_all_docs():
    txt_files = list(DOCS_DIR.glob("*.txt"))

    if not txt_files:
        print(f"Aucun document trouvé dans {DOCS_DIR}/")
        return

    print(f"{len(txt_files)} documents trouvés : {[f.name for f in txt_files]}\n")

    all_chunks = []

    for txt_path in txt_files:
        print(f"Lecture de {txt_path.name}...")
        text   = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(text, source=txt_path.name)
        print(f"  → {len(text)} caractères, {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal : {len(all_chunks)} chunks à embedder...")

    texts      = [c["text"]   for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids        = [f"{c['source']}_{c['chunk_id']}" for c in all_chunks],
        embeddings = embeddings,
        documents  = texts,
        metadatas  = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in all_chunks]
    )

    print(f"\n✅ Base vectorielle créée dans {DB_DIR}/")
    print(f"   Collection : '{COLLECTION}'")
    print(f"   {collection.count()} chunks stockés")


if __name__ == "__main__":
    ingest_all_docs()
