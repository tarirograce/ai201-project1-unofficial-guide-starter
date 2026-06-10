"""
FILE: src/embed.py

Embedding + vector store layer for "The Unofficial Guide".

  * Embedding model: sentence-transformers "all-MiniLM-L6-v2"
  * Vector store:     ChromaDB (persistent, on disk under ./chroma_db)

This module owns:
  * loading the embedding model once and reusing it (it is expensive to load)
  * turning text into vectors
  * creating / accessing the Chroma collection
  * build_index(): the end-to-end ingest -> chunk -> embed -> store pipeline

Both retrieve.py and the query pipeline import the helpers here so there is a
single source of truth for "which model" and "which collection".
"""

import os

import chromadb
from sentence_transformers import SentenceTransformer

import ingest
import chunk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(ROOT, "chroma_db")

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "unofficial_guide"

# Cache the model at module level — loading it takes a few seconds and we do not
# want to pay that cost on every query.
_model = None


def get_model():
    """Lazily load and cache the SentenceTransformer model."""
    global _model
    if _model is None:
        print(f"[embed] Loading embedding model '{MODEL_NAME}' (first call only)...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts):
    """Embed a list of strings and return a list of float vectors (lists)."""
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # pairs naturally with cosine distance
    )
    return vectors.tolist()


def get_client():
    """Return a persistent Chroma client backed by ./chroma_db."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection(reset=False):
    """Return the Chroma collection, optionally wiping it first.

    We use cosine distance because our embeddings are L2-normalized, which makes
    distances comparable and easy to threshold.
    """
    client = get_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            # Collection may not exist yet on a first run — that's fine.
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_index():
    """Run the full pipeline: ingest -> chunk -> embed -> store in ChromaDB.

    Rebuilds the collection from scratch each time so re-running is idempotent.
    Returns the number of chunks indexed.
    """
    documents = ingest.run(save=True)
    chunks = chunk.chunk_documents(documents)
    if not chunks:
        raise RuntimeError("No chunks produced — nothing to index.")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # IDs must be unique; (source, chunk_id) is unique by construction.
    ids = [f"{c['source']}::{c['chunk_id']}" for c in chunks]
    # Metadata must include source and chunk_id (spec requirement).
    metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]

    collection = get_collection(reset=True)
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(
        f"[embed] Indexed {len(chunks)} chunks into Chroma collection "
        f"'{COLLECTION_NAME}' at {CHROMA_DIR}"
    )
    return len(chunks)


if __name__ == "__main__":
    n = build_index()
    print(f"[embed] Done. {n} chunks indexed.")
