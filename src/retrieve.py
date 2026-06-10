"""
FILE: src/retrieve.py

Semantic retrieval for "The Unofficial Guide".

retrieve(query, k=5) embeds the user's question with the same MiniLM model used
for indexing, runs a nearest-neighbour search against the Chroma collection, and
returns the top-k chunks with their source and distance score.

Output shape (per result):
    {
        "source":   "housing_reviews.txt",
        "distance": 0.23,           # cosine distance; lower = more similar
        "chunk_id": 2,
        "text":     "..."
    }
"""

import embed


def retrieve(query, k=5):
    """Return the top-k most semantically similar chunks for ``query``.

    Raises a clear error if the index has not been built yet.
    """
    if not query or not query.strip():
        return []

    collection = embed.get_collection()

    # Guard: an empty collection means build_index() was never run.
    if collection.count() == 0:
        raise RuntimeError(
            "The vector store is empty. Run `python src/embed.py` "
            "(or build_index()) to ingest and index documents first."
        )

    # Don't ask for more neighbours than we actually have.
    k = max(1, min(k, collection.count()))

    query_embedding = embed.embed_texts([query])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    # Chroma returns parallel lists nested one level deep (one per query).
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []
    for text, meta, distance in zip(documents, metadatas, distances):
        retrieved.append(
            {
                "source": meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id"),
                "distance": round(float(distance), 4),
                "text": text,
            }
        )
    return retrieved


if __name__ == "__main__":
    # Standalone retrieval test harness.
    demo_queries = [
        "What is the best dorm for first-year students?",
        "Where can I find a quiet place to study?",
        "How do I get downtown without a car?",
    ]
    for q in demo_queries:
        print(f"\nQuery: {q}")
        for r in retrieve(q, k=3):
            preview = r["text"][:90].replace("\n", " ")
            print(f"  [{r['distance']:.3f}] {r['source']} #{r['chunk_id']}: {preview}...")
