"""
FILE: src/chunk.py

Chunking strategy for "The Unofficial Guide".

We split each cleaned document into overlapping, fixed-size character windows.

  CHUNK_SIZE = 500 characters
  OVERLAP    = 100 characters

WHY THIS CHUNK SIZE?
  500 characters is roughly 80-100 words / 3-5 sentences. That is large enough
  to hold a complete thought (a dorm review, a single piece of advice) so the
  embedding captures real meaning, but small enough that a retrieved chunk is
  tightly focused on one topic. Tiny chunks lose context; huge chunks dilute the
  embedding with many topics and hurt retrieval precision. 500 chars is a good
  middle ground for short, student-written documents like ours.

WHY OVERLAP?
  An idea (or a sentence) often straddles the boundary between two chunks. If we
  cut with no overlap, a fact split across the boundary could be lost from both
  chunks. A 100-character (20%) overlap means each boundary sentence appears in
  both neighbouring chunks, so the relevant context is preserved no matter where
  the answer happens to fall. The cost is a modest amount of duplicated text.

The chunker also:
  * avoids splitting words (it backs up to the last whitespace near the boundary)
  * never emits empty / whitespace-only chunks
  * attaches metadata: {"source": filename, "chunk_id": index, "text": chunk}
"""

CHUNK_SIZE = 500   # characters per chunk
OVERLAP = 100      # characters shared between consecutive chunks

# How far back from a hard boundary we are willing to look for whitespace so we
# don't cut a word in half. If no whitespace is found within this window we fall
# back to a hard cut (better a clean size than an infinite search).
_WORD_BOUNDARY_LOOKBACK = 40


def chunk_text(text, source, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split a single document's text into overlapping chunks with metadata.

    Returns a list of {"source", "chunk_id", "text"} dicts.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    chunk_id = 0
    text_len = len(text)
    step = chunk_size - overlap  # how far the window advances each iteration

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try not to split a word: if we're not at the very end of the text and
        # the boundary lands in the middle of a word, back up to the last space.
        if end < text_len and not text[end].isspace():
            window_start = max(start + 1, end - _WORD_BOUNDARY_LOOKBACK)
            last_space = text.rfind(" ", window_start, end)
            if last_space != -1:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:  # never store empty / whitespace-only chunks
            chunks.append(
                {
                    "source": source,
                    "chunk_id": chunk_id,
                    "text": chunk,
                }
            )
            chunk_id += 1

        if end >= text_len:
            break

        # Advance the window. We advance from the *actual* end so that a
        # word-boundary adjustment still keeps a consistent overlap.
        start = max(end - overlap, start + step)

    return chunks


def chunk_documents(documents):
    """Chunk a list of {"source", "text"} docs into a flat list of chunk dicts.

    chunk_id is per-document, so (source, chunk_id) uniquely identifies a chunk.
    """
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_text(doc["text"], doc["source"])
        all_chunks.extend(doc_chunks)
    print(
        f"[chunk] Created {len(all_chunks)} chunks from {len(documents)} documents "
        f"(size={CHUNK_SIZE}, overlap={OVERLAP})"
    )
    return all_chunks


if __name__ == "__main__":
    # Standalone smoke test: ingest then chunk, and print a sample.
    import ingest

    docs = ingest.run(save=False)
    chunks = chunk_documents(docs)
    if chunks:
        sample = chunks[0]
        print("\nSample chunk:")
        print(f"  source   : {sample['source']}")
        print(f"  chunk_id : {sample['chunk_id']}")
        print(f"  text     : {sample['text'][:200]}...")
