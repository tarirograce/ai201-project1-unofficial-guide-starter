"""
FILE: src/query.py

The end-to-end query pipeline for "The Unofficial Guide".

    ask(question)  ->  {"answer": "...", "sources": [...]}

Pipeline:
    Question
       |
       v
    Retrieve (semantic search over ChromaDB)
       |
       v
    Generate (grounded answer via Groq LLM)
       |
       v
    Return answer + programmatically-attached sources

Sources are derived from the retrieved chunks, NOT from the model output, so
they can never be hallucinated. When the model refuses (no grounded answer), the
sources list is empty.
"""

import retrieve as _retrieve
import generate as _generate


def ask(question, k=5):
    """Answer a question against the indexed documents.

    Returns a dict: {"answer": str, "sources": list[str]}.
    The returned ``answer`` already has a "Sources:" block appended when the
    question was answered from the documents.
    """
    if not question or not question.strip():
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    # 1. Retrieve the most relevant chunks.
    chunks = _retrieve.retrieve(question, k=k)

    # 2. Generate a grounded answer from those chunks.
    answer = _generate.generate_answer(question, chunks)

    # 3. Decide sources. If the model refused (or nothing was retrieved), there
    #    are no grounded sources to cite.
    if not chunks or _generate.is_refusal(answer):
        return {"answer": _generate.REFUSAL, "sources": []}

    # Unique source filenames, preserving retrieval order (most relevant first).
    sources = list(dict.fromkeys(c["source"] for c in chunks))

    # Append the Sources block programmatically.
    answer_with_sources = answer + "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

    return {"answer": answer_with_sources, "sources": sources}


if __name__ == "__main__":
    # Quick manual test of the full pipeline.
    for q in [
        "What is the best dorm for first-year students?",
        "How do I become a professional astronaut?",  # out of scope -> refusal
    ]:
        print("=" * 70)
        print("Q:", q)
        result = ask(q)
        print("A:", result["answer"])
        print("Sources:", result["sources"])
