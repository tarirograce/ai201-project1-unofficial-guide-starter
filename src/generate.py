"""
FILE: src/generate.py

Grounded answer generation for "The Unofficial Guide".

We send the retrieved chunks to a Groq-hosted LLM (llama-3.3-70b-versatile) with
a STRICT grounding prompt. The model is instructed to:
  * answer ONLY using the retrieved chunks
  * never use outside / prior knowledge
  * refuse with an exact sentence when the answer is not in the chunks
  * NOT fabricate a "Sources" list (sources are attached programmatically in the
    query pipeline so they can never be hallucinated)

The Groq API key is loaded from the environment (.env via python-dotenv).
"""

import os

from dotenv import load_dotenv
from groq import Groq

# Load environment variables from a .env file if present.
load_dotenv()

MODEL = "llama-3.3-70b-versatile"

# The exact refusal sentence required by the spec. retrieve/generate/query all
# reference this single constant so the wording can never drift.
REFUSAL = "I don't have enough information in the provided documents to answer that."

# Strict grounding system prompt.
SYSTEM_PROMPT = f"""You are "The Unofficial Guide", a question-answering assistant for new university students.

You will be given a user QUESTION and a set of CONTEXT passages retrieved from a collection of student-written guide documents.

Follow these rules without exception:
1. Answer ONLY using information found in the CONTEXT passages.
2. NEVER use outside knowledge, prior training, or assumptions. If the CONTEXT does not contain the answer, you do not know it.
3. If the CONTEXT does not contain enough information to answer the question, reply with EXACTLY this sentence and nothing else:
   "{REFUSAL}"
4. Do not invent facts, numbers, names, or details that are not in the CONTEXT.
5. Be concise and helpful. Write in a friendly, direct tone.
6. Do NOT add a "Sources" section or cite filenames yourself — sources are attached automatically after your answer.
"""


def _format_context(chunks):
    """Render retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[Passage {i} | source: {c['source']}]\n{c['text']}")
    return "\n\n".join(blocks)


def _get_client():
    """Construct a Groq client, failing with a clear message if the key is missing."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "Groq API key, or export GROQ_API_KEY in your environment."
        )
    return Groq(api_key=api_key)


def generate_answer(question, chunks):
    """Generate a grounded answer from the retrieved chunks.

    Returns the model's answer text only (no sources). If there are no chunks,
    we short-circuit to the refusal without calling the API.
    """
    if not chunks:
        return REFUSAL

    client = _get_client()
    context = _format_context(chunks)
    user_message = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the CONTEXT above, following all the rules."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,   # deterministic, factual answers
            max_tokens=600,
        )
    except Exception as exc:
        # Surface API/network errors clearly rather than crashing the UI.
        raise RuntimeError(f"Groq generation failed: {exc}") from exc

    return response.choices[0].message.content.strip()


def is_refusal(answer):
    """Return True if the model's answer is (or contains) the refusal sentence."""
    return REFUSAL.lower() in answer.lower()


if __name__ == "__main__":
    # Standalone test using the real retrieval + generation path.
    import retrieve

    question = "What is the best dorm for first-year students?"
    chunks = retrieve.retrieve(question, k=5)
    print("Question:", question)
    print("Answer:\n", generate_answer(question, chunks))
