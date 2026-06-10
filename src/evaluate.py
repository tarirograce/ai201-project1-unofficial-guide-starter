"""
FILE: src/evaluate.py

Evaluation framework for "The Unofficial Guide".

Reads a set of test questions (each with an expected answer) from
sample_questions.json, runs the full ask() pipeline on each, and records:
  * the question
  * the expected answer
  * the retrieved chunks (source + distance)
  * the actual generated answer
  * an accuracy judgment: Accurate | Partially Accurate | Inaccurate

The accuracy judgment is produced by an LLM judge (Groq) that compares the
expected and actual answers. The judge is a separate, strict prompt so the
evaluation is reproducible. Results are written to a markdown report.
"""

import os
import json

from dotenv import load_dotenv
from groq import Groq

import retrieve as _retrieve
import query as _query
import generate as _generate

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS_PATH = os.path.join(ROOT, "sample_questions.json")
REPORT_PATH = os.path.join(ROOT, "evaluation_report.md")

JUDGE_MODEL = "llama-3.3-70b-versatile"
VALID_JUDGMENTS = {"Accurate", "Partially Accurate", "Inaccurate"}

JUDGE_SYSTEM_PROMPT = """You are a strict evaluation judge for a question-answering system.
You compare an EXPECTED answer to an ACTUAL answer and decide how accurate the ACTUAL answer is.

Respond with EXACTLY one of these three labels and nothing else:
- Accurate            (the actual answer captures the key facts of the expected answer with no contradictions)
- Partially Accurate  (the actual answer is partly correct but misses key facts or includes minor errors)
- Inaccurate          (the actual answer is wrong, contradicts the expected answer, or fails to answer)

Output only the label."""


def load_questions():
    """Load the evaluation questions from sample_questions.json."""
    if not os.path.exists(QUESTIONS_PATH):
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_PATH}")
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list) or not data:
        raise ValueError("sample_questions.json must be a non-empty JSON list.")
    return data


def judge_answer(question, expected, actual):
    """Use the LLM judge to classify the actual answer's accuracy.

    Falls back to 'Inaccurate' on any error so evaluation never crashes.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set; cannot run the LLM judge.")

    client = Groq(api_key=api_key)
    user_message = (
        f"QUESTION: {question}\n\n"
        f"EXPECTED ANSWER: {expected}\n\n"
        f"ACTUAL ANSWER: {actual}\n\n"
        "Label:"
    )
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        label = response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[evaluate] WARNING: judge failed for '{question}': {exc}")
        return "Inaccurate"

    # Normalize the label to one of the three valid options.
    for valid in VALID_JUDGMENTS:
        if valid.lower() in label.lower():
            return valid
    return "Inaccurate"


def run_evaluation():
    """Run evaluation over all questions and return a list of result records."""
    questions = load_questions()
    results = []

    for item in questions:
        question = item["question"]
        expected = item.get("expected", "")

        print(f"[evaluate] Running: {question}")

        # Retrieve chunks separately so we can report them (the ask() pipeline
        # uses the same retrieval under the hood).
        chunks = _retrieve.retrieve(question, k=5)
        result = _query.ask(question)
        actual = result["answer"]

        judgment = judge_answer(question, expected, actual)

        results.append(
            {
                "question": question,
                "expected": expected,
                "retrieved": [
                    {"source": c["source"], "distance": c["distance"]} for c in chunks
                ],
                "actual": actual,
                "sources": result["sources"],
                "judgment": judgment,
            }
        )

    return results


def _md_escape(text):
    """Escape pipe characters and newlines so text fits inside a markdown cell."""
    return text.replace("\n", " ").replace("|", "\\|").strip()


def write_report(results, path=REPORT_PATH):
    """Write a markdown evaluation report with a summary table and details."""
    counts = {"Accurate": 0, "Partially Accurate": 0, "Inaccurate": 0}
    for r in results:
        counts[r["judgment"]] = counts.get(r["judgment"], 0) + 1
    total = len(results)

    lines = []
    lines.append("# The Unofficial Guide — Evaluation Report\n")
    lines.append(f"Evaluated **{total}** questions.\n")
    lines.append(
        f"- Accurate: {counts['Accurate']}\n"
        f"- Partially Accurate: {counts['Partially Accurate']}\n"
        f"- Inaccurate: {counts['Inaccurate']}\n"
    )

    # Summary table.
    lines.append("\n## Summary\n")
    lines.append("| # | Question | Expected | Actual | Judgment |")
    lines.append("| - | -------- | -------- | ------ | -------- |")
    for i, r in enumerate(results, start=1):
        lines.append(
            f"| {i} "
            f"| {_md_escape(r['question'])} "
            f"| {_md_escape(r['expected'])} "
            f"| {_md_escape(r['actual'])} "
            f"| {r['judgment']} |"
        )

    # Per-question detail with retrieved chunks.
    lines.append("\n## Detailed Results\n")
    for i, r in enumerate(results, start=1):
        lines.append(f"### {i}. {r['question']}\n")
        lines.append(f"**Judgment:** {r['judgment']}\n")
        lines.append(f"**Expected:** {r['expected']}\n")
        lines.append(f"**Actual:**\n\n{r['actual']}\n")
        lines.append("**Retrieved chunks:**\n")
        for c in r["retrieved"]:
            lines.append(f"- `{c['source']}` (distance {c['distance']})")
        lines.append("")

    report = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"[evaluate] Report written to {path}")
    return report


if __name__ == "__main__":
    eval_results = run_evaluation()
    write_report(eval_results)
    # Print a compact summary to the console.
    for r in eval_results:
        print(f"  [{r['judgment']:>18}] {r['question']}")
