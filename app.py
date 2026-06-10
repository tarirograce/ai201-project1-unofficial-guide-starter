"""
FILE: app.py

Gradio web interface for "The Unofficial Guide".

A simple, demo-ready UI built with Gradio Blocks:
  * Input:  a question textbox
  * Output: an answer textbox and a sources textbox
  * Submit button AND press-Enter-to-submit
  * On first launch, automatically builds the vector index if it is empty

Run with:  python app.py
"""

import os
import sys

# Make the src/ modules importable when running from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import gradio as gr

import embed
from query import ask


def ensure_index():
    """Build the vector index on startup if it hasn't been built yet."""
    collection = embed.get_collection()
    if collection.count() == 0:
        print("[app] Vector store is empty — building the index now...")
        embed.build_index()
    else:
        print(f"[app] Vector store ready ({collection.count()} chunks indexed).")


def answer_question(question):
    """Gradio callback: take a question, return (answer_text, sources_text)."""
    if not question or not question.strip():
        return "Please enter a question.", ""

    try:
        result = ask(question)
    except Exception as exc:
        # Show the error in the UI instead of crashing the app.
        return f"Error: {exc}", ""

    answer = result["answer"]
    sources = result["sources"]
    sources_text = "\n".join(f"- {s}" for s in sources) if sources else "(no sources)"
    return answer, sources_text


def build_ui():
    """Construct and return the Gradio Blocks interface."""
    with gr.Blocks(title="The Unofficial Guide") as demo:
        gr.Markdown(
            "# 🎓 The Unofficial Guide\n"
            "Ask anything about life at Northgate University. Answers are grounded "
            "**only** in student-written guide documents — if the answer isn't in "
            "them, the assistant will say so."
        )

        with gr.Row():
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What is the best dorm for first-year students?",
                lines=2,
                autofocus=True,
            )

        submit_btn = gr.Button("Ask", variant="primary")

        answer_box = gr.Textbox(label="Answer", lines=8, interactive=False)
        sources_box = gr.Textbox(label="Sources", lines=4, interactive=False)

        gr.Examples(
            examples=[
                "What is the best dorm for first-year students?",
                "Where can I find a quiet place to study?",
                "How do I get downtown without a car?",
                "Which dining hall has the best vegetarian options?",
            ],
            inputs=question_box,
        )

        # Wire up both the button click and Enter-to-submit (textbox submit).
        submit_btn.click(
            fn=answer_question,
            inputs=question_box,
            outputs=[answer_box, sources_box],
        )
        question_box.submit(
            fn=answer_question,
            inputs=question_box,
            outputs=[answer_box, sources_box],
        )

    return demo


if __name__ == "__main__":
    ensure_index()
    ui = build_ui()
    ui.launch()
