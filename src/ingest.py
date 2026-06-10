"""
FILE: src/ingest.py

Document ingestion + cleaning pipeline for "The Unofficial Guide".

Responsibilities:
  * Load every .txt file from data/raw/
  * Clean the text (strip HTML, navigation menus, cookie banners,
    boilerplate, and collapse extra whitespace) while preserving real content
  * Write the cleaned text back out to data/processed/
  * Return a list of {"source": filename, "text": cleaned_text} dicts that the
    rest of the pipeline (chunk -> embed) consumes

This module is intentionally dependency-light (standard library only) so the
cleaning logic is easy to read and explain during a demo.
"""

import os
import re
import html

# ---------------------------------------------------------------------------
# Paths are resolved relative to the project root so the module works no matter
# the current working directory (running from root or from src/).
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

# Lines that look like navigation, cookie banners, or boilerplate. If a line
# matches one of these (case-insensitive), it is dropped during cleaning. These
# patterns are deliberately conservative so we never delete real content.
BOILERPLATE_PATTERNS = [
    r"^\s*home\s*\|\s*about\s*\|",            # nav bars like "Home | About | Contact"
    r"cookie",                                 # cookie banners
    r"accept all cookies",
    r"we use cookies",
    r"^\s*menu\s*$",                           # standalone "Menu"
    r"^\s*navigation\s*$",
    r"all rights reserved",                    # footer boilerplate
    r"^\s*copyright\b",
    r"subscribe to our newsletter",
    r"follow us on",
    r"skip to (main )?content",
    r"^\s*share this\b",
]
_BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

# Matches any HTML/XML tag, e.g. <div class="x"> or </p>
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text):
    """Clean a single raw document string and return readable plain text.

    Steps:
      1. Unescape HTML entities (&amp; -> &) and strip HTML tags.
      2. Drop lines that match navigation / cookie / boilerplate patterns.
      3. Collapse runs of whitespace and blank lines.
    """
    # 1. HTML: unescape entities first, then remove tags.
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)

    cleaned_lines = []
    for line in text.splitlines():
        # 2. Skip boilerplate / navigation / cookie lines.
        if _BOILERPLATE_RE.search(line):
            continue
        # Collapse internal runs of whitespace within the line.
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    # 3. Rejoin and collapse 3+ consecutive blank lines down to a single blank
    #    line, preserving paragraph structure (which helps chunking).
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def load_raw_documents():
    """Read every .txt file in data/raw/ and return a list of (filename, raw_text)."""
    if not os.path.isdir(RAW_DIR):
        raise FileNotFoundError(
            f"Raw data directory not found: {RAW_DIR}. "
            "Create it and add .txt documents before running ingestion."
        )

    documents = []
    for filename in sorted(os.listdir(RAW_DIR)):
        if not filename.lower().endswith(".txt"):
            continue
        path = os.path.join(RAW_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw_text = fh.read()
        except (OSError, UnicodeDecodeError) as exc:
            # Don't let one bad file kill the whole run; report and continue.
            print(f"[ingest] WARNING: could not read {filename}: {exc}")
            continue
        documents.append((filename, raw_text))
    return documents


def run(save=True):
    """Run the full ingestion pipeline.

    Returns a list of {"source": filename, "text": cleaned_text} dicts and,
    when ``save`` is True, also writes the cleaned text to data/processed/.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    raw_documents = load_raw_documents()
    if not raw_documents:
        raise RuntimeError(
            f"No .txt files found in {RAW_DIR}. Add documents and try again."
        )

    processed = []
    for filename, raw_text in raw_documents:
        cleaned = clean_text(raw_text)
        if not cleaned:
            print(f"[ingest] WARNING: {filename} was empty after cleaning, skipping.")
            continue

        # Attach metadata exactly as the spec requires.
        processed.append({"source": filename, "text": cleaned})

        if save:
            out_path = os.path.join(PROCESSED_DIR, filename)
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(cleaned)

    print(f"[ingest] Processed {len(processed)} documents -> {PROCESSED_DIR}")
    return processed


if __name__ == "__main__":
    docs = run()
    for d in docs:
        print(f"  - {d['source']}: {len(d['text'])} chars")
