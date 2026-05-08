"""
extractor.py
------------
WHAT THIS FILE DOES:
  Receives raw PDF bytes → returns a list of clean text chunks.

TRIGGER CHAIN INSIDE THIS FILE:
  extract_pdf_text(pdf_bytes)
      └── pdfplumber.open(BytesIO(pdf_bytes))   ← opens PDF from memory, no disk write
              └── for each page → page.extract_text()  ← pdfplumber reads text per page
      └── clean_text(raw_text)                  ← strips noise from the extracted text
      └── chunk_text(clean_text)                ← splits into LLM-friendly pieces
          └── returns List[str]                 ← each string is ~800 tokens

WHY CHUNKS?
  LLMs have a context window limit (~200k tokens for Claude, ~128k for GPT-4).
  A large PDF could be 50,000+ words. Chunking ensures we never overflow.
  We later join the most relevant chunks into the final prompt.
"""

import re
from io import BytesIO
from typing import List

import pdfplumber


# ── STEP 1: Main entry point ──────────────────────────────────────────────────
# Called by main.py → this is what kicks off the entire extraction pipeline.

def extract_pdf_text(pdf_bytes: bytes) -> List[str]:
    """
    INPUT:  raw PDF as bytes (received from FastAPI's UploadFile)
    OUTPUT: list of text chunks, each ~800 tokens long

    WHAT TRIGGERS WHAT INSIDE:
      1. BytesIO wraps the bytes so pdfplumber can read it like a file
         (avoids writing to disk — faster and safer)
      2. pdfplumber opens the "file" and iterates page by page
      3. extract_text() pulls text from each page (handles columns, tables)
      4. Pages joined with newline → one big raw string
      5. clean_text() is called → removes garbage
      6. chunk_text() is called → splits into chunks
      7. Returns chunks list to main.py
    """
    raw_pages: List[str] = []

    # pdfplumber.open() accepts a file-like object (BytesIO), not just file paths.
    # This is the key step — it parses the PDF structure, finds text layers,
    # and makes them accessible page by page.
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages):
            # page.extract_text() is pdfplumber's core method.
            # It reads the PDF's internal text layer (not OCR — that's for scanned PDFs).
            # Returns None if the page has no text (e.g. a page that's just an image).
            text = page.extract_text()

            if text:  # skip blank or image-only pages
                # We tag each page so the LLM knows where content came from.
                # This helps it understand document structure.
                raw_pages.append(f"[Page {page_number + 1}]\n{text}")

    if not raw_pages:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned image — try OCR tools like Adobe Acrobat first."
        )

    # Join all pages into one big string for cleaning
    full_raw_text = "\n\n".join(raw_pages)

    # ── STEP 2: Clean it ──────────────────────────────────────────────────────
    cleaned = clean_text(full_raw_text)

    # ── STEP 3: Chunk it ──────────────────────────────────────────────────────
    chunks = chunk_text(cleaned)

    return chunks


# ── STEP 2: Cleaning function ─────────────────────────────────────────────────
# Triggered by: extract_pdf_text() after all pages are joined.
# Purpose: remove noise that would confuse the LLM (headers, page numbers, etc.)

def clean_text(text: str) -> str:
    """
    INPUT:  raw joined text (messy, full of artifacts)
    OUTPUT: cleaned text string

    PDF extraction is messy. Common problems:
      - "Page 1 of 24" appearing every page
      - Headers like "CONFIDENTIAL | Q3 2024" repeating
      - Excessive blank lines from layout gaps
      - Ligature artifacts like "ﬁ" instead of "fi"
      - Hyphenated words broken across lines: "require-\nment" → "requirement"

    Each re.sub() call is a targeted fix for one class of problem.
    """
    # Fix hyphenated line-breaks: "require-\nment" → "requirement"
    # The PDF text layer breaks words at line ends with hyphens.
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Collapse multiple blank lines into a single blank line.
    # PDFs often have huge gaps between sections.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove common PDF header/footer patterns:
    #   "Page 1", "Page 1 of 24", "1 | Page"
    text = re.sub(r"\bPage\s+\d+(\s+of\s+\d+)?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\s*\|\s*Page\b", "", text, flags=re.IGNORECASE)

    # Fix PDF ligature artifacts — common encoding issues
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    text = text.replace("ﬃ", "ffi").replace("ﬄ", "ffl")

    # Remove lines that are ONLY whitespace or punctuation (layout artifacts)
    lines = text.split("\n")
    lines = [line for line in lines if len(line.strip()) > 2]
    text = "\n".join(lines)

    return text.strip()


# ── STEP 3: Chunking function ─────────────────────────────────────────────────
# Triggered by: extract_pdf_text() after clean_text() returns.
# Purpose: split the full text into bite-sized pieces the LLM can process.

def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 200) -> List[str]:
    """
    INPUT:  cleaned full text string
    OUTPUT: list of overlapping text chunks

    WHY OVERLAP?
      If a sentence spans a chunk boundary, both chunks get part of it.
      Overlap of 200 chars ensures that boundary sentences aren't lost.
      This is the same technique used in RAG (Retrieval-Augmented Generation).

    HOW IT WORKS:
      - We split by paragraph first (double newline) to avoid breaking mid-sentence
      - We keep adding paragraphs to the current chunk until it would exceed chunk_size
      - When full, we save the chunk and start a new one (carrying over `overlap` chars)
      - Result: chunks that respect paragraph boundaries

    chunk_size=3000 characters ≈ ~750 tokens (4 chars per token on average).
    With Claude's 200k context window, we can fit ~60 chunks comfortably.
    """
    # Split by paragraph boundaries first — we never want to split mid-sentence
    paragraphs = text.split("\n\n")

    chunks: List[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # If adding this paragraph keeps us under the limit, add it
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            current_chunk += ("\n\n" if current_chunk else "") + paragraph

        else:
            # Current chunk is full — save it
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Start the new chunk with overlap from the END of the previous chunk.
            # This ensures we don't lose context at the boundary.
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + "\n\n" + paragraph

    # Don't forget the last chunk (loop ends before it's saved)
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ── UTILITY: Summarise what was extracted ─────────────────────────────────────
# Called by main.py to log what the extractor found — useful for debugging.

def extraction_summary(chunks: List[str]) -> dict:
    """Returns a summary dict — main.py logs this to the console."""
    total_chars = sum(len(c) for c in chunks)
    return {
        "chunks": len(chunks),
        "total_characters": total_chars,
        "estimated_tokens": total_chars // 4,
        "avg_chunk_size": total_chars // max(len(chunks), 1),
    }