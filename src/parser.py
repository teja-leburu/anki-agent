"""Document parser — extracts text from PDFs and splits into chunks."""

import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from a PDF, returning a list of pages with metadata."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page_number": i + 1, "text": text.strip()})
    return pages


def chunk_pages(pages: list[dict], max_tokens: int = 800, overlap: int = 100) -> list[dict]:
    """Split pages into semantic chunks roughly bounded by max_tokens (word-level approximation).

    Tries to break at paragraph boundaries (double newlines) to keep ideas together.
    """
    chunks = []
    buffer = ""
    source_pages = []

    for page in pages:
        paragraphs = page["text"].split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            word_count = len(buffer.split()) + len(para.split())
            if word_count > max_tokens and buffer:
                chunks.append({
                    "text": buffer.strip(),
                    "source_pages": list(source_pages),
                })
                # keep tail for overlap
                words = buffer.strip().split()
                buffer = " ".join(words[-overlap:]) + "\n\n" + para
                source_pages = [page["page_number"]]
            else:
                buffer += "\n\n" + para if buffer else para
                if page["page_number"] not in source_pages:
                    source_pages.append(page["page_number"])

    if buffer.strip():
        chunks.append({"text": buffer.strip(), "source_pages": list(source_pages)})

    return chunks
