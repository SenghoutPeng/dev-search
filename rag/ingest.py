"""
Ingestion: load raw documents from disk and split them into retrievable chunks.

Supports .txt, .md, and .pdf files. Chunking is sentence-aware: text is split
into sentences, then sentences are packed into chunks of roughly `chunk_size`
words each, with a little overlap so context isn't lost at a chunk boundary.
"""

import os
import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    text: str


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _read_pdf_file(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


LOADERS = {
    ".txt": _read_text_file,
    ".md": _read_text_file,
    ".pdf": _read_pdf_file,
}


def load_documents(folder: str) -> List[dict]:
    """Load every supported file in `folder` into {"title": ..., "text": ...} dicts."""
    docs = []
    for filename in sorted(os.listdir(folder)):
        ext = os.path.splitext(filename)[1].lower()
        loader = LOADERS.get(ext)
        if loader is None:
            continue
        path = os.path.join(folder, filename)
        text = loader(path)
        if not text:
            continue
        title = os.path.splitext(filename)[0].replace("_", " ").title()
        docs.append({"title": title, "text": text})
    return docs


def split_sentences(text: str) -> List[str]:
    """Split on sentence-ending punctuation or blank lines (paragraph breaks)."""
    pieces = re.split(r"(?<=[.!?])\s+|\n\s*\n", text)
    return [p.strip() for p in pieces if p.strip()]


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> List[str]:
    """Pack sentences into chunks of roughly `chunk_size` words, with `overlap`
    words repeated at the start of the next chunk for continuity."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_words: List[str] = []
    for sentence in sentences:
        current_words.extend(sentence.split())
        if len(current_words) >= chunk_size:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap:] if overlap else []
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def build_chunk_records(docs: List[dict], chunk_size: int = 80, overlap: int = 20) -> List[Chunk]:
    """Turn loaded documents into a flat list of Chunk records ready for embedding."""
    records = []
    for doc in docs:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            records.append(Chunk(chunk_id=f"{doc['title']}::{i}", doc_title=doc["title"], text=piece))
    return records


if __name__ == "__main__":
    docs = load_documents("data/git_docs")
    chunks = build_chunk_records(docs)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks "
          f"({len(chunks) / len(docs):.1f} chunks/doc avg)")
