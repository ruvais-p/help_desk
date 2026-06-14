"""Build the vector index from PDFs in the documents/ folder.

Run directly:  python ingest.py
Or import and call build_index() from the app.

The index is cached on disk and only rebuilt when the set of PDFs
(or their modification times) changes, so startup stays fast.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import List

import numpy as np
from pypdf import PdfReader

import config

_model = None  # lazily loaded SentenceTransformer


def get_embedder():
    """Load the embedding model once and reuse it."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        print(f"[ingest] loading embedding model: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(path: Path) -> List[tuple[int, str]]:
    """Return [(page_number, text), ...] for a single PDF."""
    pages = []
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # corrupted / unreadable file
        print(f"[ingest] WARNING: could not read {path.name}: {exc}")
        return pages
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = _clean(page.extract_text() or "")
        except Exception:
            text = ""
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if window:
            chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def _manifest_signature() -> dict:
    """Fingerprint the documents folder so we know when to rebuild."""
    sig = {}
    for pdf in sorted(config.DOCUMENTS_DIR.glob("*.pdf")):
        stat = pdf.stat()
        sig[pdf.name] = {"size": stat.st_size, "mtime": int(stat.st_mtime)}
    return sig


def _signature_hash(sig: dict) -> str:
    return hashlib.sha256(
        json.dumps(sig, sort_keys=True).encode("utf-8")
    ).hexdigest()


def index_is_current() -> bool:
    if not (config.EMBEDDINGS_FILE.exists() and config.CHUNKS_FILE.exists()
            and config.MANIFEST_FILE.exists()):
        return False
    try:
        manifest = json.loads(config.MANIFEST_FILE.read_text())
    except Exception:
        return False
    current = _signature_hash(_manifest_signature())
    return manifest.get("signature") == current and manifest.get(
        "embedding_model"
    ) == config.EMBEDDING_MODEL


def build_index(force: bool = False) -> dict:
    """Parse PDFs, embed chunks, persist the index. Returns a small summary."""
    config.DOCUMENTS_DIR.mkdir(exist_ok=True)
    config.INDEX_DIR.mkdir(exist_ok=True)

    if not force and index_is_current():
        chunks = json.loads(config.CHUNKS_FILE.read_text())
        return {
            "rebuilt": False,
            "chunks": len(chunks),
            "documents": len({c["source"] for c in chunks}),
        }

    pdfs = sorted(config.DOCUMENTS_DIR.glob("*.pdf"))
    records: List[dict] = []
    for pdf in pdfs:
        for page_no, text in extract_pdf_pages(pdf):
            for chunk in chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                records.append(
                    {"source": pdf.name, "page": page_no, "text": chunk}
                )

    if not records:
        # Persist an empty index so the app can still start and answer "not found".
        np.save(config.EMBEDDINGS_FILE, np.zeros((0, 384), dtype="float32"))
        config.CHUNKS_FILE.write_text(json.dumps([], ensure_ascii=False))
        config.MANIFEST_FILE.write_text(
            json.dumps(
                {
                    "signature": _signature_hash(_manifest_signature()),
                    "embedding_model": config.EMBEDDING_MODEL,
                }
            )
        )
        print("[ingest] no readable text found in documents/ — empty index written")
        return {"rebuilt": True, "chunks": 0, "documents": 0}

    embedder = get_embedder()
    print(f"[ingest] embedding {len(records)} chunks from {len(pdfs)} PDF(s)...")
    embeddings = embedder.encode(
        [r["text"] for r in records],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # so dot product == cosine similarity
        convert_to_numpy=True,
    ).astype("float32")

    np.save(config.EMBEDDINGS_FILE, embeddings)
    config.CHUNKS_FILE.write_text(json.dumps(records, ensure_ascii=False))
    config.MANIFEST_FILE.write_text(
        json.dumps(
            {
                "signature": _signature_hash(_manifest_signature()),
                "embedding_model": config.EMBEDDING_MODEL,
            }
        )
    )
    print(f"[ingest] index built: {len(records)} chunks")
    return {
        "rebuilt": True,
        "chunks": len(records),
        "documents": len(pdfs),
    }


if __name__ == "__main__":
    summary = build_index(force=True)
    print(summary)
