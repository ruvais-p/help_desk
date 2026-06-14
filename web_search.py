"""Internet retrieval: search the web, fetch pages, and return the most
relevant chunks — scored with the same local embedder used for documents.

Used as a fallback (or blend) when the local PDFs don't answer a question.
Everything returned is real fetched text with a source URL, so answers stay
grounded and citable — never hallucinated.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import urllib3

import config
from ingest import chunk_text, get_embedder

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
STRIP = ["script", "style", "noscript", "nav", "footer", "header", "aside",
         "form", "svg", "iframe", "head"]


def _search(query: str, max_results: int) -> list[dict]:
    """Return [{title, url, snippet}, ...] from web search (no API key).

    Tries multiple backends so an intermittent rate-limit on one doesn't cause
    a false 'not found'.
    """
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS

    backends = ["auto", "html", "lite", "bing", "google"]
    for backend in backends:
        out = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results, backend=backend):
                    url = r.get("href") or r.get("url")
                    if url and "bing.com/aclick" not in url:  # skip ad redirects
                        out.append(
                            {
                                "title": r.get("title", "") or url,
                                "url": url,
                                "snippet": r.get("body", ""),
                            }
                        )
            if out:
                return out
        except Exception as exc:
            print(f"[web] search backend '{backend}' failed: {exc}")
            continue
    print("[web] all search backends returned nothing")
    return []


def _fetch_text(url: str, timeout: int = 12) -> str:
    """Fetch a page and return cleaned main text (best-effort)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.SSLError:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout, verify=False)
            resp.raise_for_status()
        except Exception:
            return ""
    except Exception:
        return ""

    if "html" not in resp.headers.get("Content-Type", "").lower():
        return ""
    resp.encoding = resp.apparent_encoding or resp.encoding

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(resp.text, "lxml")
    for t in soup(STRIP):
        t.decompose()
    root = soup.body or soup
    return re.sub(r"\s+", " ", root.get_text(" ")).strip()


def retrieve_web(query: str, top_k: int | None = None) -> list[dict]:
    """Search + fetch + rank. Returns chunk dicts compatible with the doc
    retriever: {source, url, page, text, kind='web', score}."""
    top_k = top_k or config.TOP_K
    results = _search(query, config.WEB_RESULTS)
    if not results:
        return []

    # Fetch candidate pages CONCURRENTLY (a few extra to cover thin/failed
    # pages), then keep the first WEB_MAX_PAGES that yielded usable text —
    # preserving search-result order so the most relevant hits win. Fetching in
    # parallel turns a worst-case 5×timeout sequential wait into ~one timeout.
    candidates = results[: config.WEB_MAX_PAGES + 4]
    with ThreadPoolExecutor(max_workers=min(len(candidates), 8)) as pool:
        texts = list(pool.map(lambda r: _fetch_text(r["url"]), candidates))

    chunks: list[dict] = []
    pages_used = 0
    for r, text in zip(candidates, texts):
        if pages_used >= config.WEB_MAX_PAGES:
            break
        if not text or len(text) < 200:
            continue
        pages_used += 1
        # Cap chunks per page so one huge page can't dominate.
        for c in chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)[:50]:
            chunks.append(
                {
                    "source": r["title"],
                    "url": r["url"],
                    "page": None,
                    "text": c,
                    "kind": "web",
                }
            )

    if not chunks:
        return []

    embedder = get_embedder()
    vecs = embedder.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64,
    ).astype("float32")
    qv = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")[0]

    scores = vecs @ qv
    k = min(top_k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]

    out = []
    for i in idx:
        item = dict(chunks[int(i)])
        item["score"] = float(scores[int(i)])
        out.append(item)
    return out
