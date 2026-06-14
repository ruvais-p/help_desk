"""Retrieval + grounded answering.

Two anti-hallucination guards:
  1. Pre-LLM gate: if the best retrieved chunk scores below RELEVANCE_THRESHOLD,
     we refuse immediately without calling the model.
  2. In-prompt grounding: the system prompt forbids outside knowledge and tells
     the model to emit NOT_FOUND when the context doesn't contain the answer.
"""
from __future__ import annotations

import json
from typing import Iterator, List

import numpy as np
from openai import OpenAI

import config
from ingest import build_index, get_embedder
from web_search import retrieve_web

# --- module-level state (loaded once) ---
_embeddings: np.ndarray | None = None
_chunks: List[dict] | None = None
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )
    return _client


def load_index(force_rebuild: bool = False) -> dict:
    """Ensure the index exists, then load it into memory."""
    global _embeddings, _chunks
    summary = build_index(force=force_rebuild)
    _embeddings = np.load(config.EMBEDDINGS_FILE)
    _chunks = json.loads(config.CHUNKS_FILE.read_text())
    return summary


def _ensure_loaded() -> None:
    if _embeddings is None or _chunks is None:
        load_index()


def retrieve(question: str, top_k: int | None = None) -> List[dict]:
    """Return the most relevant chunks with a cosine 'score' field, best first."""
    _ensure_loaded()
    top_k = top_k or config.TOP_K
    if _embeddings is None or len(_chunks) == 0:
        return []

    q_vec = get_embedder().encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")[0]

    scores = _embeddings @ q_vec  # cosine similarity (vectors are normalized)
    k = min(top_k, len(scores))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    results = []
    for i in top_idx:
        item = dict(_chunks[int(i)])
        item["score"] = float(scores[int(i)])
        item["kind"] = "doc"
        item["url"] = None
        results.append(item)
    return results


SYSTEM_PROMPT = f"""You are a precise assistant. You answer ONLY using the \
context passages provided by the user. The context may contain excerpts from \
local documents AND from web pages. Follow these rules strictly:

1. Use ONLY information found in the CONTEXT. Never use prior knowledge or assumptions.
2. If the CONTEXT does not contain enough information to answer, reply with EXACTLY \
this token and nothing else: {config.NOT_FOUND_SENTINEL}
3. Do not invent facts, numbers, names, dates, or sources. No guessing.
4. When you answer, be clear and concise, and ground every statement in the context.
5. Prefer document sources when they conflict with web sources, but you may use \
web sources to fill gaps the documents don't cover.
"""


def _build_context(chunks: List[dict]) -> str:
    blocks = []
    for idx, c in enumerate(chunks, start=1):
        if c.get("kind") == "web":
            head = f"[Source {idx}] (web page: {c.get('url')})"
        else:
            head = f"[Source {idx}] (file: {c['source']}, page: {c['page']})"
        blocks.append(f"{head}\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def _sources_payload(chunks: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for c in chunks:
        key = c.get("url") or (c["source"], c["page"])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "source": c["source"],
                "page": c["page"],
                "url": c.get("url"),
                "kind": c.get("kind", "doc"),
                "score": round(c["score"], 3),
                "snippet": c["text"][:240] + ("…" if len(c["text"]) > 240 else ""),
            }
        )
    return out


def _contextualize(question: str, history: List[dict] | None) -> str:
    """Build a standalone retrieval query from a possibly-elliptical follow-up
    by prepending the last couple of user turns (e.g. 'what about fees?' ->
    'B.Tech programmes what about fees?')."""
    if not history:
        return question
    prev_users = [
        h["content"] for h in history
        if h.get("role") == "user" and h.get("content")
    ][-2:]
    if not prev_users:
        return question
    return " ".join(prev_users + [question]).strip()


def _safe_web(question: str) -> List[dict]:
    """Web retrieval that never raises; returns relevant chunks only."""
    if not config.ENABLE_WEB_SEARCH:
        return []
    try:
        web = retrieve_web(question)
    except Exception as exc:
        print(f"[web] retrieval error: {exc}")
        return []
    web = [c for c in web if c["score"] >= config.WEB_RELEVANCE_THRESHOLD]
    return sorted(web, key=lambda c: c["score"], reverse=True)[: config.TOP_K]


def _attempt(question: str, chunks: List[dict],
             history: List[dict] | None = None) -> Iterator[dict]:
    """Try to answer from `chunks`. Buffers the start of the response to detect
    the NOT_FOUND sentinel BEFORE showing anything, so a refusal leaks nothing.

    Emits sources + tokens + done on success. On a grounded refusal it emits
    NOTHING and returns 'refused' (via StopIteration value) so the caller can
    fall back to another source. Returns 'answered' or 'error' otherwise.
    """
    context = _build_context(chunks)
    user_msg = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using only the CONTEXT above. If the answer is not in the "
        f"CONTEXT, reply with {config.NOT_FOUND_SENTINEL}."
    )
    sources = _sources_payload(chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Recent conversation (for resolving follow-up references), trimmed.
    for h in (history or [])[-10:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"][:4000]})
    messages.append({"role": "user", "content": user_msg})

    try:
        client = _get_client()
        stream = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": config.SITE_URL,
                "X-Title": config.SITE_NAME,
            },
            extra_body={},
            model=config.CHAT_MODEL,
            temperature=0,
            messages=messages,
            stream=True,
        )
    except Exception as exc:
        yield {"type": "error", "message": f"AI service error: {exc}"}
        return "error"

    buffer = ""
    committed = False  # have we decided this is a real answer (not a refusal)?
    try:
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            buffer += delta
            if not committed:
                stripped = buffer.strip()
                if config.NOT_FOUND_SENTINEL in stripped and len(stripped) <= len(
                    config.NOT_FOUND_SENTINEL
                ) + 3:
                    return "refused"  # emit nothing; caller may try the web
                if len(stripped) < len(config.NOT_FOUND_SENTINEL) + 2:
                    continue
                committed = True
                yield {"type": "sources", "sources": sources}
                yield {"type": "token", "text": buffer}
                buffer = ""
            else:
                yield {"type": "token", "text": delta}
    except Exception as exc:
        yield {"type": "error", "message": f"Streaming error: {exc}"}
        return "error"

    # Very short answer that never crossed the sentinel-length check.
    if not committed and buffer.strip():
        if config.NOT_FOUND_SENTINEL in buffer.strip():
            return "refused"
        yield {"type": "sources", "sources": sources}
        yield {"type": "token", "text": buffer}

    yield {"type": "done"}
    return "answered"


def answer_stream(question: str,
                  history: List[dict] | None = None) -> Iterator[dict]:
    """Document-first, internet-fallback answering with conversational context.

    Flow: try local documents -> if the model says the answer isn't there
    (or there were no relevant docs) -> search the web and try again ->
    if neither has it, refuse. Nothing is ever invented.
    """
    question = (question or "").strip()
    if not question:
        yield {"type": "error", "message": "Empty question."}
        return

    # Resolve follow-up references for retrieval (the LLM still gets the raw
    # question plus the conversation history for phrasing).
    search_query = _contextualize(question, history)

    doc_chunks = retrieve(search_query)
    doc_relevant = sorted(
        [c for c in doc_chunks if c["score"] >= config.RELEVANCE_THRESHOLD],
        key=lambda c: c["score"],
        reverse=True,
    )[: config.TOP_K]

    always = config.ENABLE_WEB_SEARCH and config.WEB_MODE == "always"

    # --- "always" mode: blend documents + web in a single attempt ---
    if always:
        web_relevant = _safe_web(search_query)
        combined = sorted(
            doc_relevant + web_relevant, key=lambda c: c["score"], reverse=True
        )[: config.TOP_K + config.WEB_MAX_PAGES]
        if not combined:
            yield {"type": "sources", "sources": []}
            yield {"type": "refusal", "message": config.REFUSAL_MESSAGE}
            return
        result = yield from _attempt(question, combined, history)
        if result == "refused":
            yield {"type": "sources", "sources": []}
            yield {"type": "refusal", "message": config.REFUSAL_MESSAGE}
        return

    # --- fallback mode: documents first ---
    if doc_relevant:
        result = yield from _attempt(question, doc_relevant, history)
        if result in ("answered", "error"):
            return
        # 'refused' -> documents don't contain it; fall through to the web.

    # --- web fallback ---
    if config.ENABLE_WEB_SEARCH:
        yield {"type": "status", "message": "Searching the web…"}
        web_relevant = _safe_web(search_query)
        if web_relevant:
            result = yield from _attempt(question, web_relevant, history)
            if result in ("answered", "error"):
                return

    # Nothing in documents or on the web answered it.
    yield {"type": "sources", "sources": []}
    yield {"type": "refusal", "message": config.REFUSAL_MESSAGE}
