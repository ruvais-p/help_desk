"""Retrieval + grounded answering.

Two anti-hallucination guards:
  1. Pre-LLM gate: if the best retrieved chunk scores below RELEVANCE_THRESHOLD,
     we refuse immediately without calling the model.
  2. In-prompt grounding: the system prompt forbids outside knowledge and tells
     the model to emit NOT_FOUND when the context doesn't contain the answer.
"""
from __future__ import annotations

import json
import re
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


SYSTEM_PROMPT = f"""You are the CUSAT Help Desk assistant. You answer questions \
ONLY about Cochin University of Science and Technology (CUSAT) and its B.Tech \
programs — including admissions, the CUSAT CAT, courses and branches, fee \
structure, placements, campuses (e.g. SOE, CUCEK), and related student \
information. You answer ONLY using the context passages provided by the user. \
The context may contain excerpts from local documents AND from web pages. \
Follow these rules strictly:

0. SCOPE: Only answer questions about CUSAT and its B.Tech programs. If the \
question is NOT about CUSAT or its B.Tech programs (for example: general \
knowledge, other universities, weather, coding help, current events, or \
anything unrelated), reply with EXACTLY this token and nothing else, \
REGARDLESS of what the CONTEXT contains: {config.NOT_FOUND_SENTINEL}
1. Use ONLY information found in the CONTEXT. Never use prior knowledge or assumptions.
2. If the CONTEXT does not contain enough information to answer an in-scope \
question, reply with EXACTLY this token and nothing else: {config.NOT_FOUND_SENTINEL}
3. Do not invent facts, numbers, names, dates, or sources. No guessing.
4. When you answer, be clear and concise, and ground every statement in the context.
5. Prefer document sources when they conflict with web sources, but you may use \
web sources to fill gaps the documents don't cover.
6. If you can answer PART of an in-scope question, answer that part and briefly \
state in plain words what information is not available. Only output the \
{config.NOT_FOUND_SENTINEL} token when you cannot answer ANY part — never mix it \
into an answer.
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


# CUSAT B.Tech branch / domain abbreviations → full forms. Used to enrich the
# RETRIEVAL + web-search query (never the answer prompt) so short forms like
# "cs" or "ee" match the full programme names written in the PDFs.
_ABBREVIATIONS = {
    "cs": "computer science and engineering",
    "cse": "computer science and engineering",
    "ee": "electrical and electronics engineering",
    "eee": "electrical and electronics engineering",
    "ec": "electronics and communication engineering",
    "ece": "electronics and communication engineering",
    "it": "information technology",
    "me": "mechanical engineering",
    "ce": "civil engineering",
    "sfe": "safety and fire engineering",
    "na": "naval architecture and ship building",
    "prt": "polymer science and rubber technology",
    "ic": "instrumentation and control engineering",
    "ice": "instrumentation and control engineering",
    "cat": "cusat common admission test",
    "soe": "school of engineering",
    "cucek": "cochin university college of engineering kuttanad",
    "nri": "non-resident indian quota",
}
# Short forms that are also ordinary English words ("it", "me", ...): only
# expand these when the query has clear academic context, to avoid noise.
_AMBIGUOUS = {"it", "me", "ce", "ic", "na", "cat"}
_CONTEXT_WORDS = re.compile(
    r"\b(b\.?tech|branch(?:es)?|engineering|course|courses|stream|department|"
    r"programme|program|compare|comparison|vs|versus|admission|admissions|"
    r"fee|fees|placement|placements|exam|test|syllabus|scholarship|hostel|"
    r"salary|seats?|rank|cutoff|cusat)\b",
    re.I,
)


def _expand_query(text: str) -> str:
    """Append full forms of any branch abbreviations found in `text` so the
    embedding/web search matches the programme names used in the documents.
    Originals are kept; expansions are added in parentheses."""
    has_ctx = bool(_CONTEXT_WORDS.search(text))
    extra, seen = [], set()
    for ab, full in _ABBREVIATIONS.items():
        if ab in _AMBIGUOUS and not has_ctx:
            continue
        if full in seen:
            continue
        if re.search(rf"\b{re.escape(ab)}\b", text, re.I):
            extra.append(full)
            seen.add(full)
    return f"{text} ({'; '.join(extra)})" if extra else text


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


def _in_scope(question: str) -> bool:
    """Cheap LLM classifier: is this about CUSAT / its B.Tech programs?

    Used to skip the (slow) web fallback for clearly off-topic questions, so
    they refuse instantly instead of searching + fetching several web pages.
    Fails OPEN (returns True) on any error — the system prompt's SCOPE rule
    still refuses off-topic answers at generation time, so a missed check
    costs latency, never correctness.
    """
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": config.SITE_URL,
                "X-Title": config.SITE_NAME,
            },
            model=config.CHAT_MODEL,
            temperature=0,
            max_tokens=4,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a topic filter for the CUSAT (Cochin University "
                        "of Science and Technology) B.Tech Help Desk. The user is "
                        "ALREADY assumed to be asking about CUSAT, so treat "
                        "anything about B.Tech study as IN SCOPE even if it does "
                        "not say 'CUSAT' — this includes admissions, the CUSAT "
                        "CAT, any branch or comparison of branches (e.g. Computer "
                        "Science/CS, Electrical/EE, Naval Architecture, Polymer "
                        "Science), courses, syllabus, fees, scholarships, "
                        "placements, salaries, recruiters, campuses such as "
                        "SOE/CUCEK, hostels, and student life. Reply N ONLY for "
                        "clearly unrelated topics (weather, sports, general "
                        "programming help, other universities, world news, "
                        "homework). Reply with exactly one character: Y (in "
                        "scope) or N (out of scope). Output nothing else."
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        verdict = (resp.choices[0].message.content or "").strip().upper()
        return not verdict.startswith("N")
    except Exception as exc:
        print(f"[scope] check failed, allowing through: {exc}")
        return True


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


def _drain(buf: str, sentinel: str) -> tuple[str, str]:
    """Strip any complete sentinel from `buf` and hold back the longest trailing
    fragment that could still be the START of a sentinel split across tokens
    (e.g. 'NOT_' before 'FOUND' arrives). Returns (safe_to_emit, held_tail)."""
    buf = buf.replace(sentinel, "")
    for k in range(min(len(buf), len(sentinel) - 1), 0, -1):
        if sentinel.startswith(buf[-k:]):
            return buf[:-k], buf[-k:]
    return buf, ""


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
    carry = ""         # holds a possible split sentinel fragment between tokens
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
                # Even after committing, scrub any sentinel the model mixes in.
                emit, carry = _drain(buffer, config.NOT_FOUND_SENTINEL)
                if emit:
                    yield {"type": "token", "text": emit}
                buffer = ""
            else:
                emit, carry = _drain(carry + delta, config.NOT_FOUND_SENTINEL)
                if emit:
                    yield {"type": "token", "text": emit}
    except Exception as exc:
        yield {"type": "error", "message": f"Streaming error: {exc}"}
        return "error"

    # Flush any held fragment that never turned out to be a sentinel.
    tail = carry.replace(config.NOT_FOUND_SENTINEL, "")
    if committed and tail:
        yield {"type": "token", "text": tail}

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

    # Resolve follow-up references for retrieval and expand branch
    # abbreviations (cs -> computer science, ...). The LLM still gets the raw
    # question plus the conversation history for phrasing.
    search_query = _expand_query(_contextualize(question, history))

    doc_chunks = retrieve(search_query)
    doc_relevant = sorted(
        [c for c in doc_chunks if c["score"] >= config.RELEVANCE_THRESHOLD],
        key=lambda c: c["score"],
        reverse=True,
    )[: config.TOP_K]

    always = config.ENABLE_WEB_SEARCH and config.WEB_MODE == "always"

    # --- "always" mode: blend documents + web in a single attempt ---
    if always:
        # Skip the slow web fetch for off-topic questions (instant refusal).
        if not _in_scope(search_query):
            yield {"type": "sources", "sources": []}
            yield {"type": "refusal", "message": config.REFUSAL_MESSAGE}
            return
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
        # Gate the slow web path: off-topic questions (no relevant docs) refuse
        # immediately instead of searching + fetching several pages first.
        if not _in_scope(search_query):
            yield {"type": "sources", "sources": []}
            yield {"type": "refusal", "message": config.REFUSAL_MESSAGE}
            return
        yield {"type": "status", "message": "Searching the web…"}
        web_relevant = _safe_web(search_query)
        if web_relevant:
            result = yield from _attempt(question, web_relevant, history)
            if result in ("answered", "error"):
                return

    # Nothing in documents or on the web answered it.
    yield {"type": "sources", "sources": []}
    yield {"type": "refusal", "message": config.REFUSAL_MESSAGE}
