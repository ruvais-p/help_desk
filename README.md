# CUSAT Help Desk — Document RAG

A retrieval-augmented chat system that answers questions **only** from your PDF
documents. If the answer isn't in the documents, it refuses instead of
hallucinating. Streaming chat UI included. No chat history, no authentication.

## How it works

```
PDFs (documents/) ──► chunk ──► local embeddings ──► vector index (index/)
                                                          │
        question ──► embed ──► cosine search ──► top-K relevant chunks
                                                          │
                          ┌── best score < threshold ──► REFUSE (no LLM call)
                          │
                          └── relevant chunks ──► OpenRouter LLM (grounded prompt)
                                                          │
                                          answer  OR  NOT_FOUND ──► REFUSE
```

**Document-first, internet-fallback:**
1. It tries to answer from your **local PDFs** first.
2. If the model says the answer isn't in the documents (or no relevant document
   was found), it **searches the web** (DuckDuckGo, no API key), fetches the top
   pages, and answers from those — with clickable source links.
3. If neither the documents nor the web contain the answer, it **refuses**.

**Anti-hallucination guards (still apply to web answers):**
- **Grounded prompt** — the model uses *only* the supplied context (documents
  *or* fetched web text) and emits `NOT_FOUND` when the answer isn't there.
  Temperature is `0`.
- **Refusal leaks nothing** — the start of every response is buffered to detect
  `NOT_FOUND` before any text reaches the user.
- Every answer cites its sources (file + page, or web URL).

Set `WEB_MODE=always` to blend web results with documents on every query, or
`ENABLE_WEB_SEARCH=false` to go back to documents-only.

Embeddings run **locally** (`sentence-transformers`) — free, offline, no extra
API key. Only the final answer uses OpenRouter.

## Setup

```bash
# 1. Install dependencies (a virtualenv is recommended)
pip install -r requirements.txt

# 2. Configure your key
cp .env.example .env
#   then edit .env and set OPENROUTER_API_KEY (and CHAT_MODEL if you like)

# 3. Add your PDFs
cp /path/to/*.pdf documents/

# 4. Build the index (optional — also runs automatically on first start)
python ingest.py

# 5. Run the server
python app.py
```

Open **http://localhost:8000** and start chatting.

## Adding / updating documents

Drop or remove PDFs in `documents/`, then either restart the server or hit the
reindex endpoint (no restart needed):

```bash
curl -X POST http://localhost:8000/reindex
```

The index is cached and only rebuilt when the PDF set changes.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | — | **Required.** Your OpenRouter key. |
| `CHAT_MODEL` | `openrouter/owl-alpha` | Any OpenRouter model id. |
| `TOP_K` | `5` | Chunks fed to the model. |
| `RELEVANCE_THRESHOLD` | `0.30` | Min cosine score to be "relevant". Raise it to be stricter (refuse more), lower it to be more permissive. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `220` / `40` | Chunking in words. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model. |

## Endpoints

- `GET  /` — chat UI
- `POST /chat` — `{ "message": "..." }` → streamed SSE answer + sources
- `POST /reindex` — rebuild the index after changing PDFs
- `GET  /health` — status

## Notes

- Scanned/image-only PDFs have no extractable text. Add OCR (e.g. `ocrmypdf`)
  first if your PDFs are scans.
- Tune `RELEVANCE_THRESHOLD` to your documents: too many false refusals → lower
  it; answers leaking outside the docs → raise it.
