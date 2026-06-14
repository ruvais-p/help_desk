"""FastAPI server: serves the chat UI and a streaming RAG endpoint."""
from __future__ import annotations

import json

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.DOCUMENTS_DIR.mkdir(exist_ok=True)
    summary = rag.load_index()
    print(
        f"[startup] index ready — {summary['chunks']} chunks "
        f"from {summary['documents']} document(s)"
    )
    if summary["chunks"] == 0:
        print("[startup] NOTE: no document text indexed. Add PDFs to ./documents "
              "and POST /reindex (or restart).")
    yield


app = FastAPI(title="CUSAT Help Desk RAG", lifespan=lifespan)


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Turn] = []


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": config.CHAT_MODEL}


@app.post("/reindex")
def reindex() -> dict:
    """Rebuild the index after adding/removing PDFs (no restart needed)."""
    summary = rag.load_index(force_rebuild=True)
    return summary


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    """Stream the grounded answer as Server-Sent Events."""

    history = [{"role": t.role, "content": t.content} for t in req.history]

    def event_gen():
        try:
            for event in rag.answer_stream(req.message, history):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # safety net
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the frontend (index.html) at "/".
app.mount("/", StaticFiles(directory=str(config.BASE_DIR / "static"), html=True),
          name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
