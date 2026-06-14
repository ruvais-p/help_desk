"""Central configuration loaded from environment / .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
INDEX_DIR = BASE_DIR / "index"

# --- OpenRouter (AI layer) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_MODEL = os.getenv("CHAT_MODEL", "openrouter/owl-alpha").strip()
SITE_URL = os.getenv("SITE_URL", "http://localhost:8000")
SITE_NAME = os.getenv("SITE_NAME", "CUSAT Help Desk")

# --- Retrieval ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.30"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "220"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))

# --- Web search (internet fallback) ---
# When True, if the local documents don't answer the question, the system
# searches the web and answers from those results (still grounded + cited).
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
# "fallback" = only search the web when the documents are insufficient.
# "always"   = always blend web results with document results.
WEB_MODE = os.getenv("WEB_MODE", "fallback").strip().lower()
WEB_RESULTS = int(os.getenv("WEB_RESULTS", "10"))         # search hits to consider
WEB_MAX_PAGES = int(os.getenv("WEB_MAX_PAGES", "5"))      # pages actually fetched
# Web text is noisier than curated PDFs, so a lower bar is reasonable.
WEB_RELEVANCE_THRESHOLD = float(os.getenv("WEB_RELEVANCE_THRESHOLD", "0.22"))

# Index files
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
CHUNKS_FILE = INDEX_DIR / "chunks.json"
MANIFEST_FILE = INDEX_DIR / "manifest.json"

# The exact phrase the model is told to emit when the answer is not in the docs.
NOT_FOUND_SENTINEL = "NOT_FOUND"
REFUSAL_MESSAGE = (
    "I can only help with questions about Cochin University of Science and "
    "Technology (CUSAT) and its B.Tech programs — admissions, courses, fees, "
    "placements, and campus details. I couldn't answer that from the available "
    "CUSAT information. Please ask a CUSAT B.Tech–related question."
)
