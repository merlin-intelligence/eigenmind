"""Single source of truth for constants and env-based secrets.

All other modules read from here instead of redefining their own constants.
Secrets (HF_TOKEN, NEBIUS_API_KEY...) are read from environment variables only —
never hard-coded.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import pytesseract  # noqa: F401
    from pdf2image import convert_from_path  # noqa: F401
    _OCR_LIBS_AVAILABLE = True
except ImportError:
    _OCR_LIBS_AVAILABLE = False

if load_dotenv is not None:
    load_dotenv(override=False)


# ── Embedding ──
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIM_DEFAULT = 768

# ── Chunking ──
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
BATCH_SIZE = 100
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md")

# ── Graph exploration ──
MAX_CHUNKS = 100
MAX_CHUNKS_FOR_CONTEXT = 30
NEIGHBORS_TO_FETCH = 5
SIMILARITY_THRESHOLD = 0.65

# ── Theta (conflict-graph) parameters ──
THETA_CONFLICT_THRESHOLD = SIMILARITY_THRESHOLD
THETA_RANK = 24
THETA_MAX_ITERS = 400
THETA_STEP0 = 0.25
THETA_DIAG_SHIFT = 1e-2

# ── Local LLM (used by eigenmind.core.llm) ──
LLM_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
MAX_CONTEXT_LENGTH = 4096

# ── Chat-page LLM backend ──
# The /ask/ page can answer either through the Nebius / AI Hub cloud API or a
# local Ollama server. The provider is selected with the LLM_PROVIDER env var.
DEFAULT_LLM_PROVIDER = "nebius"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
NEBIUS_MODELS = (
    "meta-llama/Llama-3.3-70B-Instruct",
    "moonshotai/Kimi-K2.5-fast",
    "openai/gpt-oss-120b",
)

# ── Visualization ──
PROMPT_NODE_COLOR = "#FF0000"
NODE_BASE_SIZE = 10
NODE_SIZE_MULTIPLIER = 3
DOCUMENT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
METHOD_COLORS = {
    "Singular": "#FFD700",
    "Hinge":    "#32CD32",
    "Theta":    "#1E90FF",
}

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
USER_DATA_DIR = PROJECT_ROOT / "user_data"
DOWNLOADED_CORPUS_DIR = PROJECT_ROOT / "downloaded_corpus"
TEMP_GRAPH_OUTPUTS = PROJECT_ROOT / "temp_graph_outputs"


def qdrant_host() -> str:
    return os.getenv("QDRANT_HOST", "localhost")


def qdrant_port() -> int:
    return int(os.getenv("QDRANT_PORT", "6333"))


def hf_token() -> str | None:
    """HuggingFace token from env. Returns None if not set."""
    tok = os.getenv("HF_TOKEN", "").strip()
    return tok or None


def nebius_api_key() -> str:
    """Nebius / AI Hub API key from env (or empty string if not set)."""
    return os.getenv("NEBIUS_API_KEY", "").strip()


def llm_provider() -> str:
    """Active LLM backend for the Chat page: 'ollama' (local) or 'nebius' (cloud).

    Defaults to 'nebius' so the original cloud behaviour is preserved when unset.
    """
    return os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower() or DEFAULT_LLM_PROVIDER


def ollama_host() -> str:
    """Base URL of the local Ollama server (e.g. http://localhost:11434)."""
    return os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).strip() or DEFAULT_OLLAMA_HOST


def ollama_models() -> list[str]:
    """Optional comma-separated fallback model list (e.g. 'qwen2.5:7b,llama3.1').

    Used only when the Ollama server cannot be queried for its installed models.
    """
    raw = os.getenv("OLLAMA_MODELS", "").strip()
    return [m.strip() for m in raw.split(",") if m.strip()]


def sharepoint_credentials() -> tuple[str, str]:
    """SharePoint OAuth credentials from st.secrets if available, else env."""
    client_id = os.getenv("SHAREPOINT_CLIENT_ID", "").strip()
    client_secret = os.getenv("SHAREPOINT_CLIENT_SECRET", "").strip()
    try:
        import streamlit as st
        if "SHAREPOINT_CLIENT_ID" in st.secrets:
            client_id = st.secrets["SHAREPOINT_CLIENT_ID"]
        if "SHAREPOINT_CLIENT_SECRET" in st.secrets:
            client_secret = st.secrets["SHAREPOINT_CLIENT_SECRET"]
    except Exception:
        pass
    return client_id, client_secret


def ocr_available() -> bool:
    """OCR is considered available if Tesseract data path is configured AND the libs import."""
    if not os.environ.get("TESSDATA_PREFIX"):
        return False
    return _OCR_LIBS_AVAILABLE
