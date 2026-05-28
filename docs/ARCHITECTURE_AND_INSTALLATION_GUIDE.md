# Architecture & Installation Guide

## System Architecture

*   **Frontend**: Streamlit multipage app (`streamlit_app.py` + `pages/`)
*   **Vector Database**: Qdrant (Running locally via Docker)
*   **Embeddings**: SentenceTransformers (`intfloat/multilingual-e5-base`, 768-dim, multilingual). E5 `query:` / `passage:` prefixes are applied automatically by the wrapper in `eigenmind/core/embeddings.py`. Runs locally on CPU/CUDA/MPS, and is cached process-wide (single instance shared across all Streamlit sessions and users).
*   **LLM Provider**: Nebius AI (Llama, Kimi, OSS models) via REST API
*   **Processing**: NLTK (Key Concept Extraction), Langchain (Chunking), NetworkX (Graph Mathematics)
*   **Multi-User**: per-user authentication, isolated Qdrant collections (namespaced `<user>_<collection>`), and per-user OAuth token storage under `user_data/<user>/`.

---

## Prerequisites

1.  **Python 3.10+** installed.
2.  **Docker** and **Docker Compose** installed (for Qdrant).
3.  **Tesseract OCR** (Optional, but recommended for processing scanned PDFs).

---

## Step 1: Clone and Environment Setup

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/foustry/eigenmind.git
cd eigenmind
python -m venv venv

# On Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate
```

## Step 2: Install Dependencies

Install the project in editable mode (preferred — uses `pyproject.toml` and exposes the `eigenmind-ingest` CLI):

```bash
pip install -e .
# Optional extras: ocr, gdrive, sharepoint, dev
pip install -e ".[ocr,gdrive,sharepoint]"
```

Alternatively, the legacy `requirements.txt` is still maintained:

```bash
pip install -r requirements.txt
```

*Note: PyTorch will install a default version. If you require specialized CUDA support, please install PyTorch according to your system specs from pytorch.org.*

## Step 3: Launch Qdrant Vector DB (Docker)

The application expects Qdrant to be accessible at `localhost:6333`. Launch it using the provided Docker Compose file:

```bash
docker-compose up -d
```

This will download the Qdrant image and start the database in the background. Your vectors will be saved persistently in the `./qdrant_storage` folder.

## Step 4: Configure API Keys (Secrets)

Eigenmind relies on external APIs for LLM generation and cloud ingestion. Two equivalent configuration paths are supported — pick one.

### Option A — `.env` file (recommended for local dev)

Copy the template and fill in your values:

```bash
cp .env.example .env
```

```dotenv
# .env
NEBIUS_API_KEY=your_nebius_api_key_here
SHAREPOINT_CLIENT_ID=your_azure_client_id
SHAREPOINT_CLIENT_SECRET=your_azure_client_secret
QDRANT_HOST=localhost
QDRANT_PORT=6333
# HF_TOKEN=...   # only if you need a gated local LLM
```

### Option B — Streamlit secrets (recommended for deployments)

```toml
# .streamlit/secrets.toml

# Nebius AI (Required for /ask/)
NEBIUS_API_KEY = "your_nebius_api_key_here"

# SharePoint (Optional: For SharePoint ingestion)
SHAREPOINT_CLIENT_ID = "your_azure_client_id"
SHAREPOINT_CLIENT_SECRET = "your_azure_client_secret"

# Multi-user accounts (optional; bootstraps user_data/users.json on first run)
[USERS]
alice = "alice_password"
bob   = "bob_password"
```

When both sources are present, `st.secrets` takes precedence over `.env`.

*Note for Google Drive*: Upload your `client_secrets.json` or `service_account.json` directly through the app interface when prompted. The OAuth token is then cached at `user_data/<user>/gdrive_token.json`.

## Performance & Stability Optimizations

Eigenmind is optimized to run on resource-constrained environments (e.g., 4GB RAM VMs).

### 1. Memory Management
- **Shared Model Cache**: The SentenceTransformer is loaded once on first use via `@st.cache_resource` (see `get_embedder()` in `eigenmind/ui/components.py`) and kept resident in the Streamlit server process. The same ~300 MB instance is reused across **all sessions and all users** of the service — no per-request reload, no per-user copy. The cache is released only when the process exits (e.g. `systemctl restart eigenmind`).
- **CLI Ingestion**: When the embedder is not injected (e.g. `eigenmind-ingest` from the CLI), the ingester falls back to a scoped `with EmbeddingModel(...)` that releases the model and clears the PyTorch cache at end of run.
- **CPU-First**: By default, the app uses CPU-only PyTorch to ensure stability and avoid GPU-related memory overhead on low-end systems. CUDA / MPS are auto-detected when available.

### 2. Swap File Recommendation (Linux)
If running on a system with 4GB RAM or less, it is highly recommended to configure a swap file to absorb the cold-start spike (model download + load) and prevent OOM (Out Of Memory) crashes during embedding. Note that once the model is loaded, it stays resident — peak memory is reached on first request, not steadily during use.

```bash
sudo fallocate -l 3.3G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
# To make it permanent, add '/swapfile none swap sw 0 0' to /etc/fstab
```

### 3. Smart Resume Feature
The ingestion pipeline tracks processed files by storing their filenames as metadata in Qdrant.
- When re-running an ingestion on the same directory, Eigenmind will automatically skip files that have already been indexed.
- The `downloaded_corpus/` directory acts as a local cache for cloud-synced documents (Google Drive, SharePoint).

## Step 5: Run the Application

```bash
streamlit run streamlit_app.py
```

Navigate between pages from the Streamlit sidebar:

| Page | Purpose |
|---|---|
| `pages/1_Ingest.py`         | `/enrich corpus/` - Build / extend a corpus from local dirs, Google Drive, SharePoint |
| `pages/2_Chat.py`           | `/ask/` - Hybrid RAG question answering against a selected collection |
| `pages/3_Graph_Explorer.py` | `/explore graphs/` - Subgraph view (Singular / Hinge / Theta nodes) |
| `pages/4_Manage.py`         | `/manage/` - List & delete documents per ingestion date |

### CLI ingestion (no UI)

```bash
eigenmind-ingest directories.txt my_collection --device cpu
```