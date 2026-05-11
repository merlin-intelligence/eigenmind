# Eigenmind - /accelerate clarity/

Eigenmind is a sophisticated knowledge management and exploration application built with Streamlit. It leverages a local Vector Database (Qdrant via Docker) and advanced AI models (via Nebius AI) to ingest, map, and query your proprietary document corpus.

## Core Features

1. **Multi-Source Ingestion (`/add files to your corpus/`)**
   - Local directories (PDF, Word, Excel, PowerPoint, Text) with OCR support.
   - Direct synchronization with Google Drive (OAuth or Service Account).
   - Direct synchronization with Microsoft SharePoint.
   - **Smart Resume**: File-level resume feature using Qdrant filename tracking to skip already processed documents.
   - Local embedding utilizing `sentence-transformers`.

2. **Knowledge Graph Navigation (`/navigate your experience/`)**
   - Generates interactive subgraphs of knowledge relationships based on specific prompts.
   - Performs eigenvector/Laplacian analysis and identifies **Singular**, **Hinge**, and **Theta** nodes for unique, non-obvious insights.

3. **Advanced Question Answering (`/ask/`)**
   - Hybrid Retrieval-Augmented Generation (RAG) using both standard semantic similarity and singular chunk analysis.
   - Interacts with remote LLMs (Llama 3.3, Kimi 2.5, OpenAI OSS 120B) hosted on Nebius AI endpoints.

## Stability & Performance

- **Memory Optimized**: Designed for systems with limited RAM (e.g., 4GB). Features delayed model loading, explicit garbage collection, and a 3.3GB swap file configuration for stability.
- **CPU-First**: Switched to CPU-only PyTorch by default for broad compatibility and predictable memory usage.
- **Persistent Storage**: All ingested data and vector embeddings are stored persistently in the `qdrant_storage/` directory.

## Getting Started

Please refer to the [Architecture and Installation Guide](architecture_and_installation_guide.md) for detailed instructions on deploying the Qdrant database, installing dependencies, and configuring API keys.

## Project layout (v0.2)

```
eigenmind/                  importable package
├── config.py               single source of truth for constants and env-based secrets
├── core/                   embeddings, LLM, chunking, document loaders
├── vectordb/               Qdrant client, ingestion, retrieval helpers
├── graph/                  graph algorithms (singular, connectivity, theta, exploration)
├── connectors/             Google Drive, SharePoint
├── pipelines/              orchestration: ingest, rag
└── ui/                     Streamlit-only code (auth, components, styles)

streamlit_app.py            entry point — `streamlit run streamlit_app.py`
pages/                      Streamlit multipage navigation (Ingest, Chat, Graph Explorer, Manage)
scripts/ingest_recursive.py CLI ingestion (also exposed as `eigenmind-ingest`)
tests/unit/                 pure-numpy graph-math tests
legacy/                     pre-refactor flat scripts, kept for reference
```

Run the app:

```bash
pip install -e .
streamlit run streamlit_app.py
```

CLI ingestion:

```bash
eigenmind-ingest directories.txt my_collection --device cpu
```

Secrets come from `.env` or `.streamlit/secrets.toml` (see `.env.example`).
The HuggingFace token is **never** hard-coded — set `HF_TOKEN` in your env if
you need a gated local LLM.

---
© 2025 Prax Value Eurl (PxV)