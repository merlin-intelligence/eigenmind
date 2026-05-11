"""Ingestion pipelines: load → chunk → embed → upsert.

The :class:`Ingester` class holds the host/port/device/callback configuration so
callers don't pass it to every method. Two strategies are exposed:

- :meth:`Ingester.run_chunknorris` — recursive PDF ingestion using ChunkNorris.
- :meth:`Ingester.run_multi_format` — PDF/DOCX/PPTX/XLSX/TXT/MD via Langchain.
"""
from __future__ import annotations

import datetime
import os
import traceback
from typing import Callable

from eigenmind.config import (
    BATCH_SIZE,
    EMBEDDING_DIM_DEFAULT,
    SUPPORTED_EXTENSIONS,
)
from eigenmind.core.chunking import chunknorris_pipeline, langchain_chunker
from eigenmind.core.document_loaders import extract_text
from eigenmind.core.embeddings import EmbeddingModel, detect_device
from eigenmind.vectordb.store import QdrantStore, make_point


def _read_directories_file(path: str, log) -> list[str]:
    if not os.path.exists(path):
        log(f"Error: The file '{path}' was not found.")
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip().strip("'\"") for line in f if line.strip()]


def _walk_files(directories: list[str], extensions: tuple[str, ...], skip: set[str], log) -> list[str]:
    out: list[str] = []
    skipped = 0
    for d in directories:
        real = os.path.realpath(d)
        if not os.path.isdir(real):
            log(f"Warning: '{d}' is not a valid directory. Skipping.")
            continue
        for root, _, files in os.walk(real):
            for fn in files:
                if fn.lower().endswith(extensions):
                    if fn in skip:
                        skipped += 1
                        continue
                    out.append(os.path.join(root, fn))
    if skipped:
        log(f"Skipped {skipped} files already present in Qdrant.")
    return out


class Ingester:
    """Bundle of {Qdrant store, embedding device, progress callback, history file}.

    One instance per ingestion run. Lazily creates the embedding model on first use
    and releases it at the end.
    """

    def __init__(
        self,
        store: QdrantStore | None = None,
        device: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        history_file: str | None = None,
    ):
        self.store = store or QdrantStore()
        self.device = device or detect_device()
        self.progress_callback = progress_callback
        self.history_file = history_file
        self.logs: list[str] = []

    def _log(self, msg: str) -> None:
        self.logs.append(msg)

    # ─── strategy 1: ChunkNorris (PDF only) ────────────────────────

    def run_chunknorris(self, directories_file_path: str, collection: str) -> list[str]:
        """Recursive PDF-only ingestion using ChunkNorris. Returns log messages."""
        self.logs = []
        log = self._log
        pipeline = chunknorris_pipeline()

        self.store.ensure_collection(collection, EMBEDDING_DIM_DEFAULT)
        skip = self.store.existing_filenames(collection)
        if skip:
            log(f"Found {len(skip)} unique documents already in Qdrant.")

        directories = _read_directories_file(directories_file_path, log)
        if not directories:
            log("No directories to process.")
            return self.logs
        log(f"Found {len(directories)} directories to process.")

        files = _walk_files(directories, (".pdf",), skip, log)
        if not files:
            log("No new files to index.")
            return self.logs
        log(f"Total new PDF files to process: {len(files)}")

        log(f"Loading embedding model on {self.device}...")
        with EmbeddingModel(device=self.device) as embedder:
            total = self.store.batched_upsert(
                collection,
                self._chunknorris_points(files, pipeline, embedder, log),
                BATCH_SIZE,
                log,
            )

        if self.history_file:
            self._update_history(directories, log)

        log(f"\nProcess complete! Added a total of {total} new points.")
        return self.logs

    def _chunknorris_points(self, files: list[str], pipeline, embedder: EmbeddingModel, log):
        for processed, fp in enumerate(files, start=1):
            fname = os.path.basename(fp)
            log(f"Processing file: {fp}")
            try:
                ingestion_date = datetime.datetime.now().isoformat()
                for i, chunk in enumerate(pipeline.chunk_file(fp)):
                    text = chunk.get_text()
                    if not text.strip():
                        continue
                    yield make_point(fname, i, text, embedder.encode(text).tolist(), ingestion_date)
            except Exception as e:  # noqa: BLE001
                log(f"  -> Error processing {fp}: {e}")
                log(f"  -> Traceback: {traceback.format_exc()}")
            if self.progress_callback:
                self.progress_callback(processed, len(files))

    # ─── strategy 2: multi-format (Langchain) ──────────────────────

    def run_multi_format(self, directories_file_path: str, collection: str) -> list[str]:
        """Multi-format ingestion (PDF/DOCX/PPTX/XLSX/TXT/MD) with Langchain chunking."""
        self.logs = []
        log = self._log

        with EmbeddingModel(device=self.device) as embedder:
            self.store.ensure_collection(collection, embedder.dim)
            skip = self.store.existing_filenames(collection)
            if skip:
                log(f"Found {len(skip)} documents already in the collection. They will be skipped.")

            directories = _read_directories_file(directories_file_path, log)
            if not directories:
                return self.logs

            files = _walk_files(directories, SUPPORTED_EXTENSIONS, skip, log)
            if not files:
                log("No new files to process.")
                return self.logs
            log(f"Found {len(files)} new files to index.")

            chunker = langchain_chunker()
            total = self.store.batched_upsert(
                collection,
                self._multi_format_points(files, chunker, embedder, log),
                BATCH_SIZE,
                log,
            )

        log(f"\nProcess complete! Added a total of {total} new points from {len(files)} files.")
        return self.logs

    def _multi_format_points(self, files: list[str], chunker, embedder: EmbeddingModel, log):
        for processed, fp in enumerate(files, start=1):
            fname = os.path.basename(fp)
            log(f"Processing file: {fp}")
            try:
                full_text = extract_text(fp, log)
                if not full_text.strip():
                    log(f"  -> Warning: Extracted text is empty for {fname}. Skipping.")
                    continue
                ingestion_date = datetime.datetime.now().isoformat()
                for i, chunk_text in enumerate(chunker.split_text(full_text)):
                    yield make_point(fname, i, chunk_text, embedder.encode(chunk_text).tolist(), ingestion_date)
            except Exception as e:  # noqa: BLE001
                log(f"  -> Error processing {fp}: {e}")
                log(f"  -> Traceback: {traceback.format_exc()}")
            if self.progress_callback:
                self.progress_callback(processed, len(files))

    # ─── helpers ───────────────────────────────────────────────────

    def _update_history(self, directories: list[str], log) -> None:
        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                for d in directories:
                    f.write(f"{os.path.realpath(d)}\n")
            log(f"Updated history file '{self.history_file}'.")
        except Exception as e:  # noqa: BLE001
            log(f"Warning: Could not update history file: {e}")
