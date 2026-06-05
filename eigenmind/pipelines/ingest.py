"""Ingestion pipeline: load → chunk → embed → upsert.

The :class:`Ingester` class holds the host/port/device/callback configuration so
callers don't pass it to every method. A single strategy is exposed:

- :meth:`Ingester.run_chunknorris` — recursive PDF/DOCX/XLSX/CSV/PPTX/TXT/MD ingestion,
  every format parsed to markdown and chunked through ChunkNorris.
"""
from __future__ import annotations

import contextlib
import datetime
import logging
import os
import traceback
from typing import Callable

logger = logging.getLogger(__name__)

from eigenmind.config import (
    BATCH_SIZE,
    CHUNKNORRIS_EXTENSIONS,
    EMBEDDING_DIM_DEFAULT,
    UNSUPPORTED_EXTENSIONS,
)
from eigenmind.core.chunking import chunk_with_chunknorris
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
        embedder: EmbeddingModel | None = None,
    ):
        self.store = store or QdrantStore()
        self.device = device or detect_device()
        self.progress_callback = progress_callback
        self.history_file = history_file
        self.logs: list[str] = []
        # If an embedder is injected (e.g. from a Streamlit cache_resource), reuse it
        # without taking ownership — caller controls its lifecycle.
        self._injected_embedder = embedder

    def _log(self, msg: str) -> None:
        self.logs.append(msg)
        logger.info(msg)

    @contextlib.contextmanager
    def _embedder_scope(self, log):
        """Yield an embedder, reusing the injected one or loading+releasing a fresh one."""
        if self._injected_embedder is not None:
            yield self._injected_embedder
            return
        log(f"Loading embedding model on {self.device}...")
        with EmbeddingModel(device=self.device) as embedder:
            yield embedder

    # ─── strategy 1: ChunkNorris (PDF/DOCX/TXT/MD) ─────────────────

    def run_chunknorris(self, directories_file_path: str, collection: str) -> list[str]:
        """Recursive ingestion of PDF/DOCX/TXT/MD using ChunkNorris. Returns log messages.

        DOCX and TXT are converted to markdown so they go through the same
        ChunkNorris pipeline as PDF and MD. PPTX/XLSX/CSV/JSON are skipped with a
        "not supported yet" notice.
        """
        self.logs = []
        log = self._log

        self.store.ensure_collection(collection, EMBEDDING_DIM_DEFAULT)
        skip = self.store.existing_filenames(collection)
        if skip:
            log(f"Found {len(skip)} unique documents already in Qdrant.")

        directories = _read_directories_file(directories_file_path, log)
        if not directories:
            log("No directories to process.")
            return self.logs
        log(f"Found {len(directories)} directories to process.")

        # Report recognised-but-unsupported formats so the user knows they were skipped.
        ignored = _walk_files(directories, UNSUPPORTED_EXTENSIONS, set(), lambda _m: None)
        for fp in ignored:
            ext = os.path.splitext(fp)[1].lower()
            log(f"Skipping {os.path.basename(fp)}: format '{ext}' is not supported for now.")

        files = _walk_files(directories, CHUNKNORRIS_EXTENSIONS, skip, log)
        if not files:
            log("No new files to index.")
            return self.logs
        log(f"Total new files to process: {len(files)}")

        with self._embedder_scope(log) as embedder:
            total = self.store.batched_upsert(
                collection,
                self._chunknorris_points(files, embedder, log),
                BATCH_SIZE,
                log,
            )

        if self.history_file:
            self._update_history(directories, log)

        log(f"\nProcess complete! Added a total of {total} new points.")
        return self.logs

    def _chunknorris_points(self, files: list[str], embedder: EmbeddingModel, log):
        for processed, fp in enumerate(files, start=1):
            fname = os.path.basename(fp)
            log(f"Processing file: {fp}")
            try:
                ingestion_date = datetime.datetime.now().isoformat()
                for i, chunk in enumerate(chunk_with_chunknorris(fp)):
                    text = chunk.get_text()
                    if not text.strip():
                        continue
                    yield make_point(fname, i, text, embedder.encode_passage(text).tolist(), ingestion_date)
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
