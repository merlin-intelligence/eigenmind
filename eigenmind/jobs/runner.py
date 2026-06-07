"""Daemon thread that processes ingestion jobs from an in-memory queue."""
from __future__ import annotations

import os
import queue
import threading
import traceback

from eigenmind.jobs.store import JobStore
from eigenmind.pipelines.ingest import Ingester
from eigenmind.vectordb.store import QdrantStore


class JobRunner:
    """Single-worker background runner.

    One daemon thread per (qdrant_host, qdrant_port) pair, shared across all
    browser sessions. Jobs are processed sequentially to avoid GPU contention.
    Survives browser disconnections and tab closures — only stops with the process.
    """

    def __init__(self, store: JobStore, qdrant_host: str = "localhost", qdrant_port: int = 6333) -> None:
        self.store = store
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="eigenmind-ingestion"
        )
        self._thread.start()

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

    # ── private ──────────────────────────────────────────────────────

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run(job_id)
            except Exception:  # noqa: BLE001
                self.store.set_status(job_id, "failed")
                self.store.append_log(job_id, f"Unexpected runner error:\n{traceback.format_exc()}")
            finally:
                self._queue.task_done()

    def _run(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if job is None:
            return

        self.store.set_status(job_id, "running")

        ingester = Ingester(
            store=QdrantStore(self.qdrant_host, self.qdrant_port),
            device=job["device"],
            progress_callback=lambda cur, tot: self.store.set_progress(job_id, cur, tot),
            history_file=job["history_file"],
        )
        # Wire Ingester logs into the job store so they appear in the UI
        ingester._log = lambda msg: self.store.append_log(job_id, msg)

        try:
            ingester.run_chunknorris(job["directories_file"], job["collection"])
            self.store.set_status(job_id, "done")
        except Exception as e:  # noqa: BLE001
            self.store.set_status(job_id, "failed")
            self.store.append_log(job_id, f"Fatal: {e}\n{traceback.format_exc()}")
        finally:
            dirs_file = job["directories_file"]
            try:
                if os.path.exists(dirs_file):
                    os.remove(dirs_file)
            except OSError:
                pass
