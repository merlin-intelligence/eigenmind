"""``QdrantStore`` — single class wrapping all Qdrant operations.

Consolidates what previously lived in three files (client, ingestion, retrieval)
so callers don't have to thread a ``QdrantClient`` through every helper.
"""
from __future__ import annotations

import datetime
import logging
import uuid
from typing import Iterable

import numpy as np
from qdrant_client import QdrantClient, models

from eigenmind.config import (
    BATCH_SIZE,
    EMBEDDING_DIM_DEFAULT,
    qdrant_host,
    qdrant_port,
)

logger = logging.getLogger(__name__)


def make_point(
    filename: str,
    chunk_number: int,
    text: str,
    vector: list[float],
    ingestion_date: str | None = None,
) -> models.PointStruct:
    """Build a Qdrant ``PointStruct`` with a fresh UUID and standard payload."""
    return models.PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "filename": filename,
            "chunk_number": chunk_number,
            "text": text,
            "ingestion_date": ingestion_date or datetime.datetime.now().isoformat(),
        },
    )


def date_range_filter(date: datetime.date) -> models.Filter:
    """Filter for ``ingestion_date`` falling on the given calendar day."""
    next_day = date + datetime.timedelta(days=1)
    return models.Filter(must=[
        models.FieldCondition(
            key="ingestion_date",
            range=models.DatetimeRange(
                gte=f"{date.isoformat()}T00:00:00",
                lt=f"{next_day.isoformat()}T00:00:00",
            ),
        )
    ])


class QdrantStore:
    """Qdrant connection + collection management + ingestion + retrieval, in one place."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        timeout: int = 60,
    ):
        self.host = host or qdrant_host()
        self.port = port or qdrant_port()
        self._client = QdrantClient(host=self.host, port=self.port, timeout=timeout)

    # ─── connection / introspection ────────────────────────────────

    @property
    def client(self) -> QdrantClient:
        """Escape hatch to the underlying ``QdrantClient``."""
        return self._client

    @classmethod
    def is_reachable(cls, host: str | None = None, port: int | None = None) -> bool:
        """Quick health check that does not raise (used by the sidebar status pill)."""
        try:
            QdrantClient(host=host or qdrant_host(), port=port or qdrant_port(), timeout=3).get_collections()
            return True
        except Exception:
            return False

    def list_collections(self) -> list[str]:
        try:
            return sorted(c.name for c in self._client.get_collections().collections)
        except Exception:
            return []

    def collection_stats(self, name: str) -> tuple[int | None, int | str | None]:
        """Return ``(point_count, vector_size)`` or ``(None, None)`` on error."""
        try:
            info = self._client.get_collection(collection_name=name)
            count = info.points_count or 0
            size = info.config.params.vectors.size if hasattr(info.config.params.vectors, "size") else "—"
            return count, size
        except Exception:
            return None, None

    def ensure_collection(self, name: str, vector_size: int = EMBEDDING_DIM_DEFAULT) -> int:
        """Create the collection if missing. Returns the actual vector dimension stored."""
        try:
            info = self._client.get_collection(collection_name=name)
            if hasattr(info.config.params.vectors, "size"):
                return int(info.config.params.vectors.size)
            return vector_size
        except Exception:
            logger.info("Creating Qdrant collection %r (dim=%d, cosine)", name, vector_size)
            self._client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            return vector_size

    # ─── ingestion ─────────────────────────────────────────────────

    def existing_filenames(self, collection: str, scroll_limit: int = 1000) -> set[str]:
        """Distinct ``filename`` values already stored in the collection."""
        seen: set[str] = set()
        next_offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=collection,
                limit=scroll_limit,
                offset=next_offset,
                with_payload=["filename"],
                with_vectors=False,
            )
            for p in points:
                if p.payload and "filename" in p.payload:
                    seen.add(p.payload["filename"])
            if not next_offset:
                break
        return seen

    def batched_upsert(
        self,
        collection: str,
        points_iter: Iterable,
        batch_size: int = BATCH_SIZE,
        log=lambda _msg: None,
    ) -> int:
        """Drain ``points_iter`` into Qdrant in batches. Returns total upserted."""
        buffer: list = []
        total = 0
        for point in points_iter:
            buffer.append(point)
            if len(buffer) >= batch_size:
                log(f"  -> Upserting a batch of {len(buffer)} points...")
                self._client.upsert(collection_name=collection, points=buffer, wait=True)
                total += len(buffer)
                buffer = []
        if buffer:
            log(f"  -> Upserting the final batch of {len(buffer)} points...")
            self._client.upsert(collection_name=collection, points=buffer, wait=True)
            total += len(buffer)
        return total

    # ─── retrieval ─────────────────────────────────────────────────

    def similarity_search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[dict]:
        """Top-k cosine similarity search. Returns plain dicts ready for display/use."""
        res = self._client.query_points(collection_name=collection, query=query_vector, limit=limit)
        return [
            {
                "text": h.payload.get("text", ""),
                "filename": h.payload.get("filename", "unknown"),
                "chunk_number": h.payload.get("chunk_number", "?"),
                "score": round(h.score, 3) if hasattr(h, "score") else None,
            }
            for h in res.points
        ]

    def all_documents(self, collection: str, scroll_limit: int = 500) -> dict[str, dict]:
        """Reconstruct full documents from their chunks.

        Returns ``{filename: {"text": str, "vector": np.ndarray | None}}`` where
        ``text`` is the chunks concatenated in ``chunk_number`` order and ``vector``
        is the L2-normalised mean of the chunk embeddings (``None`` if a file has
        no usable vectors stored).
        """
        buckets: dict[str, list[tuple[int, str, list[float] | None]]] = {}
        next_offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=collection,
                limit=scroll_limit,
                offset=next_offset,
                with_payload=["filename", "chunk_number", "text"],
                with_vectors=True,
            )
            for p in points:
                payload = p.payload or {}
                fname = payload.get("filename", "unknown")
                buckets.setdefault(fname, []).append((
                    int(payload.get("chunk_number", 0)),
                    payload.get("text", ""),
                    p.vector,
                ))
            if not next_offset:
                break

        out: dict[str, dict] = {}
        for fname, chunks in buckets.items():
            chunks_sorted = sorted(chunks, key=lambda c: c[0])
            text = "\n".join(c[1] for c in chunks_sorted)
            raw_vecs = [c[2] for c in chunks_sorted if c[2] is not None]
            mean_vec: np.ndarray | None = None
            if raw_vecs:
                arr = np.asarray(raw_vecs, dtype=np.float32).mean(axis=0)
                norm = float(np.linalg.norm(arr))
                mean_vec = arr / norm if norm > 0 else arr
            out[fname] = {"text": text, "vector": mean_vec}
        return out

    def documents_for_date(self, collection: str, date: datetime.date) -> dict[str, int]:
        """``{filename: chunk_count}`` for documents ingested on the given date."""
        docs: dict[str, int] = {}
        flt = date_range_filter(date)
        offset = None
        while True:
            res, next_offset = self._client.scroll(
                collection_name=collection,
                scroll_filter=flt,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for p in res:
                fname = p.payload.get("filename", "unknown")
                docs[fname] = docs.get(fname, 0) + 1
            offset = next_offset
            if offset is None:
                break
        return docs

    # ─── deletion ──────────────────────────────────────────────────

    def delete_collection(self, name: str) -> None:
        logger.info("Deleting Qdrant collection %r", name)
        self._client.delete_collection(collection_name=name)

    def delete_by_filter(self, collection: str, flt: models.Filter) -> None:
        self._client.delete(collection_name=collection, points_selector=models.FilterSelector(filter=flt))

    def delete_for_date(self, collection: str, date: datetime.date) -> None:
        self.delete_by_filter(collection, date_range_filter(date))

    def delete_document_on_date(self, collection: str, filename: str, date: datetime.date) -> None:
        next_day = date + datetime.timedelta(days=1)
        flt = models.Filter(must=[
            models.FieldCondition(key="filename", match=models.MatchValue(value=filename)),
            models.FieldCondition(
                key="ingestion_date",
                range=models.DatetimeRange(
                    gte=f"{date.isoformat()}T00:00:00",
                    lt=f"{next_day.isoformat()}T00:00:00",
                ),
            ),
        ])
        self.delete_by_filter(collection, flt)
