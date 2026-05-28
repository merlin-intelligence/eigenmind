"""SentenceTransformer wrapper with device auto-detection."""
from __future__ import annotations

import gc

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from eigenmind.config import EMBEDDING_MODEL_NAME


def detect_device() -> str:
    """Return the best available compute device for embeddings: cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EmbeddingModel:
    """Lifecycle wrapper around a SentenceTransformer.

    Use as a context manager to release GPU memory automatically::

        with EmbeddingModel() as embedder:
            vector = embedder.encode("hello world")
    """

    def __init__(self, device: str | None = None, model_name: str = EMBEDDING_MODEL_NAME):
        self.device = device or detect_device()
        self.model_name = model_name
        gc.collect()
        self._model = SentenceTransformer(model_name, device=self.device)

    def encode(self, text: str | list[str]) -> np.ndarray:
        """Encode one or more strings to a (batch of) embedding vector(s).

        Raw passthrough — does NOT apply E5 prefixes. Prefer
        :meth:`encode_query` / :meth:`encode_passage` for retrieval workloads.
        """
        return self._model.encode(text)

    def encode_query(self, text: str | list[str]) -> np.ndarray:
        """Encode a search query. Applies the ``"query: "`` prefix expected by E5."""
        prefixed = f"query: {text}" if isinstance(text, str) else [f"query: {t}" for t in text]
        return self._model.encode(prefixed)

    def encode_passage(self, text: str | list[str]) -> np.ndarray:
        """Encode a passage for indexing. Applies the ``"passage: "`` prefix expected by E5."""
        prefixed = f"passage: {text}" if isinstance(text, str) else [f"passage: {t}" for t in text]
        return self._model.encode(prefixed)

    @property
    def dim(self) -> int:
        """Return the embedding dimensionality of the underlying model."""
        return self._model.get_sentence_embedding_dimension()

    @property
    def raw(self) -> SentenceTransformer:
        """Escape hatch to the underlying SentenceTransformer."""
        return self._model

    def release(self) -> None:
        """Release the model and clear the CUDA cache. Subsequent calls are no-ops."""
        if self._model is None:
            return
        del self._model
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "EmbeddingModel":
        return self

    def __exit__(self, *_) -> None:
        self.release()
