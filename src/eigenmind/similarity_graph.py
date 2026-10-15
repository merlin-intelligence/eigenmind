"""``SimilarityGraph`` — high-level API around the Eigenmind selection methods.

Holds the cosine matrix C and the adjacency matrix W of a set of retrieved
chunks and exposes:
  1. Spectral Attention (spectral vertices / medoids / tails)
  2. Semantic Bridges (spanning-tree bridges between components)
  3. Supported Diversity (exact MILP or LP-guided greedy)
plus the comparison baselines and Gaussian-kernel diverse coverage. Results are
memoized per full argument set.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from eigenmind.coverage import build_gaussian_kernel
from eigenmind.coverage import diverse_coverage as compute_diverse_coverage
from eigenmind.defaults import (
    BRIDGES_MIN_COMPONENT_SIZE,
    BRIDGES_MIN_SIMILARITY,
    BRIDGES_MODE,
    BRIDGES_TOP_M,
    COVERAGE_BANDWIDTH,
    COVERAGE_RELEVANCE_WEIGHT,
    DIVERSITY_BUDGET,
    DIVERSITY_MIN_DEGREE,
    DIVERSITY_RELAXATION,
    DIVERSITY_SOLVER,
    DIVERSITY_TIME_LIMIT,
    SIMILARITY_THRESHOLD,
    SPECTRAL_ALLOCATION,
    SPECTRAL_DEGREE_REGULARIZATION,
    SPECTRAL_LAPLACIAN,
    SPECTRAL_MAX_GAP_EXTENSION,
    SPECTRAL_MAX_PER_SELECTION,
    SPECTRAL_MIN_RELATIVE_GAP,
    SPECTRAL_POSITIVE_COUNT,
    SPECTRAL_SELECTION,
)
from eigenmind.graph_construction import (
    cosine_matrix,
    graph_summary,
    normalize_embeddings,
    threshold_graph,
)
from eigenmind.selection_baselines import facility_location as baseline_facility_location
from eigenmind.selection_baselines import farthest_first as baseline_farthest_first
from eigenmind.selection_baselines import kdpp_greedy as baseline_kdpp_greedy
from eigenmind.selection_baselines import mmr as baseline_mmr
from eigenmind.semantic_bridges import semantic_bridges
from eigenmind.spectral_attention import component_spectrum, spectral_attention
from eigenmind.supported_diversity import supported_diversity


def _freeze(value: Any) -> Any:
    """Hashable, value-based representation of an argument for cache keys."""
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _cache_key(**kwargs: Any) -> tuple:
    return tuple(sorted((k, _freeze(v)) for k, v in kwargs.items()))


class SimilarityGraph:
    """A weighted graph over chunks, built from their embedding vectors.

    ``retrieved_points`` are objects exposing ``.id`` (hashable, unique),
    ``.vector`` (same length for every point) and optionally ``.payload``
    (a dict; ``payload["text"]`` is used as chunk text when present).
    Vectors are normalized here, optionally after mean-centring and removal of
    the top principal directions (``center``, ``remove_top``).
    """

    def __init__(
        self,
        retrieved_points: list[Any],
        threshold: float = SIMILARITY_THRESHOLD,
        center: bool = False,
        remove_top: int = 0,
    ):
        self.points = retrieved_points
        self.threshold = threshold
        self.id_to_point = {p.id: p for p in retrieved_points}
        if len(self.id_to_point) != len(retrieved_points):
            raise ValueError("Point ids must be unique; deduplicate retrieved_points first")
        self.ordered_ids = [p.id for p in retrieved_points]

        if retrieved_points:
            raw_matrix = np.array([p.vector for p in retrieved_points], dtype=float)
            self.embedding_matrix = normalize_embeddings(
                raw_matrix, center=center, remove_top=remove_top
            )
            self.C = cosine_matrix(self.embedding_matrix)
            self.W = threshold_graph(self.C, tau=threshold)
        else:
            self.embedding_matrix = np.empty((0, 0))
            self.C = np.empty((0, 0))
            self.W = np.empty((0, 0))

        self._cache: dict[tuple, Any] = {}

    # ─── Basics ───────────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        return self.W.shape[0]

    def _cached(self, name: str, compute: Any, **kwargs: Any) -> Any:
        """Compute once per (name, arguments); returns the stored object itself."""
        key = (name, _cache_key(**kwargs))
        if key not in self._cache:
            self._cache[key] = compute(**kwargs)
        return self._cache[key]

    def _memo(self, name: str, compute: Any, **kwargs: Any) -> Any:
        """Memoized result as a deep copy, so callers cannot alter the cache."""
        return copy.deepcopy(self._cached(name, compute, **kwargs))

    def _default_chunks(self) -> list[str] | None:
        """Payload texts when every point has a string ``payload['text']``."""
        payloads = [getattr(p, "payload", None) for p in self.points]
        texts = [pl.get("text") if isinstance(pl, dict) else None for pl in payloads]
        return texts if texts and all(isinstance(t, str) for t in texts) else None

    def summary(self) -> dict[str, Any]:
        """Graph statistics to log before selecting (components, isolates, sizes)."""
        return graph_summary(self.W)

    def laplacian(self) -> np.ndarray:
        """Symmetric normalized Laplacian of W (isolated vertices get a unit diagonal)."""
        degrees = self.W.sum(axis=1)
        inv_sqrt = np.zeros_like(degrees)
        nz = degrees > 0
        inv_sqrt[nz] = 1.0 / np.sqrt(degrees[nz])
        return np.eye(self.n) - inv_sqrt[:, None] * self.W * inv_sqrt[None, :]

    def component_spectra(
        self,
        laplacian: str = SPECTRAL_LAPLACIAN,
        degree_regularization: float = SPECTRAL_DEGREE_REGULARIZATION,
    ) -> list[dict[str, Any]]:
        """Eigen-decomposition of each connected component with at least two chunks.

        Returns a list of ``{"indices", "eigenvalues", "eigenvectors"}`` dicts, one
        per nontrivial component, as used internally by Spectral Attention.
        """
        if self.n == 0:
            return []
        count, labels = connected_components(sparse.csr_matrix(self.W), directed=False)
        out = []
        for c in range(count):
            idx = np.flatnonzero(labels == c)
            if len(idx) < 2:
                continue
            lam, U, _ = component_spectrum(
                self.W[np.ix_(idx, idx)], laplacian, degree_regularization
            )
            out.append({"indices": idx, "eigenvalues": lam, "eigenvectors": U})
        return out

    # ─── 1. Spectral Attention ────────────────────────────────────────────────

    def spectral_attention(
        self,
        positive_count: int = SPECTRAL_POSITIVE_COUNT,
        allocation: str = SPECTRAL_ALLOCATION,
        min_relative_gap: float = SPECTRAL_MIN_RELATIVE_GAP,
        max_gap_extension: int | None = SPECTRAL_MAX_GAP_EXTENSION,
        selection: str | tuple[str, ...] = SPECTRAL_SELECTION,
        max_per_selection: int | None = SPECTRAL_MAX_PER_SELECTION,
        laplacian: str = SPECTRAL_LAPLACIAN,
        degree_regularization: float = SPECTRAL_DEGREE_REGULARIZATION,
        chunks: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run Spectral Attention on W; see :func:`eigenmind.spectral_attention`."""
        return self._memo(
            "spectral_attention",
            lambda **kw: spectral_attention(self.W, **kw),
            chunks=chunks if chunks is not None else self._default_chunks(),
            positive_count=positive_count,
            allocation=allocation,
            min_relative_gap=min_relative_gap,
            max_gap_extension=max_gap_extension,
            selection=selection,
            max_per_selection=max_per_selection,
            laplacian=laplacian,
            degree_regularization=degree_regularization,
            **kwargs,
        )

    # ─── 2. Semantic Bridges ──────────────────────────────────────────────────

    def semantic_bridges(
        self,
        mode: str = BRIDGES_MODE,
        top_m: int = BRIDGES_TOP_M,
        min_component_size: int = BRIDGES_MIN_COMPONENT_SIZE,
        min_similarity: float | None = BRIDGES_MIN_SIMILARITY,
        max_bridges: int | None = None,
        chunks: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run Semantic Bridges on (W, C); see :func:`eigenmind.semantic_bridges`."""
        return self._memo(
            "semantic_bridges",
            lambda **kw: semantic_bridges(self.W, self.C, **kw),
            chunks=chunks if chunks is not None else self._default_chunks(),
            mode=mode,
            top_m=top_m,
            min_component_size=min_component_size,
            min_similarity=min_similarity,
            max_bridges=max_bridges,
            **kwargs,
        )

    # ─── 3. Supported Diversity ───────────────────────────────────────────────

    def supported_diversity(
        self,
        utilities: np.ndarray | list[float] | None = None,
        budget: int = DIVERSITY_BUDGET,
        min_degree: float = DIVERSITY_MIN_DEGREE,
        relaxation: str = DIVERSITY_RELAXATION,
        solver: str = DIVERSITY_SOLVER,
        time_limit: float = DIVERSITY_TIME_LIMIT,
        chunks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run Supported Diversity on W; see :func:`eigenmind.supported_diversity`.

        ``utilities=None`` means unit utilities: a cardinality criterion with no
        relevance or quality signal.
        """
        u = np.ones(self.n) if utilities is None else np.asarray(utilities, dtype=float)
        return self._memo(
            "supported_diversity",
            lambda **kw: supported_diversity(self.W, **kw),
            utilities=u,
            budget=budget,
            chunks=chunks if chunks is not None else self._default_chunks(),
            min_degree=min_degree,
            relaxation=relaxation,
            solver=solver,
            time_limit=time_limit,
        )

    # ─── Baselines ────────────────────────────────────────────────────────────

    def mmr(
        self, relevance: np.ndarray | list[float], budget: int = 10, lam: float = 0.5
    ) -> list[int]:
        """Maximal Marginal Relevance baseline on C."""
        return baseline_mmr(relevance, self.C, budget=budget, lam=lam)

    def facility_location(
        self,
        budget: int = 10,
        weights: np.ndarray | list[float] | None = None,
        utilities: np.ndarray | list[float] | None = None,
        lam: float = 0.0,
    ) -> list[int]:
        """Facility-location baseline on clipped cosines (evaluation arm)."""
        return baseline_facility_location(
            self.C, budget=budget, weights=weights, utilities=utilities, lam=lam
        )

    def kdpp_greedy(
        self, budget: int = 10, quality: np.ndarray | list[float] | None = None
    ) -> list[int]:
        """k-DPP greedy MAP baseline on C."""
        return baseline_kdpp_greedy(self.C, budget=budget, quality=quality)

    def farthest_first(self, budget: int = 10, start: int | None = None) -> list[int]:
        """Farthest-first traversal baseline on cosine distance."""
        return baseline_farthest_first(self.C, budget=budget, start=start)

    # ─── Diverse coverage (Gaussian kernel) ───────────────────────────────────

    def _kernel(self, bandwidth: float) -> np.ndarray:
        return self._cached(
            "kernel",
            lambda **kw: build_gaussian_kernel(self.embedding_matrix, **kw),
            bandwidth=bandwidth,
        )

    def kernel(self, bandwidth: float = COVERAGE_BANDWIDTH) -> np.ndarray:
        """Gaussian chord kernel K (unit diagonal), memoized per bandwidth."""
        if self.n == 0:
            return np.empty((0, 0))
        return self._kernel(bandwidth).copy()

    def diverse_coverage(
        self,
        budget: int = 10,
        relevance: np.ndarray | list[float] | None = None,
        weights: np.ndarray | list[float] | None = None,
        relevance_weight: float = COVERAGE_RELEVANCE_WEIGHT,
        bandwidth: float = COVERAGE_BANDWIDTH,
    ) -> dict[str, Any]:
        """Facility-location coverage over the Gaussian kernel, with an upper bound."""
        if self.n == 0:
            return {
                "indices": [],
                "marginal_gains": [],
                "coverage": [],
                "objective": 0.0,
                "upper_bound": 0.0,
            }
        return compute_diverse_coverage(
            self._kernel(bandwidth),
            budget=budget,
            relevance=relevance,
            weights=weights,
            relevance_weight=relevance_weight,
        )

    # ─── Tag aggregation ──────────────────────────────────────────────────────

    def selection_tags(
        self,
        top_k: int = 10,
        include_coverage: bool = False,
        utilities: np.ndarray | list[float] | None = None,
    ) -> dict[Any, list[str]]:
        """Return ``{point_id: [tags...]}`` from the three selection mechanisms at their defaults.

        Tags are ``SpectralAttention``, ``SemanticBridge``, ``SupportedDiversity``
        and, with ``include_coverage``, ``DiverseCoverage``. ``top_k`` is the budget
        of the budgeted mechanisms (Supported Diversity, coverage); Spectral
        Attention and Semantic Bridges have bounded outputs of their own.
        ``utilities`` is passed to Supported Diversity (unit utilities if None).
        """
        tags: dict[Any, list[str]] = {}
        if self.n == 0:
            return tags

        def add(indices: Any, tag: str) -> None:
            for i in indices:
                tags.setdefault(self.ordered_ids[int(i)], []).append(tag)

        add((c["index"] for c in self.spectral_attention()["selected_chunks"]), "SpectralAttention")
        add((c["index"] for c in self.semantic_bridges()["selected_chunks"]), "SemanticBridge")
        budget = min(top_k, self.n)
        add(
            self.supported_diversity(utilities, budget=budget)["selected_indices"],
            "SupportedDiversity",
        )
        if include_coverage:
            add(self.diverse_coverage(budget=budget)["indices"], "DiverseCoverage")
        return tags
