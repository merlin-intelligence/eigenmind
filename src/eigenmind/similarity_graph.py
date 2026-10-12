"""``SimilarityGraph`` — high-level API around Eigenmind v1.8 selection methods.

Holds cosine matrix C and adjacency matrix W, exposing the three v1.8 mechanisms:
  1. Spectral Attention (spectral vertices / medoids / tails)
  2. Semantic Bridges (MST inter-component bridges)
  3. Supported Diversity (MILP exact solver with clique cover / LP greedy)
alongside graph construction utilities and baseline selection comparison methods.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eigenmind.coverage import (
    build_gaussian_kernel,
    diverse_coverage as compute_diverse_coverage,
)
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
    SPECTRAL_MIN_COMPONENT_SIZE,
    SPECTRAL_MIN_RELATIVE_GAP,
    SPECTRAL_POSITIVE_COUNT,
    SPECTRAL_SELECTION,
    SPECTRAL_SPA_RTOL,
    SPECTRAL_SPA_TIE_RTOL,
)
from eigenmind.graph_construction import (
    cosine_matrix,
    graph_summary,
    knn_graph,
    normalize_embeddings,
    quantile_threshold,
    threshold_graph,
)
from eigenmind.selection_baselines import (
    facility_location as baseline_facility_location,
    farthest_first as baseline_farthest_first,
    kdpp_greedy as baseline_kdpp_greedy,
    mmr as baseline_mmr,
)
from eigenmind.semantic_bridges import semantic_bridges
from eigenmind.spectral_attention import (
    adjacency_from_cosine,
    component_spectrum,
    spectral_attention,
)
from eigenmind.supported_diversity import supported_diversity


class SimilarityGraph:
    """A weighted graph over chunks, built from their embedding matrix.

    Memoizes the cosine matrix C, adjacency matrix W, and selection results
    so clients running multiple strategies pay computation costs only once.
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

        # Memoization caches
        self._spectral_attention_cache: dict[str, Any] = {}
        self._semantic_bridges_cache: dict[str, Any] = {}
        self._supported_diversity_cache: dict[str, Any] = {}
        self._K: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.W.shape[0]

    def summary(self) -> dict[str, Any]:
        """Summary statistics of graph topology (components, isolates, sizes)."""
        return graph_summary(self.W)

    def laplacian(self, laplacian: str = SPECTRAL_LAPLACIAN) -> tuple[np.ndarray, np.ndarray]:
        """Eigenvalue decomposition of graph components."""
        return component_spectrum(self.W, laplacian=laplacian)

    # ─── 1. Spectral Attention ─────────────────────────────────────────────────

    def spectral_attention(
        self,
        positive_count: int = SPECTRAL_POSITIVE_COUNT,
        allocation: str = SPECTRAL_ALLOCATION,
        min_relative_gap: float = SPECTRAL_MIN_RELATIVE_GAP,
        max_gap_extension: int | None = None,
        selection: str | tuple[str, ...] = SPECTRAL_SELECTION,
        max_per_selection: int = SPECTRAL_MAX_PER_SELECTION,
        laplacian: str = SPECTRAL_LAPLACIAN,
        degree_regularization: float = SPECTRAL_DEGREE_REGULARIZATION,
        chunks: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run Spectral Attention to select structural thematic representatives."""
        if self.n == 0:
            return {"selected_chunks": [], "n_components": 0, "config": {}}

        cache_key = f"{positive_count}_{allocation}_{selection}_{laplacian}_{degree_regularization}"
        if cache_key not in self._spectral_attention_cache:
            chunks_meta = chunks if chunks is not None else [str(pid) for pid in self.ordered_ids]
            self._spectral_attention_cache[cache_key] = spectral_attention(
                self.W,
                chunks=chunks_meta,
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
        return self._spectral_attention_cache[cache_key]

    # ─── 2. Semantic Bridges ───────────────────────────────────────────────────

    def semantic_bridges(
        self,
        mode: str = BRIDGES_MODE,
        top_m: int = BRIDGES_TOP_M,
        min_component_size: int = BRIDGES_MIN_COMPONENT_SIZE,
        min_similarity: float = BRIDGES_MIN_SIMILARITY,
        max_bridges: int | None = None,
        chunks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run Semantic Bridges to identify cross-component linking points."""
        if self.n < 2:
            return {"component_bridges": [], "selected_chunks": [], "n_components": self.n}

        cache_key = f"{mode}_{top_m}_{min_similarity}_{max_bridges}"
        if cache_key not in self._semantic_bridges_cache:
            chunks_meta = chunks if chunks is not None else [str(pid) for pid in self.ordered_ids]
            self._semantic_bridges_cache[cache_key] = semantic_bridges(
                self.W,
                self.C,
                chunks=chunks_meta,
                mode=mode,
                top_m=top_m,
                min_component_size=min_component_size,
                min_similarity=min_similarity,
                max_bridges=max_bridges,
            )
        return self._semantic_bridges_cache[cache_key]

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
        """Run Supported Diversity (MILP / greedy) with neighborhood support constraints."""
        if self.n == 0:
            return {"selected_indices": [], "status": "empty", "selected_utility": 0.0}

        u = np.ones(self.n, dtype=float) if utilities is None else np.asarray(utilities, dtype=float)
        chunks_meta = chunks if chunks is not None else [str(pid) for pid in self.ordered_ids]
        return supported_diversity(
            self.W,
            u,
            budget=budget,
            chunks=chunks_meta,
            min_degree=min_degree,
            relaxation=relaxation,
            solver=solver,
            time_limit=time_limit,
        )

    # ─── Baselines ────────────────────────────────────────────────────────────

    def mmr(self, relevance: np.ndarray | list[float], budget: int = 10, lam: float = 0.5) -> list[int]:
        """Maximal Marginal Relevance baseline."""
        return baseline_mmr(relevance, self.C, budget=budget, lam=lam)

    def facility_location(
        self,
        budget: int = 10,
        weights: np.ndarray | list[float] | None = None,
        utilities: np.ndarray | list[float] | None = None,
        lam: float = 0.0,
    ) -> list[int]:
        """Facility location submodular greedy baseline."""
        return baseline_facility_location(self.C, budget=budget, weights=weights, utilities=utilities, lam=lam)

    def kdpp_greedy(self, budget: int = 10, quality: np.ndarray | list[float] | None = None) -> list[int]:
        """k-DPP greedy MAP baseline."""
        return baseline_kdpp_greedy(self.C, budget=budget, quality=quality)

    def farthest_first(self, budget: int = 10, start: int | None = None) -> list[int]:
        """Farthest-first geometric traversal baseline."""
        return baseline_farthest_first(self.C, budget=budget, start=start)

    # ─── Diverse Coverage (Kernel-based) ──────────────────────────────────────

    def kernel(self, bandwidth: float = COVERAGE_BANDWIDTH) -> np.ndarray:
        """Memoized continuous Gaussian chord kernel K (diagonal 1.0)."""
        if self._K is None:
            if self.n > 0:
                self._K = build_gaussian_kernel(self.embedding_matrix, bandwidth=bandwidth)
            else:
                self._K = np.empty((0, 0))
        return self._K

    def diverse_coverage(
        self,
        budget: int = 10,
        relevance: np.ndarray | list[float] | None = None,
        weights: np.ndarray | list[float] | None = None,
        relevance_weight: float = COVERAGE_RELEVANCE_WEIGHT,
        bandwidth: float = COVERAGE_BANDWIDTH,
    ) -> dict[str, Any]:
        """Submodular facility-location coverage over Gaussian kernel."""
        if self.n == 0:
            return {
                "indices": np.array([], dtype=int),
                "marginal_gains": np.array([], dtype=float),
                "coverage": np.array([], dtype=float),
                "objective": 0.0,
                "upper_bound": 0.0,
            }
        K = self.kernel(bandwidth=bandwidth)
        return compute_diverse_coverage(
            K,
            budget=budget,
            relevance=relevance,
            weights=weights,
            relevance_weight=relevance_weight,
        )

    # ─── Tag Aggregation Across Methods ───────────────────────────────────────

    def selection_tags(self, top_k: int = 10, include_coverage: bool = False) -> dict[Any, list[str]]:
        """Return {point_id: [tags...]} aggregating v1.8 selection mechanisms."""
        tags: dict[Any, list[str]] = {}
        if self.n == 0:
            return tags

        # 1. Spectral Attention
        sa_res = self.spectral_attention(positive_count=top_k)
        for c in sa_res.get("selected_chunks", []):
            pid = self.ordered_ids[c["index"]]
            tags.setdefault(pid, []).append("SpectralAttention")

        # 2. Semantic Bridges
        if self.n > 2:
            sb_res = self.semantic_bridges(top_m=1)
            for c in sb_res.get("selected_chunks", []):
                pid = self.ordered_ids[c["index"]]
                tags.setdefault(pid, []).append("SemanticBridge")

        # 3. Supported Diversity
        sd_res = self.supported_diversity(budget=min(top_k, self.n))
        for idx in sd_res.get("selected_indices", []):
            pid = self.ordered_ids[idx]
            tags.setdefault(pid, []).append("SupportedDiversity")

        # Optional Diverse Coverage
        if include_coverage:
            cov = self.diverse_coverage(budget=top_k)
            for idx in cov["indices"]:
                pid = self.ordered_ids[idx]
                tags.setdefault(pid, []).append("DiverseCoverage")

        return tags

    # ─── Backwards Compatibility Aliases ──────────────────────────────────────

    def singular_chunks(self) -> list[Any]:
        """Compatibility shim: returns points selected by Spectral Attention."""
        sa_res = self.spectral_attention()
        indices = {c["index"] for c in sa_res.get("selected_chunks", [])}
        return [self.points[i] for i in sorted(indices)]

    def theta_diversity(self, k: int = 10, **kwargs: Any) -> list[int]:
        """Compatibility shim: maps theta_diversity to supported_diversity indices."""
        sd_res = self.supported_diversity(budget=k)
        return sd_res.get("selected_indices", [])
