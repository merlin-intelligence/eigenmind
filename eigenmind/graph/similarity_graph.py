"""``SimilarityGraph`` — high-level API around the graph math primitives.

Holds a similarity matrix W and exposes the four ranking strategies (singular,
hinge, theta, eigenvalue analysis) without recomputing W or the conflict mask
every time. The lower-level pure functions in
:mod:`eigenmind.graph.singular`, :mod:`eigenmind.graph.connectivity` and
:mod:`eigenmind.graph.theta` remain available and are still directly tested.
"""
from __future__ import annotations

import numpy as np

from eigenmind.config import (
    SIMILARITY_THRESHOLD,
    THETA_CONFLICT_THRESHOLD,
    THETA_DIAG_SHIFT,
    THETA_MAX_ITERS,
    THETA_RANK,
    THETA_STEP0,
)
from eigenmind.graph.connectivity import (
    compute_linf_connectivity_optimizer,
    rank_relevant_but_not_obvious_chunks,
)
from eigenmind.graph.singular import (
    analyze_laplacian_eigenvectors,
    build_similarity_matrix,
    find_singular_chunks,
    normalized_laplacian,
)
from eigenmind.graph.theta import (
    build_conflict_mask,
    theta_diversity_picker,
    theta_subgradient_approximation,
)


class SimilarityGraph:
    """A weighted graph over chunks, built from their embedding matrix.

    Memoizes the similarity matrix, the ℓ∞ optimizer, the conflict mask and the
    θ low-rank factor so clients that ask for several rankings only pay the cost
    once.
    """

    def __init__(
        self,
        retrieved_points: list,
        threshold: float = SIMILARITY_THRESHOLD,
    ):
        self.points = retrieved_points
        self.threshold = threshold
        self.id_to_point = {p.id: p for p in retrieved_points}
        self.ordered_ids = [p.id for p in retrieved_points]
        self.embedding_matrix = (
            np.array([p.vector for p in retrieved_points])
            if retrieved_points
            else np.empty((0, 0))
        )
        self.W = build_similarity_matrix(self.embedding_matrix, threshold) if len(retrieved_points) else np.empty((0, 0))
        self._x_star: np.ndarray | None = None
        self._conflict_mask: np.ndarray | None = None
        self._theta_factor: np.ndarray | None = None

    # ─── basic linear algebra ──────────────────────────────────────

    @property
    def n(self) -> int:
        return self.W.shape[0]

    def laplacian(self) -> np.ndarray:
        return normalized_laplacian(self.W)

    # ─── strategy 1: singular chunks (eigenvector poles) ───────────

    def singular_chunks(self) -> list:
        """Points lying at the +/- poles of the first k Laplacian eigenvectors."""
        return find_singular_chunks(self.points)

    def eigenvalue_analysis(self, output_filename_base: str):
        """Persist the eigenvalue plot and return the per-chunk EV pole tags."""
        return analyze_laplacian_eigenvectors(self.W, self.id_to_point, self.ordered_ids, output_filename_base)

    # ─── strategy 2: ℓ∞ connectivity + hinge ranking ───────────────

    def linf_optimizer(self) -> np.ndarray:
        """Memoized ``x*`` — the ℓ∞-connectivity optimizer."""
        if self._x_star is None:
            self._x_star = compute_linf_connectivity_optimizer(self.W)
        return self._x_star

    def hinge_ranking(self, pole_quantile: float = 0.90):
        """Ranking of relevant-but-not-obvious chunks. See :func:`rank_relevant_but_not_obvious_chunks`."""
        return rank_relevant_but_not_obvious_chunks(self.W, self.linf_optimizer(), pole_quantile)

    # ─── strategy 3: Lovász θ + diversity ──────────────────────────

    def conflict_mask(self, threshold: float = THETA_CONFLICT_THRESHOLD) -> np.ndarray:
        if self._conflict_mask is None:
            self._conflict_mask = build_conflict_mask(self.W, threshold)
        return self._conflict_mask

    def theta_factor(
        self,
        max_iters: int = THETA_MAX_ITERS,
        step0: float = THETA_STEP0,
        diag_shift: float = THETA_DIAG_SHIFT,
        rank: int = THETA_RANK,
    ) -> np.ndarray:
        """Memoized low-rank factor Y from the θ approximation."""
        if self._theta_factor is None:
            _, _, Y, _, _, _ = theta_subgradient_approximation(
                self.conflict_mask(),
                max_iters=max_iters, step0=step0, diag_shift=diag_shift, rank=rank,
            )
            self._theta_factor = Y
        return self._theta_factor

    def theta_diversity(self, k: int = 10, seed: str = "leverage") -> list[int]:
        """Indices of ``k`` diverse chunks in the conflict graph."""
        selected, _ = theta_diversity_picker(self.theta_factor(), k=k, seed=seed)
        return selected

    # ─── consensus across strategies ───────────────────────────────

    def selection_tags(self, top_k: int = 10) -> dict:
        """Return ``{chunk_id: [tags...]}`` aggregating singular/hinge/theta selections."""
        tags: dict = {}
        if self.n == 0:
            return tags

        for p in self.singular_chunks():
            tags.setdefault(p.id, []).append("Singular")

        if self.n > 2:
            ranking, _, _, _, _ = self.hinge_ranking()
            for r in ranking[:top_k]:
                tags.setdefault(self.ordered_ids[r[0]], []).append("Hinge")
            for idx in self.theta_diversity(k=top_k):
                tags.setdefault(self.ordered_ids[idx], []).append("Theta")

        return tags
