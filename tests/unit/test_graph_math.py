"""Smoke tests on the graph math modules — pure numpy, no Qdrant or Streamlit needed.

Run with: pytest tests/unit/
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from eigenmind.graph.connectivity import (
    compute_linf_connectivity_optimizer,
    rank_relevant_but_not_obvious_chunks,
)
from eigenmind.graph.similarity_graph import SimilarityGraph
from eigenmind.graph.singular import build_similarity_matrix, normalized_laplacian
from eigenmind.graph.theta import (
    build_conflict_mask,
    theta_diversity_picker,
    theta_subgradient_approximation,
)


def _toy_embeddings(seed: int = 0, n: int = 8, d: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, d))
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    return M


def test_similarity_matrix_is_symmetric_with_zero_diagonal():
    W = build_similarity_matrix(_toy_embeddings(), threshold=0.0)
    assert W.shape == (8, 8)
    assert np.allclose(np.diag(W), 0.0)
    assert np.allclose(W, W.T)


def test_normalized_laplacian_is_symmetric():
    W = build_similarity_matrix(_toy_embeddings(), threshold=0.0) + 0.1
    np.fill_diagonal(W, 0)
    L = normalized_laplacian(W)
    assert np.allclose(L, L.T, atol=1e-10)


def test_linf_optimizer_in_unit_interval_or_zero():
    W = build_similarity_matrix(_toy_embeddings(seed=1), threshold=0.0) + 0.1
    np.fill_diagonal(W, 0)
    x = compute_linf_connectivity_optimizer(W)
    assert x.shape == (8,)
    assert np.max(np.abs(x)) <= 1.0 + 1e-9


def test_rank_returns_one_entry_per_node():
    W = build_similarity_matrix(_toy_embeddings(seed=2), threshold=0.0) + 0.1
    np.fill_diagonal(W, 0)
    x = compute_linf_connectivity_optimizer(W)
    ranking, _, _, _, _ = rank_relevant_but_not_obvious_chunks(W, x)
    assert len(ranking) == W.shape[0]
    assert ranking[0][1] >= ranking[-1][1]  # sorted descending by hinge


def test_theta_pipeline_runs_end_to_end():
    W = build_similarity_matrix(_toy_embeddings(seed=3), threshold=0.0) + 0.1
    np.fill_diagonal(W, 0)
    H = build_conflict_mask(W, threshold=0.5)
    assert np.all(np.diag(H) == 0)
    assert np.array_equal(H, H.T)

    _, _, Y, _, _, _ = theta_subgradient_approximation(H, max_iters=20)
    selected, _ = theta_diversity_picker(Y, k=4)
    assert len(set(selected)) == 4  # picker returns distinct indices


def _toy_points(seed: int = 0, n: int = 8):
    """Build mock points compatible with what Qdrant returns: id, vector, payload."""
    M = _toy_embeddings(seed=seed, n=n)
    return [
        SimpleNamespace(
            id=i,
            vector=M[i].tolist(),
            payload={"filename": f"f{i}.pdf", "chunk_number": i, "text": f"chunk {i} text"},
        )
        for i in range(n)
    ]


def test_similarity_graph_memoizes_x_star_and_theta_factor():
    points = _toy_points(seed=4)
    g = SimilarityGraph(points, threshold=0.0)
    assert g.n == 8
    # First call computes, second returns memoized array
    x1 = g.linf_optimizer()
    x2 = g.linf_optimizer()
    assert x1 is x2  # same object reference proves memoization

    Y1 = g.theta_factor(max_iters=10)
    Y2 = g.theta_factor(max_iters=10)
    assert Y1 is Y2


def test_similarity_graph_selection_tags_aggregates_methods():
    points = _toy_points(seed=5)
    g = SimilarityGraph(points, threshold=0.0)
    tags = g.selection_tags(top_k=4)
    # Some chunks should have at least one tag
    assert any(len(v) > 0 for v in tags.values())
    # Tags must be a subset of the three known methods
    allowed = {"Singular", "Hinge", "Theta"}
    for vs in tags.values():
        assert set(vs) <= allowed
