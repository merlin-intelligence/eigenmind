"""Unit and regression tests for Eigenmind modules.

Tests:
  - Graph construction (normalization, top-PC removal, cosine, quantiles, k-NN)
  - Spectral Attention (SPA vertices, spectral medoids, tails)
  - Semantic Bridges (spanning tree, top-m, component affinities)
  - Supported Diversity (exact MILP solver via HiGHS, LP-greedy fallback)
  - Selection baselines (MMR, Facility Location, k-DPP, Farthest-First)
  - High-level SimilarityGraph API

Run with: pytest
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from eigenmind import (
    SimilarityGraph,
    cosine_matrix,
    facility_location,
    farthest_first,
    graph_summary,
    kdpp_greedy,
    knn_graph,
    mmr,
    normalize_embeddings,
    quantile_threshold,
    semantic_bridges,
    spectral_attention,
    supported_diversity,
    threshold_graph,
)


def _toy_embeddings(seed: int = 0, n: int = 16, d: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, d))
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def _planted_two_clusters(seed: int = 1, n_per: int = 8, d: int = 16) -> np.ndarray:
    """Create two well-separated clusters on the unit sphere."""
    rng = np.random.default_rng(seed)
    c1 = np.zeros(d)
    c1[0] = 1.0
    c2 = np.zeros(d)
    c2[1] = 1.0
    v1 = c1 + 0.2 * rng.standard_normal((n_per, d))
    v2 = c2 + 0.2 * rng.standard_normal((n_per, d))
    V = np.vstack([v1, v2])
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def _toy_points(seed: int = 0, n: int = 12):
    """Mock points compatible with Qdrant retrieval format."""
    M = _toy_embeddings(seed=seed, n=n)
    return [
        SimpleNamespace(
            id=i,
            vector=M[i].tolist(),
            payload={"filename": f"doc_{i}.pdf", "chunk_id": i},
        )
        for i in range(n)
    ]


# ─── 1. Graph Construction ───────────────────────────────────────────────────


def test_normalize_embeddings_properties():
    V = _toy_embeddings(seed=2, n=10, d=8)
    U = normalize_embeddings(V, center=True, remove_top=1)
    assert U.shape == (10, 8)
    norms = np.linalg.norm(U, axis=1)
    assert np.allclose(norms, 1.0)


def test_cosine_and_threshold_graph():
    U = _toy_embeddings(seed=3, n=8, d=8)
    C = cosine_matrix(U)
    assert C.shape == (8, 8)
    assert np.allclose(C, C.T)
    assert np.allclose(np.diag(C), 1.0)

    W = threshold_graph(C, tau=0.3)
    assert np.allclose(np.diag(W), 0.0)
    assert np.allclose(W, W.T)
    assert np.all((W == 0.0) | (W > 0.3))


def test_quantile_and_knn_graph():
    U = _toy_embeddings(seed=4, n=12, d=8)
    C = cosine_matrix(U)
    tau = quantile_threshold(C, q=0.8)
    assert 0.0 <= tau < 1.0

    W_knn = knn_graph(C, k=3, mutual=True)
    assert W_knn.shape == (12, 12)
    assert np.allclose(W_knn, W_knn.T)
    assert np.allclose(np.diag(W_knn), 0.0)

    summary = graph_summary(W_knn)
    assert "components" in summary
    assert "isolates" in summary
    assert summary["n"] == 12


# ─── 2. Spectral Attention ────────────────────────────────────────────────────


def test_spectral_attention_vertices():
    V = _planted_two_clusters()
    C = cosine_matrix(V)
    W = threshold_graph(C, tau=0.5)

    res = spectral_attention(W, selection="vertices", positive_count=6)
    assert "selected_chunks" in res
    assert "config" in res
    selected = res["selected_chunks"]
    assert len(selected) > 0
    all_kinds = [r["kind"] for c in selected for r in c.get("reasons", [])]
    assert "spectral_vertex" in all_kinds


def test_spectral_attention_medoids_and_tails():
    V = _planted_two_clusters()
    C = cosine_matrix(V)
    W = threshold_graph(C, tau=0.5)

    # Medoids mode
    res_m = spectral_attention(W, selection="medoids", positive_count=4)
    assert len(res_m["selected_chunks"]) > 0

    # Tails mode
    res_t = spectral_attention(W, selection="tails", positive_count=4)
    assert len(res_t["selected_chunks"]) > 0


# ─── 3. Semantic Bridges ──────────────────────────────────────────────────────


def test_semantic_bridges_spanning_tree():
    V = _planted_two_clusters()  # 2 distinct clusters
    C = cosine_matrix(V)
    W = threshold_graph(C, tau=0.7)  # strict tau creates >= 2 components

    res = semantic_bridges(W, C, mode="spanning_tree")
    assert "component_bridges" in res
    assert "selected_chunks" in res
    bridges = res["component_bridges"]
    if res["n_components"] >= 2:
        assert len(bridges) >= 1
        b0 = bridges[0]
        assert "component_a" in b0
        assert "component_b" in b0
        assert "maximum_cosine_similarity" in b0
        assert b0["maximum_cosine_similarity"] > 0.0


# ─── 4. Supported Diversity ───────────────────────────────────────────────────


def test_supported_diversity_milp_solver():
    V = _toy_embeddings(seed=5, n=12, d=8)
    C = cosine_matrix(V)
    W = threshold_graph(C, tau=0.2)
    utilities = np.ones(12)

    res = supported_diversity(W, utilities, budget=4, solver="milp")
    assert "selected_indices" in res
    assert "status" in res
    assert len(res["selected_indices"]) == 4
    assert len(set(res["selected_indices"])) == 4
    assert res["selected_utility"] > 0.0


def test_supported_diversity_lp_greedy_fallback():
    V = _toy_embeddings(seed=6, n=10, d=8)
    C = cosine_matrix(V)
    W = threshold_graph(C, tau=0.2)
    utilities = np.linspace(1.0, 2.0, 10)

    res = supported_diversity(W, utilities, budget=3, solver="lp_greedy")
    assert len(res["selected_indices"]) == 3
    assert len(set(res["selected_indices"])) == 3


# ─── 5. Selection Baselines ───────────────────────────────────────────────────


def test_selection_baselines():
    V = _toy_embeddings(seed=7, n=10, d=8)
    C = cosine_matrix(V)
    relevance = np.linspace(0.1, 1.0, 10)

    # MMR
    s_mmr = mmr(relevance, C, budget=4, lam=0.5)
    assert len(s_mmr) == 4
    assert len(set(s_mmr)) == 4

    # Facility Location
    s_fl = facility_location(C, budget=4)
    assert len(s_fl) == 4

    # k-DPP Greedy MAP
    s_dpp = kdpp_greedy(C, budget=4)
    assert len(s_dpp) <= 4

    # Farthest First
    s_ff = farthest_first(C, budget=4)
    assert len(s_ff) == 4


# ─── 6. High-Level SimilarityGraph API ────────────────────────────────────────


def test_similarity_graph_v1_8_integration():
    points = _toy_points(seed=8, n=12)
    sg = SimilarityGraph(points, threshold=0.3)
    assert sg.n == 12

    # Graph summary
    summary = sg.summary()
    assert summary["n"] == 12

    # Spectral Attention
    sa = sg.spectral_attention(positive_count=4)
    assert "selected_chunks" in sa

    # Semantic Bridges
    sb = sg.semantic_bridges()
    assert "component_bridges" in sb

    # Supported Diversity
    sd = sg.supported_diversity(budget=4)
    assert 0 < len(sd["selected_indices"]) <= 4

    # Tag aggregation
    tags = sg.selection_tags(top_k=4, include_coverage=True)
    assert isinstance(tags, dict)
    valid_tags = {"SpectralAttention", "SemanticBridge", "SupportedDiversity", "DiverseCoverage"}
    for t_list in tags.values():
        assert set(t_list).issubset(valid_tags)
