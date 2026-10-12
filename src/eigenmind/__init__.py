"""eigenmind — eigenvalue-based graph computations for semantic selection.

Package v1.8 introduces the three refined selection mechanisms:
  1. Spectral Attention (thematic representatives, spectral vertices & medoids)
  2. Semantic Bridges (MST inter-component affinity bridges)
  3. Supported Diversity (MILP / LP-greedy with neighborhood support)

alongside upstream graph construction and downstream evaluation baselines.

Public API re-exported here for convenience::

    from eigenmind import (
        SimilarityGraph,
        spectral_attention,
        semantic_bridges,
        supported_diversity,
        graph_construction,
    )
"""

from __future__ import annotations

import numpy as np

# 1. High-level orchestrator
from eigenmind.similarity_graph import SimilarityGraph

# 2. Three core selection methods (v1.8)
from eigenmind.spectral_attention import (
    adjacency_from_cosine,
    component_spectrum,
    medoids,
    spa,
    spectral_attention,
)
from eigenmind.semantic_bridges import (
    component_affinity,
    semantic_bridges,
)
from eigenmind.supported_diversity import (
    supported_diversity,
)

# 3. Upstream graph construction
from eigenmind.graph_construction import (
    cosine_matrix,
    graph_summary,
    knn_graph,
    normalize_embeddings,
    quantile_threshold,
    threshold_graph,
)

# 4. Downstream selection baselines
from eigenmind.selection_baselines import (
    facility_location,
    farthest_first,
    kdpp_greedy,
    mmr,
)

# 5. Continuous Gaussian chord coverage
from eigenmind.coverage import (
    build_gaussian_kernel,
    diverse_coverage,
)


# ─── Backward compatibility shims ──────────────────────────────────────────

def build_similarity_matrix(embedding_matrix: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Backward-compatible alias for cosine threshold graph."""
    if embedding_matrix.size == 0:
        return np.empty((0, 0))
    norms = np.linalg.norm(embedding_matrix, axis=1)
    bad = np.where(np.abs(norms - 1.0) > 1e-3)[0]
    if len(bad):
        raise ValueError(
            "embedding_matrix rows must be unit-normalized (L2 norm ≈ 1); "
            f"{len(bad)} row(s) are not."
        )
    C = embedding_matrix @ embedding_matrix.T
    return threshold_graph(C, tau=threshold)


def normalized_laplacian(W: np.ndarray) -> np.ndarray:
    """Backward-compatible normalized Laplacian computation."""
    degrees = np.sum(W, axis=1)
    d_inv_sqrt = np.zeros_like(degrees, dtype=float)
    nz = degrees > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(degrees[nz])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return np.identity(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt


def find_singular_chunks(retrieved_points: list, W: np.ndarray | None = None, threshold: float = 0.5) -> list:
    """Backward-compatible singular chunks via spectral attention."""
    if len(retrieved_points) < 2:
        return []
    if W is None:
        emb = np.array([p.vector for p in retrieved_points], dtype=float)
        W = build_similarity_matrix(emb, threshold=threshold)
    res = spectral_attention(W)
    idx_set = {c["index"] for c in res.get("selected_chunks", [])}
    return [retrieved_points[i] for i in sorted(idx_set) if i < len(retrieved_points)]


__all__ = [
    # High-level
    "SimilarityGraph",
    # Three selection mechanisms
    "spectral_attention",
    "component_spectrum",
    "spa",
    "medoids",
    "adjacency_from_cosine",
    "semantic_bridges",
    "component_affinity",
    "supported_diversity",
    # Graph construction
    "normalize_embeddings",
    "cosine_matrix",
    "threshold_graph",
    "quantile_threshold",
    "knn_graph",
    "graph_summary",
    # Baselines
    "mmr",
    "facility_location",
    "kdpp_greedy",
    "farthest_first",
    # Coverage
    "build_gaussian_kernel",
    "diverse_coverage",
    # Backward compatibility
    "build_similarity_matrix",
    "normalized_laplacian",
    "find_singular_chunks",
]
