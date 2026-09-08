"""eigenmind — eigenvalue-based graph computations for semantic chunk retrieval.

Public API re-exported here for convenience::

    from eigenmind import SimilarityGraph
"""

from __future__ import annotations

from eigenmind.connectivity import (
    compute_linf_connectivity_optimizer,
    rank_relevant_but_not_obvious_chunks,
)
from eigenmind.similarity_graph import SimilarityGraph
from eigenmind.singular import (
    analyze_laplacian_eigenvectors,
    build_similarity_matrix,
    find_singular_chunks,
    normalized_laplacian,
)
from eigenmind.theta import (
    build_conflict_mask,
    theta_diversity_picker,
    theta_subgradient_approximation,
)

__all__ = [
    "SimilarityGraph",
    "analyze_laplacian_eigenvectors",
    "build_conflict_mask",
    "build_similarity_matrix",
    "compute_linf_connectivity_optimizer",
    "find_singular_chunks",
    "normalized_laplacian",
    "rank_relevant_but_not_obvious_chunks",
    "theta_diversity_picker",
    "theta_subgradient_approximation",
]
