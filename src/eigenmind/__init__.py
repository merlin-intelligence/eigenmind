"""eigenmind — eigenvalue-based graph computations for semantic selection.

Version 1.8 provides three selection mechanisms:
  1. Spectral Attention (component representatives, spectral vertices and medoids)
  2. Semantic Bridges (spanning-tree bridges between connected components)
  3. Supported Diversity (budgeted stable set of supported chunks, exact MILP)
together with graph construction, evaluation baselines and Gaussian-kernel
diverse coverage.

    from eigenmind import SimilarityGraph, spectral_attention, semantic_bridges

The pre-1.8 functions (singular chunks, hinge ranking, θ diversity) remain
importable with their original behaviour; calling them emits a
DeprecationWarning. They will be removed in 2.0.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from eigenmind import defaults

# Deprecated pre-1.8 API (behaviour unchanged; calls emit DeprecationWarning)
from eigenmind.connectivity import (
    compute_linf_connectivity_optimizer,
    rank_relevant_but_not_obvious_chunks,
)

# Diverse coverage (Gaussian-kernel facility location)
from eigenmind.coverage import build_gaussian_kernel, diverse_coverage

# Graph construction
from eigenmind.graph_construction import (
    cosine_matrix,
    graph_summary,
    knn_graph,
    normalize_embeddings,
    quantile_threshold,
    threshold_graph,
)

# Evaluation baselines
from eigenmind.selection_baselines import facility_location, farthest_first, kdpp_greedy, mmr

# Selection mechanisms
from eigenmind.semantic_bridges import component_affinity, semantic_bridges

# High-level API
from eigenmind.similarity_graph import SimilarityGraph
from eigenmind.singular import (
    analyze_laplacian_eigenvectors,
    build_similarity_matrix,
    find_singular_chunks,
    normalized_laplacian,
)
from eigenmind.spectral_attention import (
    adjacency_from_cosine,
    component_spectrum,
    medoids,
    spa,
    spectral_attention,
)
from eigenmind.supported_diversity import supported_diversity
from eigenmind.theta import (
    build_conflict_mask,
    theta_diversity_picker,
    theta_subgradient_approximation,
)

try:
    __version__ = version("eigenmind")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "unknown"

__all__ = [
    "__version__",
    "defaults",
    # High-level
    "SimilarityGraph",
    # Selection mechanisms
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
    # Deprecated (removal in 2.0)
    "build_similarity_matrix",
    "normalized_laplacian",
    "find_singular_chunks",
    "analyze_laplacian_eigenvectors",
    "compute_linf_connectivity_optimizer",
    "rank_relevant_but_not_obvious_chunks",
    "build_conflict_mask",
    "theta_subgradient_approximation",
    "theta_diversity_picker",
]
