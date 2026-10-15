"""eigenmind — eigenvalue-based graph computations for semantic selection.

Eigenmind provides three selection mechanisms:
  1. Spectral Attention (component representatives, spectral vertices and medoids)
  2. Semantic Bridges (spanning-tree bridges between connected components)
  3. Supported Diversity (budgeted stable set of supported chunks, exact MILP)
together with graph construction, evaluation baselines and Gaussian-kernel
diverse coverage.

    from eigenmind import SimilarityGraph, spectral_attention, semantic_bridges
"""

from __future__ import annotations

from eigenmind import defaults
from eigenmind._version import __version__

# Diverse coverage (Gaussian-kernel facility location)
from eigenmind.coverage import build_gaussian_kernel, diverse_coverage

# Graph construction
from eigenmind.graph_construction import (
    adjacency_from_cosine,
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
from eigenmind.spectral_attention import (
    component_spectrum,
    medoids,
    spa,
    spectral_attention,
)
from eigenmind.supported_diversity import supported_diversity

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
]
