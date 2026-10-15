"""Default parameter values for Eigenmind graph algorithms.

Every public function accepts these as overridable keyword arguments. The
selection defaults below match the module-level defaults of the three
mechanisms, so ``SimilarityGraph`` and the direct functions behave alike.
"""

from __future__ import annotations

# ─── Graph construction ────────────────────────────────────────────────────
# Cosine threshold used by SimilarityGraph (unchanged since 0.1).
SIMILARITY_THRESHOLD = 0.65

# ─── Spectral Attention ───────────────────────────────────────────────────
SPECTRAL_POSITIVE_COUNT = 10
SPECTRAL_SELECTION = "vertices"  # "vertices" | "medoids" | "tails", or a tuple of these
SPECTRAL_ALLOCATION = "global"  # "global" | "per_component"
SPECTRAL_LAPLACIAN = "normalized"  # "normalized" | "unnormalized"
SPECTRAL_MIN_RELATIVE_GAP = 0.05
SPECTRAL_MAX_GAP_EXTENSION = None  # None: at most positive_count extra directions
SPECTRAL_MAX_PER_SELECTION = 5
SPECTRAL_DEGREE_REGULARIZATION = 0.0

# ─── Semantic Bridges ──────────────────────────────────────────────────────
BRIDGES_MODE = "spanning_tree"  # "spanning_tree" | "top_m" | "all_pairs"
BRIDGES_TOP_M = 1  # used only by mode="top_m"
BRIDGES_MIN_COMPONENT_SIZE = 2
BRIDGES_MIN_SIMILARITY = None  # None: no similarity filter

# ─── Supported Diversity ───────────────────────────────────────────────────
DIVERSITY_BUDGET = 10
DIVERSITY_MIN_DEGREE = 0.0
DIVERSITY_RELAXATION = "cliques"  # "cliques" | "edges"
DIVERSITY_SOLVER = "milp"  # "milp" | "lp_greedy"
DIVERSITY_TIME_LIMIT = 10.0  # seconds

# ─── Diverse coverage (Gaussian-kernel facility location) ─────────────────
COVERAGE_BANDWIDTH = 0.5
COVERAGE_RELEVANCE_WEIGHT = 0.25
