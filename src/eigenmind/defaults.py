"""Default parameter values for Eigenmind graph algorithms (v1.8).

Every public function accepts these as overridable keyword arguments; the
values below are the ones the algorithms are originally tuned against.
"""

from __future__ import annotations

# ─── Graph Construction & Cosine Thresholds ────────────────────────────────
SIMILARITY_THRESHOLD = 0.5
DEFAULT_TAU = 0.5
DEFAULT_QUANTILE_Q = 0.99
DEFAULT_KNN_K = 10
DEFAULT_KNN_MUTUAL = True

# ─── Spectral Attention ───────────────────────────────────────────────────
SPECTRAL_POSITIVE_COUNT = 10
SPECTRAL_SELECTION = "vertices"  # "vertices" | "medoids" | "tails"
SPECTRAL_ALLOCATION = "global"   # "global" | "per_component"
SPECTRAL_LAPLACIAN = "normalized"  # "normalized" | "combinatorial"
SPECTRAL_MIN_COMPONENT_SIZE = 2
SPECTRAL_MIN_RELATIVE_GAP = 0.05
SPECTRAL_MAX_GAP_EXTENSION = 10
SPECTRAL_MAX_PER_SELECTION = 5
SPECTRAL_DEGREE_REGULARIZATION = 0.0
SPECTRAL_SPA_RTOL = 1e-8
SPECTRAL_SPA_TIE_RTOL = 1e-9

# ─── Semantic Bridges ──────────────────────────────────────────────────────
BRIDGES_MODE = "spanning_tree"  # "spanning_tree" | "top_m" | "all_pairs"
BRIDGES_TOP_M = 1
BRIDGES_MIN_COMPONENT_SIZE = 2
BRIDGES_MIN_SIMILARITY = 0.0

# ─── Supported Diversity ───────────────────────────────────────────────────
DIVERSITY_BUDGET = 10
DIVERSITY_MIN_DEGREE = 0.0
DIVERSITY_RELAXATION = "cliques"  # "cliques" | "edges"
DIVERSITY_SOLVER = "milp"         # "milp" | "lp_greedy"
DIVERSITY_TIME_LIMIT = 5.0

# ─── Legacy / Compatibility Defaults ───────────────────────────────────────
THETA_CONFLICT_THRESHOLD = SIMILARITY_THRESHOLD
THETA_RANK = 24
THETA_MAX_ITERS = 400
THETA_STEP0 = 0.25
THETA_DIAG_SHIFT = 1e-2

COVERAGE_BANDWIDTH = 0.5
COVERAGE_RELEVANCE_WEIGHT = 0.25
