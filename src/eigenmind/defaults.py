"""Default parameter values for the graph algorithms.

Every public function accepts these as overridable keyword arguments; the
values below are the ones the algorithms were originally tuned against.
"""
from __future__ import annotations

SIMILARITY_THRESHOLD = 0.65

THETA_CONFLICT_THRESHOLD = SIMILARITY_THRESHOLD
THETA_RANK = 24
THETA_MAX_ITERS = 400
THETA_STEP0 = 0.25
THETA_DIAG_SHIFT = 1e-2
