"""Monotone submodular diverse coverage with facility location and modular relevance.

Implementation of Method 3A from the EigenMind corrected methods specification:
- Continuous Gaussian chord kernel over unit-normalized embeddings.
- Greedy forward selection optimizing normalized facility-location coverage plus relevance.
- Provable (1 - 1/e) approximation ratio under cardinality constraints (Nemhauser et al., 1978).
- Analytical online upper bound min(U1, U2) on the global optimal value.
"""

from __future__ import annotations

import numpy as np

from eigenmind.defaults import COVERAGE_BANDWIDTH, COVERAGE_RELEVANCE_WEIGHT


def _count(value: int, name: str, lo: int, hi: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")
    return int(value)


def _nonnegative(value: float, name: str) -> float:
    val = float(value)
    if not np.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {val}")
    return val


def _vector(
    value: np.ndarray | list[float] | None, n: int, name: str, default: float = 0.0
) -> np.ndarray:
    x = np.full(n, default, dtype=float) if value is None else np.asarray(value, dtype=float)
    if x.shape != (n,) or not np.isfinite(x).all() or (x < 0.0).any():
        raise ValueError(f"{name} must be a finite nonnegative vector of length {n}")
    return x.copy()


def build_gaussian_kernel(
    embedding_matrix: np.ndarray,
    bandwidth: float = COVERAGE_BANDWIDTH,
) -> np.ndarray:
    """Compute dense Gaussian chord kernel K_ij = exp(-(1 - C_ij) / sigma^2) with K_ii = 1.

    Parameters
    ----------
    embedding_matrix : np.ndarray
        Array of shape (n, d) containing row-wise embedding vectors.
        Vectors are validated to have positive finite norms and are unit-normalized.
    bandwidth : float, default=0.5
        Kernel bandwidth parameter sigma > 0.

    Returns
    -------
    np.ndarray
        Dense nonnegative kernel matrix K of shape (n, n) with unit diagonal.
    """
    V = np.asarray(embedding_matrix, dtype=float)
    if V.ndim != 2 or min(V.shape) == 0 or not np.isfinite(V).all():
        raise ValueError("embedding_matrix must be a finite nonempty 2D array")

    norms = np.linalg.norm(V, axis=1)
    if (norms <= 1e-12).any() or not np.isfinite(norms).all():
        raise ValueError("embedding norms must be finite and exceed 1e-12")

    bandwidth = _nonnegative(bandwidth, "bandwidth")
    if bandwidth <= 1e-12:
        raise ValueError("bandwidth must be strictly positive")

    # Unit-normalize vectors
    V_norm = V / norms[:, None]
    C = np.clip(V_norm @ V_norm.T, -1.0, 1.0)
    K = np.exp(-(1.0 - C) / (bandwidth**2))
    np.fill_diagonal(K, 1.0)
    return K


def diverse_coverage(
    K: np.ndarray,
    budget: int,
    relevance: np.ndarray | list[float] | None = None,
    weights: np.ndarray | list[float] | None = None,
    relevance_weight: float = COVERAGE_RELEVANCE_WEIGHT,
) -> dict:
    """Greedy normalized facility-location coverage plus modular relevance (Method 3A).

    Maximizes the monotone submodular set function under a cardinality budget b:
        F(S) = sum_{j in T} a_j * max_{i in S} K_{ji} + eta * sum_{i in S} r_i
    subject to |S| <= b.

    Guarantees:
    - (1 - 1/e) approximation ratio: F(S_greedy) >= (1 - (1 - 1/b)^b) * F(S*) >= (1 - 1/e) * F(S*).
    - Returns an analytical online upper bound min(U1, U2) bounding F(S*).

    Parameters
    ----------
    K : np.ndarray
        Nonnegative coverage affinity matrix of shape (m, n), where K[j, i] describes
        how well candidate chunk i covers target item j. Rectangular matrices are supported.
    budget : int
        Maximum number of candidate chunks to select (0 <= budget <= n).
    relevance : array-like of shape (n,), optional
        Nonnegative modular relevance/utility score for each candidate chunk.
        Defaults to zeros (pure coverage objective).
    weights : array-like of shape (m,), optional
        Target weights a_j >= 0. Automatically normalized to sum to 1.0.
        Defaults to uniform weights 1/m.
    relevance_weight : float, default=0.25
        Nonnegative trade-off weight eta balancing coverage against relevance.

    Returns
    -------
    dict with keys:
        'indices' : np.ndarray
            Integer indices of selected candidate chunks, ordered by selection step.
        'marginal_gains' : np.ndarray
            Marginal objective gains Delta(i | S) at each greedy selection step.
        'coverage' : np.ndarray
            Vector of length m containing the final coverage achieved for each target.
        'objective' : float
            Final objective value F(S).
        'upper_bound' : float
            Provable analytical upper bound on the optimal value F(S*).
    """
    K_mat = np.asarray(K, dtype=float)
    if (
        K_mat.ndim != 2
        or min(K_mat.shape) == 0
        or not np.isfinite(K_mat).all()
        or (K_mat < 0.0).any()
    ):
        raise ValueError("K must be a finite nonnegative nonempty 2D matrix")

    m, n = K_mat.shape
    b = _count(budget, "budget", 0, n)
    r = _vector(relevance, n, "relevance")
    a = _vector(weights, m, "weights", default=1.0)
    if a.sum() <= 0.0:
        raise ValueError("target weights must have a strictly positive sum")
    a = a / a.sum()
    eta = _nonnegative(relevance_weight, "relevance_weight")

    covered = np.zeros(m, dtype=float)
    available = np.ones(n, dtype=bool)
    selected: list[int] = []
    gains: list[float] = []

    for _ in range(b):
        # Marginal gain for unselected candidate i:
        # Delta(i | S) = sum_j a_j * max(K_{ji} - covered_j, 0) + eta * r_i
        marginal = a @ np.maximum(K_mat - covered[:, None], 0.0) + eta * r
        masked_marginal = np.where(available, marginal, -np.inf)
        candidate = int(np.argmax(masked_marginal))

        # If best marginal gain is zero or negative, terminate early
        if masked_marginal[candidate] <= 0.0:
            break

        selected.append(candidate)
        gains.append(float(marginal[candidate]))
        available[candidate] = False
        covered = np.maximum(covered, K_mat[:, candidate])

    value = float(a @ covered + (eta * r[selected].sum() if selected else 0.0))

    # Analytical upper bound computation:
    # 1. Submodular marginal sum bound U1 = F(S) + sum_{l=1}^b Delta_(l)
    marginal_final = a @ np.maximum(K_mat - covered[:, None], 0.0) + eta * r
    remaining_gains = np.sort(marginal_final[available])[::-1]
    upper1 = value + float(remaining_gains[:b].sum()) if len(remaining_gains) else value

    # 2. Relaxed component-wise upper bound U2 = sum_j a_j max_i K_ji + eta sum_{l=1}^b r_(l)
    sorted_r = np.sort(r)[::-1]
    upper2 = float(a @ K_mat.max(axis=1) + eta * sorted_r[:b].sum())

    upper = 0.0 if b == 0 else max(value, min(upper1, upper2))

    return {
        "indices": np.array(selected, dtype=int),
        "marginal_gains": np.array(gains, dtype=float),
        "coverage": covered,
        "objective": value,
        "upper_bound": float(upper),
    }
