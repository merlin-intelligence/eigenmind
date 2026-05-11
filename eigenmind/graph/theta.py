"""Lovász θ approximation (subgradient) + diversity picker on the conflict graph."""
from __future__ import annotations

import numpy as np

from eigenmind.config import (
    THETA_DIAG_SHIFT,
    THETA_MAX_ITERS,
    THETA_RANK,
    THETA_STEP0,
)


def build_conflict_mask(similarity_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """Conflict graph H: H_ij = 1 iff similarity ≥ threshold (no self-loops, symmetric)."""
    H = (similarity_matrix >= threshold).astype(np.int8)
    np.fill_diagonal(H, 0)
    return np.maximum(H, H.T)


def _chol_ok(A: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def theta_subgradient_approximation(
    conflict_mask: np.ndarray,
    max_iters: int = THETA_MAX_ITERS,
    step0: float = THETA_STEP0,
    diag_shift: float = THETA_DIAG_SHIFT,
    rank: int = THETA_RANK,
    verbose: bool = False,
):
    """Subgradient approximation of Lovász θ(H) via the Lagrangian dual.

    Returns (best_theta, X, Y, lam, mu, best_x):
      - X is a PSD primal proxy (n×n);
      - Y is a low-rank factor with X ≈ Y Y^T (used by the diversity picker).
    """
    n = conflict_mask.shape[0]
    lam = np.ones(n, dtype=float)
    mu = np.zeros((n, n), dtype=float)
    ones = np.ones(n, dtype=float)

    def build_A(lam_vec: np.ndarray, mu_mat: np.ndarray) -> np.ndarray:
        A = np.diag(lam_vec).astype(float)
        A = A + 0.5 * (mu_mat * conflict_mask)
        return 0.5 * (A + A.T)

    best_theta = np.inf
    best_A = None
    best_x = None

    for k in range(1, max_iters + 1):
        A = build_A(lam, mu)
        if not _chol_ok(A):
            A = A + diag_shift * np.eye(n)
            lam = lam + diag_shift
            if not _chol_ok(A):
                A = A + (10.0 * diag_shift) * np.eye(n)
                lam = lam + 10.0 * diag_shift

        b = ones + lam
        try:
            x = 0.5 * np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x = 0.5 * np.linalg.solve(A + (10.0 * diag_shift) * np.eye(n), b)

        theta_k = float(b @ x - x @ (A @ x))
        if theta_k < best_theta:
            best_theta = theta_k
            best_A = A.copy()
            best_x = x.copy()

        g = x - x * x
        h = np.outer(x, x) * conflict_mask
        t = step0 / np.sqrt(k)
        lam = np.clip(lam - t * g, -0.9, 1e3)
        mu = np.clip(mu + t * h, 0.0, 1e3)

    A = best_A
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        Ainv = np.linalg.pinv(A)
    Ainv = 0.5 * (Ainv + Ainv.T)

    tr = float(np.trace(Ainv))
    X = (np.eye(n) / n) if tr <= 1e-12 else (Ainv / tr)

    w, Q = np.linalg.eigh(X)
    w = np.maximum(w, 0.0)
    r = min(rank, n)
    idx = np.argsort(w)[-r:]
    Y = Q[:, idx] * np.sqrt(np.maximum(w[idx], 0.0))[None, :]

    return best_theta, X, Y, lam, mu, best_x


def theta_diversity_picker(
    Y: np.ndarray,
    k: int = 12,
    seed: str = "leverage",
    seed_index: int = 0,
    eps: float = 1e-12,
):
    """Greedy farthest-point selection on the low-rank factor Y.

    Returns (selected_indices, min_distances).
    """
    n, _ = Y.shape
    k = min(k, n)
    norms = np.linalg.norm(Y, axis=1)
    Yn = Y / np.maximum(norms[:, None], eps)

    if seed == "leverage":
        U, _, _ = np.linalg.svd(Y, full_matrices=False)
        leverage = np.sum(U * U, axis=1)
        first = int(np.argmax(leverage))
    elif seed == "norm":
        first = int(np.argmax(norms))
    elif seed == "index":
        first = int(np.clip(seed_index, 0, n - 1))
    else:
        raise ValueError("seed must be 'leverage', 'norm', or 'index'")

    selected = [first]
    min_dist: list = [np.nan]

    d_to_S = 1.0 - (Yn @ Yn[first])
    d_to_S[first] = -np.inf

    for _ in range(1, k):
        nxt = int(np.argmax(d_to_S))
        selected.append(nxt)
        min_dist.append(float(d_to_S[nxt]))
        d_to_new = 1.0 - (Yn @ Yn[nxt])
        d_to_S = np.minimum(d_to_S, d_to_new)
        d_to_S[nxt] = -np.inf

    return selected, min_dist
