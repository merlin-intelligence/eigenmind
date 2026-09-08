"""ℓ∞-connectivity optimizer + hinge ranking ("relevant but not obvious" chunks)."""
from __future__ import annotations

import heapq

import numpy as np


def _build_weighted_lengths(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert similarity weights W_ij in [0,1] into positive edge lengths via -log."""
    L = np.full_like(W, np.inf, dtype=float)
    mask = W > 0
    L[mask] = -np.log(np.clip(W[mask], eps, 1.0))
    np.fill_diagonal(L, 0.0)
    return L


def _dijkstra_all_sources(lengths: np.ndarray) -> np.ndarray:
    """All-pairs shortest paths via Dijkstra from each node (dense, OK for n<=~500)."""
    n = lengths.shape[0]
    dist = np.full((n, n), np.inf, dtype=float)

    for src in range(n):
        d = dist[src]
        d[src] = 0.0
        pq = [(0.0, src)]
        visited = np.zeros(n, dtype=bool)

        while pq:
            du, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True

            for v in range(n):
                w = lengths[u, v]
                if w == np.inf or v == u:
                    continue
                nd = du + w
                if nd < d[v]:
                    d[v] = nd
                    heapq.heappush(pq, (nd, v))

    return dist


def compute_linf_connectivity_optimizer(W: np.ndarray) -> np.ndarray:
    """Practical ℓ∞-connectivity optimizer x* in chunk space."""
    n = W.shape[0]
    degrees = np.sum(W, axis=1)
    if np.all(degrees == 0):
        return np.zeros(n, dtype=float)

    lengths = _build_weighted_lengths(W)
    dist = _dijkstra_all_sources(lengths)

    finite = np.isfinite(dist)
    if not np.all(finite):
        max_finite = np.max(dist[finite]) if np.any(finite) else 1.0
        dist = np.where(finite, dist, 10.0 * max_finite)

    transmission = np.sum(dist, axis=1)
    v_star = int(np.argmax(transmission))

    x_raw = dist[v_star, :].copy()

    deg_sum = float(np.sum(degrees))
    if deg_sum > 0:
        weighted_mean = float(np.sum(degrees * x_raw)) / deg_sum
        x_centered = x_raw - weighted_mean
    else:
        x_centered = x_raw - np.mean(x_raw)

    denom = float(np.max(np.abs(x_centered)))
    if denom < 1e-12:
        return np.zeros(n, dtype=float)

    return x_centered / denom


def rank_relevant_but_not_obvious_chunks(
    W: np.ndarray,
    x_star: np.ndarray,
    pole_quantile: float = 0.90,
):
    """Chunk-only retrieval of relevant-but-not-obvious items.

    Returns (ranking, P_plus, P_minus, q_hi, q_lo). Each ranking entry is
    ``(idx, hinge_score, bridge, S_plus, S_minus, x_i)``.
    """
    x = x_star
    q_hi = float(np.quantile(x, pole_quantile))
    q_lo = float(np.quantile(x, 1.0 - pole_quantile))

    Pp = np.where(x >= q_hi)[0]
    Pm = np.where(x <= q_lo)[0]

    if len(Pp) == 0 or len(Pm) == 0:
        q_hi = float(np.quantile(x, 0.80))
        q_lo = float(np.quantile(x, 0.20))
        Pp = np.where(x >= q_hi)[0]
        Pm = np.where(x <= q_lo)[0]

    S_plus = np.sum(W[:, Pp], axis=1) if len(Pp) > 0 else np.zeros(W.shape[0])
    S_minus = np.sum(W[:, Pm], axis=1) if len(Pm) > 0 else np.zeros(W.shape[0])

    bridge = np.minimum(S_plus, S_minus)
    hinge = bridge * (1.0 - np.abs(x))

    ranking = sorted(
        [
            (i, float(hinge[i]), float(bridge[i]), float(S_plus[i]), float(S_minus[i]), float(x[i]))
            for i in range(W.shape[0])
        ],
        key=lambda t: t[1],
        reverse=True,
    )
    return ranking, Pp, Pm, q_hi, q_lo
