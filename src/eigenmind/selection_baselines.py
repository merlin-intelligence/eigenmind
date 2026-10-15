"""Reference baselines to compare with the three selection mechanisms.

Each function returns a list of selected row indices, in selection order, at a
fixed chunk budget. They are comparison baselines, not part of the three
mechanisms.

  * mmr: Maximal Marginal Relevance (Carbonell & Goldstein 1998).
  * facility_location: greedy maximization of
        F(S) = sum_i w_i max_{j in S} max(C_ij, 0) + lam * sum_{j in S} r_j,
    a monotone submodular coverage objective (Lin & Bilmes 2011) with the
    (1 - 1/e) greedy guarantee; it rewards chunks that represent their
    neighbourhood, the intent behind "support".
  * kdpp_greedy: greedy MAP for a DPP with kernel L = diag(q) C diag(q)
    (Kulesza & Taskar 2012), fast incremental Cholesky (Chen et al. 2018):
    a soft, determinant-based redundancy penalty instead of hard exclusion.
  * farthest_first: k-centre greedy in cosine distance (geometric spread).

Requires NumPy. The self-tests run under pytest.
"""

from __future__ import annotations

import heapq
import unittest

import numpy as np


def _check(C, budget):
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or not np.isfinite(C).all():
        raise ValueError("C must be a finite square matrix")
    if not isinstance(budget, (int, np.integer)) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    return C, min(int(budget), len(C))


def mmr(relevance, C, budget, lam=0.5):
    """argmax lam*rel_i - (1-lam)*max_{j in S} C_ij, ties by index."""
    C, budget = _check(C, budget)
    rel = np.asarray(relevance, dtype=float)
    S, red = [], np.full(len(C), -np.inf)
    for _ in range(budget):
        score = lam * rel - (1 - lam) * np.where(np.isfinite(red), red, 0.0)
        score[S] = -np.inf
        i = int(np.argmax(score))
        S.append(i)
        red = np.maximum(red, C[i])
    return S


def facility_location(C, budget, *, weights=None, utilities=None, lam=0.0):
    """Lazy greedy for F(S) = sum_i w_i max_{j in S} C+_ij + lam * r(S)."""
    C, budget = _check(C, budget)
    n = len(C)
    S_pos = np.clip(C, 0.0, None)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    r = np.zeros(n) if utilities is None else np.asarray(utilities, dtype=float)
    cover = np.zeros(n)

    def gain(j):
        return float(w @ np.maximum(S_pos[:, j] - cover, 0.0) + lam * r[j])

    heap = [(-gain(j), j) for j in range(n)]
    heapq.heapify(heap)
    S = []
    while heap and len(S) < budget:
        g, j = heapq.heappop(heap)
        fresh = gain(j)
        if not heap or fresh >= -heap[0][0] - 1e-15:
            S.append(j)
            cover = np.maximum(cover, S_pos[:, j])
        else:
            heapq.heappush(heap, (-fresh, j))
    return S


def kdpp_greedy(C, budget, *, quality=None, jitter=1e-9):
    """Greedy MAP of det(L_S), L = diag(q) C diag(q); stops if gains vanish."""
    C, budget = _check(C, budget)
    n = len(C)
    q = np.ones(n) if quality is None else np.asarray(quality, dtype=float)
    L = q[:, None] * C * q[None, :] + jitter * np.eye(n)
    d2 = np.diag(L).copy()
    V = np.zeros((budget, n))
    S = []
    for t in range(budget):
        d2m = d2.copy()
        d2m[S] = -np.inf
        j = int(np.argmax(d2m))
        if d2m[j] <= jitter * 10:
            break
        e = (L[j] - V[:t, j] @ V[:t]) / np.sqrt(d2[j])
        V[t] = e
        d2 = d2 - e**2
        S.append(j)
    return S


def farthest_first(C, budget, *, start=None):
    """k-centre greedy on cosine distance 1 - C; start defaults to row 0."""
    C, budget = _check(C, budget)
    if budget == 0:
        return []
    S = [0 if start is None else int(start)]
    dist = 1 - C[S[0]]
    while len(S) < budget:
        dm = dist.copy()
        dm[S] = -np.inf
        j = int(np.argmax(dm))
        S.append(j)
        dist = np.minimum(dist, 1 - C[j])
    return S


class Tests(unittest.TestCase):
    def setUp(self):
        a = np.deg2rad([0, 2, 4, 90, 92, 180])
        V = np.column_stack((np.cos(a), np.sin(a)))
        self.C = V @ V.T

    def test_mmr_diversifies(self):
        rel = np.array([1.0, 0.99, 0.98, 0.5, 0.5, 0.1])
        self.assertNotIn(mmr(rel, self.C, 2, lam=0.3)[1], (1, 2))
        self.assertEqual(mmr(rel, self.C, 2, lam=1.0)[1], 1)

    def test_facility_location_representatives(self):
        S = facility_location(self.C, 2)
        self.assertEqual({0 if i < 3 else 1 for i in S}, {0, 1})

    def test_kdpp_avoids_near_duplicates(self):
        S = kdpp_greedy(self.C, 3)
        self.assertEqual(len({i // 3 for i in S}), 2)
        self.assertNotIn(1, S[:2])

    def test_farthest_first(self):
        self.assertEqual(farthest_first(self.C, 2), [0, 5])

    def test_budget_bounds(self):
        for f in (lambda: facility_location(self.C, 10), lambda: farthest_first(self.C, 10)):
            self.assertEqual(len(f()), 6)
        with self.assertRaises(ValueError):
            facility_location(self.C, -1)
