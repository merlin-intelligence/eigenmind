"""Graph construction helpers for Eigenmind.

Embedding cosines are typically anisotropic: most pairs share a large common
component, so raw cosines concentrate in a narrow band and a single global
threshold gives either one giant component or a dust of isolates. This module
provides the preprocessing and graph constructions:

  * normalize_embeddings: unit rows, optional mean-centring and removal of the
    top principal directions ("all-but-the-top", Mu & Viswanath 2018);
  * threshold_graph: strict cosine threshold;
  * adjacency_from_cosine: threshold_graph with checks on the cosine matrix;
  * quantile_threshold: threshold chosen as an off-diagonal quantile;
  * knn_graph: k-nearest-neighbour graph, mutual (default) or union;
  * graph_summary: components, isolates and size distribution, to be logged
    before any selection.

Requires NumPy and SciPy. The self-tests run under pytest.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def normalize_embeddings(V, *, center=False, remove_top=0, zero_rtol=1e-10):
    """Return unit-norm rows after optional centring and top-PC removal.

    Centring and top-PC removal can leave a row (numerically) zero: a single
    point, identical points, or too few points for remove_top. Such a row has
    no direction, so preprocessing is then skipped for the whole matrix (rows
    are only normalized) and a RuntimeWarning is emitted. A row counts as zero
    when its norm is at most zero_rtol times the largest input row norm.
    """
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or not np.isfinite(V).all():
        raise ValueError("V must be a finite 2-D array")
    if not isinstance(remove_top, (int, np.integer)) or remove_top < 0 or remove_top >= V.shape[1]:
        raise ValueError("remove_top must be an integer in [0, dim)")
    if len(V) == 0:
        return V.copy()
    raw_norms = np.linalg.norm(V, axis=1, keepdims=True)
    if np.any(raw_norms == 0):
        raise ValueError("A row of V has zero norm")
    if center or remove_top:
        X = V - V.mean(axis=0)
        if remove_top:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            P = Vt[:remove_top]
            X = X - (X @ P.T) @ P
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        if not np.any(norms <= zero_rtol * raw_norms.max()):
            return X / norms
        warnings.warn(
            "Centring/top-PC removal leaves a zero row (single point, identical points or "
            "too few points for remove_top); falling back to plain row normalization.",
            RuntimeWarning,
            stacklevel=2,
        )
    return V / raw_norms


def cosine_matrix(U):
    """Gram matrix of unit rows, symmetrized with an exact unit diagonal."""
    C = U @ U.T
    C = np.clip((C + C.T) / 2, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C


def threshold_graph(C, tau):
    """W_ij = C_ij if C_ij > tau (strict), zero diagonal."""
    C = np.asarray(C, dtype=float)
    if not 0 <= tau < 1:
        raise ValueError("tau must lie in [0, 1)")
    W = np.where(C > tau, C, 0.0)
    np.fill_diagonal(W, 0.0)
    return W


def adjacency_from_cosine(C, tau=0.5):
    """threshold_graph after checking that C is a finite symmetric cosine matrix."""
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or not np.isfinite(C).all():
        raise ValueError("C must be finite and square")
    if not np.allclose(C, C.T, atol=1e-12, rtol=1e-10):
        raise ValueError("C must be symmetric")
    if np.max(np.abs(C), initial=0) > 1 + 1e-10:
        raise ValueError("Cosine entries must lie in [-1, 1]")
    return threshold_graph((C + C.T) / 2, tau)


def _validate_adjacency(W):
    """Check a square, finite, nonnegative, zero-diagonal, symmetric W; return it symmetrized.

    Sparse input is densified. An empty (0 x 0) W is accepted.
    """
    if sparse.issparse(W):
        W = W.toarray()
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square")
    if not np.isfinite(W).all() or np.any(W < 0):
        raise ValueError("W must be finite and nonnegative")
    if np.any(np.diag(W) != 0):
        raise ValueError("W must have zero diagonal; remove self-loops explicitly")
    scale = np.max(W, initial=0.0)
    if scale and not np.allclose(W / scale, W.T / scale, atol=1e-12, rtol=1e-10):
        raise ValueError("Directed W is unsupported: choose symmetrization explicitly")
    return 0.5 * W + 0.5 * W.T


def quantile_threshold(C, q):
    """tau as the q-quantile of off-diagonal cosines (clipped to [0, 1))."""
    if not 0 < q < 1:
        raise ValueError("q must lie in (0, 1)")
    vals = np.asarray(C)[np.triu_indices(len(C), 1)]
    return float(np.clip(np.quantile(vals, q), 0.0, 1 - 1e-12))


def knn_graph(C, k, *, mutual=True, min_similarity=0.0):
    """k-NN graph on cosines; weights are the cosines (> min_similarity).

    Neighbours exclude the vertex itself; ties at the k-th neighbour are broken
    by index. mutual=True keeps an edge only if each endpoint is among the
    other's k nearest (reduces hubs); mutual=False keeps the union.
    """
    C = np.asarray(C, dtype=float)
    n = len(C)
    if not isinstance(k, (int, np.integer)) or not 1 <= k < n:
        raise ValueError("k must be an integer in [1, n)")
    S = C.copy()
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1, kind="stable")[:, :k]
    M = np.zeros((n, n), bool)
    M[np.repeat(np.arange(n), k), nn.ravel()] = True
    M = (M & M.T) if mutual else (M | M.T)
    W = np.where(M & (C > min_similarity), C, 0.0)
    np.fill_diagonal(W, 0.0)
    return W


def graph_summary(W):
    """Size statistics to record before selection."""
    W = np.asarray(W.toarray() if sparse.issparse(W) else W)
    n = len(W)
    count, labels = connected_components(sparse.csr_matrix(W), directed=False)
    sizes = np.bincount(labels)
    edges = int(np.count_nonzero(np.triu(W, 1)))
    return {
        "n": n,
        "edges": edges,
        "density": edges / max(1, n * (n - 1) / 2),
        "components": int(count),
        "isolates": int(np.sum(sizes == 1)),
        "largest_component": int(sizes.max()),
        "component_size_quantiles": np.quantile(sizes, [0.5, 0.9, 1.0]).tolist(),
    }


class Tests(unittest.TestCase):
    def test_centering_spreads_cosines(self):
        rng = np.random.default_rng(0)
        V = rng.normal(size=(200, 16)) + 4.0
        raw = cosine_matrix(normalize_embeddings(V))
        cen = cosine_matrix(normalize_embeddings(V, center=True))
        off = np.triu_indices(200, 1)
        self.assertGreater(raw[off].mean(), 0.8)
        self.assertLess(abs(cen[off].mean()), 0.05)

    def test_remove_top(self):
        rng = np.random.default_rng(1)
        V = rng.normal(size=(50, 8))
        V[:, 0] *= 20
        U = normalize_embeddings(V, remove_top=1)
        self.assertTrue(np.allclose(np.linalg.norm(U, axis=1), 1))

    def test_degenerate_preprocessing_falls_back(self):
        rng = np.random.default_rng(3)
        cases = [
            (rng.normal(size=(1, 8)), {"center": True}),  # single point
            (np.tile(rng.normal(size=8), (2, 1)), {"center": True}),  # identical points
            (rng.normal(size=(3, 8)), {"remove_top": 2}),  # too few points
        ]
        for V, kwargs in cases:
            with self.assertWarns(RuntimeWarning):
                U = normalize_embeddings(V, **kwargs)
            self.assertTrue(np.allclose(U, normalize_embeddings(V)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            normalize_embeddings(rng.normal(size=(5, 8)), center=True)

    def test_zero_input_row_rejected(self):
        with self.assertRaises(ValueError):
            normalize_embeddings(np.array([[1.0, 0.0], [0.0, 0.0]]))

    def test_knn_mutual_subset_of_union(self):
        rng = np.random.default_rng(2)
        U = normalize_embeddings(rng.normal(size=(40, 5)))
        C = cosine_matrix(U)
        Wm, Wu = knn_graph(C, 3), knn_graph(C, 3, mutual=False)
        self.assertTrue(np.all((Wu > 0).sum(1) >= 3))
        self.assertTrue(np.all((Wm > 0) <= (Wu > 0)))
        self.assertTrue(np.allclose(Wm, Wm.T))

    def test_threshold_and_summary(self):
        C = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
        s = graph_summary(threshold_graph(C, 0.5))
        self.assertEqual((s["components"], s["isolates"], s["edges"]), (2, 1, 1))
        self.assertAlmostEqual(quantile_threshold(C, 0.5), 0.2)
