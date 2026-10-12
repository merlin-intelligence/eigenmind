"""Semantic Bridges: closest chunk pairs between disconnected components.

Independent companion to Spectral Attention. W is a symmetric nonnegative
zero-diagonal adjacency matrix; C is the original aligned cosine matrix.
Requires Python >=3.10, NumPy and SciPy. Does not import spectral_attention.

Version 2.0 (package 1.8). The affinity of two components a, b is
beta_ab = max C_ij over i in a, j in b (single-linkage similarity). Instead of
returning all k(k-1)/2 component pairs, the default returns a maximum
spanning tree of the component-affinity graph (Kruskal): k-1 bridges, which
are exactly the merges single-linkage clustering would perform if the graph
threshold were lowered, in that order. Components below min_component_size
(default 2, i.e. isolates) are reported separately and not bridged.
Modes: "spanning_tree" (default), "top_m" (each component's m closest
partners) and "all_pairs" (v1.x behaviour, for ablation).

Run: python semantic_bridges.py --self-test
Run: python semantic_bridges.py --demo
Run: python semantic_bridges.py W.npy C.npy --chunks chunks.json --output bridges.json
A bridge is a potential semantic connection, not an existing edge in W.
"""
from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

VERSION = "2.0"
MODES = ("spanning_tree", "top_m", "all_pairs")


def adjacency_from_cosine(C, tau=0.5):
    """Strict threshold, zero diagonal; cosine input must be symmetric."""
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or not np.isfinite(C).all():
        raise ValueError("C must be finite and square")
    if not 0 <= tau < 1:
        raise ValueError("tau must lie in [0, 1)")
    if not np.allclose(C, C.T, atol=1e-12, rtol=1e-10):
        raise ValueError("C must be symmetric")
    if np.max(np.abs(C), initial=0) > 1 + 1e-10:
        raise ValueError("Cosine entries must lie in [-1, 1]")
    C = (C + C.T) / 2
    W = np.where(C > tau, C, 0.)
    np.fill_diagonal(W, 0.)
    return W


def _validate(W):
    if sparse.issparse(W):
        W = W.toarray()
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or len(W) == 0:
        raise ValueError("W must be nonempty and square")
    if not np.isfinite(W).all() or np.any(W < 0):
        raise ValueError("W must be finite and nonnegative")
    if np.any(np.diag(W) != 0):
        raise ValueError("W must have zero diagonal; remove self-loops explicitly")
    scale = np.max(W, initial=0.)
    if scale and not np.allclose(W / scale, W.T / scale, atol=1e-12, rtol=1e-10):
        raise ValueError("Directed W is unsupported: choose symmetrization explicitly")
    return .5 * W + .5 * W.T


def _validate_cosine(C, n):
    C = np.asarray(C, dtype=float)
    if C.shape != (n, n) or not np.isfinite(C).all():
        raise ValueError("cosine_matrix must be finite with the same shape as W")
    if not np.allclose(C, C.T, atol=1e-12, rtol=1e-10):
        raise ValueError("cosine_matrix must be symmetric")
    if np.max(np.abs(C), initial=0.) > 1 + 1e-10 or not np.allclose(np.diag(C), 1., atol=1e-10, rtol=0):
        raise ValueError("cosine_matrix requires unit diagonal and entries in [-1,1]")
    return .5 * C + .5 * C.T


def component_affinity(C, members):
    """beta[a, b] = max C over members[a] x members[b] (diagonal set to -inf).

    Vectorized: one row-max pass per component, then a segmented column
    reduction. Costs O(n^2) operations and O(k^2) memory for k components.
    """
    k = len(members)
    order = np.concatenate(members)
    starts = np.cumsum([0] + [len(m) for m in members[:-1]])
    Cs = C[:, order]
    B = np.empty((k, k))
    for a, ia in enumerate(members):
        B[a] = np.maximum.reduceat(Cs[ia].max(axis=0), starts)
    np.fill_diagonal(B, -np.inf)
    return B


def _kruskal(B):
    """Maximum spanning tree: sort pairs by (-beta, a, b), union-find."""
    k = len(B)
    a, b = np.triu_indices(k, 1)
    order = np.lexsort((b, a, -B[a, b]))
    parent = list(range(k))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    edges = []
    for t in order:
        ra, rb = find(int(a[t])), find(int(b[t]))
        if ra != rb:
            parent[rb] = ra
            edges.append((int(a[t]), int(b[t])))
            if len(edges) == k - 1:
                break
    return edges


def _top_m(B, m):
    pairs = set()
    for a in range(len(B)):
        others = np.array([b for b in range(len(B)) if b != a])
        if len(others):
            best = others[np.lexsort((others, -B[a, others]))][:m]
            pairs.update(tuple(sorted((a, int(b)))) for b in best)
    return sorted(pairs, key=lambda p: (-B[p], p))


def semantic_bridges(W, C, chunks=None, *, mode="spanning_tree", top_m=1,
                     min_component_size=2, max_bridges=None, min_similarity=None,
                     tie_atol=1e-12, tie_rtol=1e-8):
    """Bridge endpoints between connected components of W, using original C.

    Component pairs are chosen by `mode` among components with at least
    min_component_size chunks, optionally filtered by min_similarity and capped
    at max_bridges (strongest beta first). For every chosen pair all numerical
    cosine maximizers are returned (ties within tie_atol + tie_rtol|beta|).
    No eigenvectors are computed and W is never modified.
    """
    W = _validate(W)
    n = len(W)
    C = _validate_cosine(C, n)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not isinstance(min_component_size, (int, np.integer)) or min_component_size < 1:
        raise ValueError("min_component_size must be a positive integer")
    if not isinstance(top_m, (int, np.integer)) or top_m < 1:
        raise ValueError("top_m must be a positive integer")
    if max_bridges is not None and (not isinstance(max_bridges, (int, np.integer)) or max_bridges < 0):
        raise ValueError("max_bridges must be a nonnegative integer or None")
    if not np.isfinite([tie_atol, tie_rtol]).all() or min(tie_atol, tie_rtol) < 0:
        raise ValueError("Bridge tie tolerances must be finite and nonnegative")
    if chunks is not None and (len(chunks) != n or not all(isinstance(s, str) for s in chunks)):
        raise ValueError("chunks must contain one string per row")
    count, labels = connected_components(sparse.csr_matrix(W), directed=False)
    all_members = [np.flatnonzero(labels == c) for c in range(count)]
    eligible = [c for c in range(count) if len(all_members[c]) >= min_component_size]
    excluded = [c for c in range(count) if len(all_members[c]) < min_component_size]
    members = [all_members[c] for c in eligible]
    pairs, dropped = [], 0
    if len(members) >= 2:
        B = component_affinity(C, members)
        if mode == "spanning_tree":
            pairs = _kruskal(B)
        elif mode == "top_m":
            pairs = _top_m(B, top_m)
        else:
            a, b = np.triu_indices(len(B), 1)
            pairs = list(zip(a.tolist(), b.tolist()))
        merge_rank = {p: r for r, p in enumerate(pairs)} if mode == "spanning_tree" else {}
        if min_similarity is not None:
            kept = [p for p in pairs if B[p] >= min_similarity]
            dropped = len(pairs) - len(kept)
            pairs = kept
        ranked = sorted(pairs, key=lambda p: (-B[p], p))
        if max_bridges is not None:
            dropped += max(0, len(ranked) - max_bridges)
            ranked = ranked[:max_bridges]
        pairs = ranked
    records = []
    reasons = [[] for _ in range(n)]
    for a, b in pairs:
        ia, ib = members[a], members[b]
        block = C[np.ix_(ia, ib)]
        best = float(block.max())
        tol = tie_atol + tie_rtol * abs(best)
        found = [{"index_a": int(ia[i]), "index_b": int(ib[j]),
                  "cosine_similarity": float(block[i, j]),
                  "gap_to_maximum": float(best - block[i, j])}
                 for i, j in np.argwhere(best - block <= tol)]
        ca, cb = eligible[a], eligible[b]
        rec = {"id": len(records), "component_a": ca, "component_b": cb,
               "maximum_cosine_similarity": best, "n_maximizing_pairs": len(found), "pairs": found}
        if mode == "spanning_tree":
            rec["single_linkage_merge_rank"] = merge_rank[(a, b)]
        records.append(rec)
        for p in found:
            for side, other in (("a", "b"), ("b", "a")):
                reasons[p["index_" + side]].append({
                    "kind": "component_bridge_endpoint", "bridge_id": rec["id"],
                    "component": rec["component_" + side], "other_component": rec["component_" + other],
                    "partner_index": p["index_" + other], "cosine_similarity": p["cosine_similarity"],
                    "gap_to_maximum": p["gap_to_maximum"]})
    return {"module": "Semantic Bridges", "version": VERSION,
            "config": {"mode": mode, "top_m": top_m, "min_component_size": int(min_component_size),
                       "max_bridges": max_bridges, "min_similarity": min_similarity,
                       "tie_atol": tie_atol, "tie_rtol": tie_rtol},
            "n_components": int(count),
            "components": [{"id": c, "indices": all_members[c].tolist(),
                            "bridged": c in set(eligible)} for c in range(count)],
            "excluded_components": excluded,
            "n_pairs_dropped_by_filters": dropped,
            "component_bridges": records,
            "selected_chunks": [{"index": i, "component": int(labels[i]),
                "text": chunks[i] if chunks is not None else None, "reasons": why}
                for i, why in enumerate(reasons) if why]}


# ---------------------------------------------------------------- tests
def _unit(deg):
    a = np.deg2rad(deg)
    return np.column_stack((np.cos(a), np.sin(a)))


def _pairs(r):
    return [(p["index_a"], p["index_b"]) for g in r["component_bridges"] for p in g["pairs"]]


class BridgeTests(unittest.TestCase):
    def test_original_cosine_maximum(self):
        V = _unit([0., 10., 60., 70.]); C = V @ V.T
        W = adjacency_from_cosine(C, .9)
        r = semantic_bridges(W, C)
        self.assertEqual(r["n_components"], 2)
        self.assertEqual(_pairs(r), [(1, 2)])
        self.assertAlmostEqual(r["component_bridges"][0]["maximum_cosine_similarity"], np.cos(np.deg2rad(50.)))
        self.assertEqual(W[1, 2], 0.)

    def test_report_example_bridges(self):
        V = _unit([0, 10, 20, 65, 75, 85]); C = V @ V.T
        r = semantic_bridges(adjacency_from_cosine(C, .9), C)
        self.assertEqual(_pairs(r), [(2, 3)])
        self.assertAlmostEqual(r["component_bridges"][0]["maximum_cosine_similarity"], 0.707107, places=6)

    def test_spanning_tree_vs_all_pairs(self):
        V = _unit([0, 5, 40, 45, 80, 85]); C = V @ V.T
        W = adjacency_from_cosine(C, .99)
        tree = semantic_bridges(W, C)
        self.assertEqual(set(_pairs(tree)), {(1, 2), (3, 4)})
        self.assertEqual([g["single_linkage_merge_rank"] for g in tree["component_bridges"]], [0, 1])
        self.assertEqual(len(semantic_bridges(W, C, mode="all_pairs")["component_bridges"]), 3)
        self.assertEqual(len(semantic_bridges(W, C, mode="top_m", top_m=1)["component_bridges"]), 2)
        self.assertEqual(len(semantic_bridges(W, C, max_bridges=1)["component_bridges"]), 1)

    def test_isolates_excluded_and_ties(self):
        V = np.array([[1., 0, 0], [1., 0, 0], [0., 1, 0], [0., 1, 0], [0., 0, 1]])
        C = V @ V.T
        W = adjacency_from_cosine(C, .5)
        r = semantic_bridges(W, C)
        self.assertEqual(r["excluded_components"], [2])
        self.assertEqual([g["n_maximizing_pairs"] for g in r["component_bridges"]], [4])
        legacy = semantic_bridges(W, C, mode="all_pairs", min_component_size=1)
        self.assertEqual([g["n_maximizing_pairs"] for g in legacy["component_bridges"]], [4, 2, 2])

    def test_affinity_matches_brute_force(self):
        rng = np.random.default_rng(5)
        V = rng.normal(size=(30, 4)); V /= np.linalg.norm(V, axis=1, keepdims=True)
        C = V @ V.T
        members = [np.flatnonzero(rng.permutation(30) % 4 == c) for c in range(4)]
        B = component_affinity(C, members)
        for a in range(4):
            for b in range(4):
                if a != b:
                    self.assertAlmostEqual(B[a, b], C[np.ix_(members[a], members[b])].max())

    def test_negative_connected_and_validation(self):
        C = np.array([[1., -1.], [-1., 1.]])
        W = adjacency_from_cosine(C, .5)
        r = semantic_bridges(W, C, min_component_size=1)
        self.assertEqual(r["component_bridges"][0]["maximum_cosine_similarity"], -1.)
        self.assertEqual(semantic_bridges(W, C, min_component_size=1, min_similarity=0.)["component_bridges"], [])
        self.assertEqual(semantic_bridges(np.array([[0., 1.], [1., 0.]]), np.ones((2, 2)))["component_bridges"], [])
        with self.assertRaises(ValueError): semantic_bridges(W, np.eye(3))
        with self.assertRaises(ValueError): semantic_bridges(W, np.zeros((2, 2)))
        with self.assertRaises(ValueError): semantic_bridges(W, C, mode="other")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matrix", nargs="?")
    parser.add_argument("cosine", nargs="?")
    parser.add_argument("--chunks")
    parser.add_argument("--output")
    parser.add_argument("--mode", choices=list(MODES), default="spanning_tree")
    parser.add_argument("--top-m", type=int, default=1)
    parser.add_argument("--min-component-size", type=int, default=2)
    parser.add_argument("--max-bridges", type=int)
    parser.add_argument("--min-similarity", type=float)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(BridgeTests)
        raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
    if args.demo:
        V = _unit([0., 3., 6., 9., 60., 63., 66., 69., 110., 113., 150.])
        C = V @ V.T
        W = adjacency_from_cosine(C, .95)
        chunks = [f"Example chunk {i}" for i in range(len(W))]
    elif args.matrix and args.cosine:
        W = np.load(args.matrix, allow_pickle=False)
        C = np.load(args.cosine, allow_pickle=False)
        chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8")) if args.chunks else None
    else:
        parser.error("Supply W.npy C.npy, --demo, or --self-test")
    result = semantic_bridges(W, C, chunks, mode=args.mode, top_m=args.top_m,
                              min_component_size=args.min_component_size,
                              max_bridges=args.max_bridges, min_similarity=args.min_similarity)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
