#!/usr/bin/env python3
"""Supported Diversity: budgeted, supported, pairwise nonadjacent chunks.

Dependencies: numpy, scipy (>=1.9 for milp). W must be symmetric, nonnegative,
zero-diagonal. Utilities are REQUIRED; the caller defines what is useful.
Original positive weighted degree defines support, not semantic quality.

Version 2.0 (package 1.8). Default solver="milp": the budgeted maximum-weight
stable-set problem is solved exactly with HiGHS branch-and-bound
(scipy.optimize.milp) under a time limit, using the same clique-strengthened
constraints as the relaxation. If the time limit is reached, the best of the
MILP incumbent, LP-guided greedy and utility-only greedy is returned with its
proven gap. solver="lp_greedy" reproduces the v1.x heuristic.

Example:
    result = supported_diversity(W, utilities, budget=12, chunks=texts)
    python supported_diversity.py W.npy utilities.npy --budget 12 --output out.json
    python supported_diversity.py --demo
    python supported_diversity.py --self-test
"""

import argparse
import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

VERSION = "2.0"


def _validate(W, utilities, budget, min_degree, chunks, relaxation, solver, time_limit):
    W = np.asarray(W.toarray() if sparse.issparse(W) else W, dtype=float)
    if W.ndim != 2 or W.shape[0] == 0 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a nonempty square matrix")
    if not np.isfinite(W).all() or (W < 0).any() or np.any(np.diag(W) != 0):
        raise ValueError("W must be finite, nonnegative and zero-diagonal")
    scale = np.max(W, initial=0.0)
    if scale and not np.allclose(W / scale, W.T / scale, atol=1e-12, rtol=1e-10):
        raise ValueError("W must be symmetric")
    W = W / 2 + W.T / 2
    r = np.asarray(utilities, dtype=float)
    if r.shape != (len(W),) or not np.isfinite(r).all() or (r < 0).any():
        raise ValueError("utilities must be a finite nonnegative vector aligned with W")
    if (
        isinstance(budget, (bool, np.bool_))
        or not isinstance(budget, (int, np.integer))
        or budget < 0
    ):
        raise ValueError("budget must be a nonnegative integer")
    if not np.isfinite(min_degree) or min_degree < 0:
        raise ValueError("min_degree must be finite and nonnegative")
    if chunks is not None and (
        len(chunks) != len(W) or not all(isinstance(s, str) for s in chunks)
    ):
        raise ValueError("chunks must contain one string per row")
    if relaxation not in ("cliques", "edges"):
        raise ValueError("relaxation must be 'cliques' or 'edges'")
    if solver not in ("milp", "lp_greedy"):
        raise ValueError("solver must be 'milp' or 'lp_greedy'")
    if not (isinstance(time_limit, (int, float)) and time_limit > 0):
        raise ValueError("time_limit must be a positive number of seconds")
    with np.errstate(over="ignore"):
        degree = W.sum(axis=1)
        total_utility = r.sum()
    if not np.isfinite(degree).all() or not np.isfinite(total_utility):
        raise ValueError("weight or utility sums overflow; rescale inputs")
    return W, r, degree


def _clique_cover(A, relaxation):
    """Deterministic cover of every edge; not enumeration of all cliques.

    Edges are visited in lexicographic order; each uncovered edge is grown into
    a clique through common neighbours (highest degree first, then index).
    O(|E| + sum of clique-growth costs) instead of a repeated min over a set.
    """
    rows, cols = np.nonzero(np.triu(A, 1))
    neighbors = [set(np.flatnonzero(row).tolist()) for row in A]
    covered = set()
    cliques = []
    for a, b in zip(rows.tolist(), cols.tolist(), strict=True):
        if (a, b) in covered:
            continue
        clique = [a, b]
        if relaxation == "cliques":
            candidates = neighbors[a] & neighbors[b]
            while candidates:
                v = min(candidates, key=lambda i: (-len(neighbors[i]), i))
                clique.append(v)
                candidates &= neighbors[v]
        clique.sort()
        cliques.append(clique)
        covered.update(itertools.combinations(clique, 2))
    return cliques


def _greedy(A, r, scores, budget):
    order = sorted(range(len(r)), key=lambda i: (-scores[i], -r[i], i))
    selected = []
    for i in order:
        if len(selected) >= budget:
            break
        if r[i] > 0 and not any(A[i, j] for j in selected):
            selected.append(i)
    return selected


def supported_diversity(
    W,
    utilities,
    *,
    budget,
    chunks=None,
    min_degree=0.0,
    relaxation="cliques",
    solver="milp",
    time_limit=10.0,
):
    """Return a feasible stable set among originally supported vertices.

    Eligibility: original weighted degree > 0 and >= min_degree. Any positive
    edge is a hard conflict. Integer problem:
        max r.z  s.t. z binary, sum z <= budget, sum_{i in Q} z_i <= 1 (Q in cover)
    The LP relaxation (0 <= x <= 1) is always solved and reported as an upper
    bound. solver="milp" then solves the integer problem with HiGHS; if it is
    not proven optimal within time_limit seconds the best feasible candidate is
    returned with the MILP dual bound. Nonunique optima are resolved by the
    solver (milp) or by utility then row index (greedy). Zero-utility nodes are
    never returned.
    """
    W, r, degree = _validate(
        W, utilities, budget, min_degree, chunks, relaxation, solver, time_limit
    )
    ids = np.flatnonzero((degree > 0) & (degree >= min_degree))
    A = W[np.ix_(ids, ids)] > 0
    rr = r[ids]
    m = len(ids)
    active = bool(m and budget and np.any(rr > 0))
    cliques = _clique_cover(A, relaxation) if active else []
    x = np.zeros(m)
    lp_bound = 0.0
    status = "trivial_empty_optimum"
    candidates = {}
    mip = {"status": None, "dual_bound": None, "node_count": None}
    if active:
        rows, cols = [], []
        for j, clique in enumerate(cliques):
            rows.extend([j] * len(clique))
            cols.extend(clique)
        rows.extend([len(cliques)] * m)
        cols.extend(range(m))
        cons = sparse.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(len(cliques) + 1, m)
        ).tocsr()
        rhs = np.r_[np.ones(len(cliques)), min(budget, m)]
        scale = float(rr.max())
        lp = linprog(-rr / scale, A_ub=cons, b_ub=rhs, bounds=(0.0, 1.0), method="highs")
        if not lp.success:
            raise RuntimeError(f"LP failed (status {lp.status}): {lp.message}")
        if not np.isfinite(lp.x).all() or np.max(cons @ lp.x - rhs) > 1e-6:
            raise RuntimeError("LP returned an invalid or infeasible numerical solution")
        x = np.clip(lp.x, 0, 1)
        lp_bound = float(-lp.fun * scale)
        status = "lp_optimal_with_numerical_tolerances"
        candidates["lp_guided_greedy"] = _greedy(A, rr, rr * x, budget)
        if solver == "milp":
            res = milp(
                -rr / scale,
                constraints=LinearConstraint(cons, -np.inf, rhs),
                integrality=np.ones(m),
                bounds=Bounds(0, 1),
                options={"time_limit": float(time_limit), "disp": False},
            )
            mip["status"] = int(res.status)
            mip["node_count"] = getattr(res, "mip_node_count", None)
            dual = getattr(res, "mip_dual_bound", None)
            mip["dual_bound"] = (
                None if dual is None or not np.isfinite(dual) else float(-dual * scale)
            )
            if res.x is not None:
                z = np.flatnonzero(res.x > 0.5)
                z = [int(i) for i in z if rr[i] > 0]
                if len(z) <= budget and not np.any(A[np.ix_(z, z)]):
                    candidates["milp"] = z
            if res.status == 0 and "milp" in candidates:
                status = "milp_proven_optimal"
            else:
                status = (
                    "milp_time_limit_best_feasible"
                    if res.status == 1
                    else f"milp_status_{res.status}"
                )
    baseline = _greedy(A, rr, rr, budget)
    candidates["utility_greedy"] = baseline
    values = {k: float(rr[v].sum()) for k, v in candidates.items()}
    preference = ["milp", "lp_guided_greedy", "utility_greedy"]
    strategy = max(values, key=lambda k: (values[k], -preference.index(k)))
    chosen, value = candidates[strategy], values[strategy]
    if len(chosen) > budget or np.any(A[np.ix_(chosen, chosen)]):
        raise RuntimeError("Discrete feasibility check failed")
    upper = lp_bound
    if mip["dual_bound"] is not None:
        upper = min(upper, mip["dual_bound"])
    if status == "milp_proven_optimal":
        upper = value
    if value > lp_bound + 1e-6 * max(1.0, value, abs(lp_bound)):
        raise RuntimeError("Discrete value exceeds numerical LP bound; rescale utilities")
    selected_indices = sorted(int(ids[i]) for i in chosen)
    lp_full = np.zeros(len(W))
    lp_full[ids] = x
    selected_chunks = []
    for i in selected_indices:
        record = {
            "index": i,
            "utility": float(r[i]),
            "weighted_degree": float(degree[i]),
            "lp_coordinate": float(lp_full[i]),
            "reasons": [
                "positive original graph support",
                "utility-based selection with no positive edge to another selected chunk",
            ],
        }
        if chunks is not None:
            record["chunk"] = chunks[i]
        selected_chunks.append(record)
    excluded = [
        {"index": i, "reason": "isolated" if degree[i] == 0 else "below_min_degree"}
        for i in range(len(W))
        if degree[i] == 0 or degree[i] < min_degree
    ]
    gap = max(0.0, upper - value)
    return {
        "module": "Supported Diversity",
        "version": VERSION,
        "config": {
            "budget": int(budget),
            "min_degree": float(min_degree),
            "relaxation": relaxation,
            "solver": solver,
            "time_limit": float(time_limit),
        },
        "eligible_indices": ids.tolist(),
        "excluded_nodes": excluded,
        "selected_indices": selected_indices,
        "selected_chunks": selected_chunks,
        "status": status,
        "strategy": strategy,
        "selected_utility": value,
        "upper_bound": upper,
        "lp_upper_bound": lp_bound,
        "mip": mip,
        "absolute_gap": gap,
        "relative_gap": gap / upper if upper > 0 else 0.0,
        "candidate_values": values,
        "utility_greedy_value": values["utility_greedy"],
        "lp_coordinates": lp_full.tolist(),
        "fractional_coordinates": int(np.sum((x > 1e-7) & (x < 1 - 1e-7))),
        "clique_constraints": [[int(ids[i]) for i in c] for c in cliques],
        "bound_note": "Floating-point solver bounds; not exact-arithmetic certificates.",
    }


def _read_chunks(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _demo():
    # Three redundant chunks, two other themes, and one unsupported outlier.
    W = np.zeros((6, 6))
    for i, j in [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]:
        W[i, j] = W[j, i] = 0.8
    return supported_diversity(
        W, np.array([1, 1, 1, 1, 1, 10]), budget=3, chunks=[f"Chunk {i}" for i in range(6)]
    )


class Tests(unittest.TestCase):
    def test_report_example_diversity(self):
        a = np.deg2rad(40)
        V = np.array([[np.cos(-a), np.sin(-a), 0], [1, 0, 0], [np.cos(a), np.sin(a), 0], [0, 0, 1]])
        C = V @ V.T
        W = np.where(C > 0.7, C, 0.0)
        np.fill_diagonal(W, 0)
        out = supported_diversity(W, [1, 1.5, 1, 10], budget=2)
        self.assertEqual(out["selected_indices"], [0, 2])
        self.assertEqual(out["status"], "milp_proven_optimal")
        self.assertAlmostEqual(out["utility_greedy_value"], 1.5)

    def test_isolated_noise(self):
        W = np.zeros((8, 8))
        W[:3, :3] = 1
        np.fill_diagonal(W, 0)
        out = supported_diversity(W, [1] * 3 + [100] * 5, budget=8)
        self.assertEqual(len(out["selected_indices"]), 1)
        self.assertEqual(len(out["excluded_nodes"]), 5)

    def test_triangle_strengthening(self):
        W = np.ones((3, 3)) - np.eye(3)
        for kind, bound in [("cliques", 1), ("edges", 1.5)]:
            out = supported_diversity(W, np.ones(3), budget=3, relaxation=kind)
            self.assertAlmostEqual(out["lp_upper_bound"], bound)
            self.assertEqual(out["selected_utility"], 1)

    def test_fractional_odd_cycle(self):
        W = np.zeros((5, 5))
        for i in range(5):
            W[i, (i + 1) % 5] = W[(i + 1) % 5, i] = 1
        for solver, upper in (("lp_greedy", 2.5), ("milp", 2.0)):
            out = supported_diversity(W, np.ones(5), budget=5, solver=solver)
            self.assertAlmostEqual(out["lp_upper_bound"], 2.5)
            self.assertAlmostEqual(out["upper_bound"], upper)
            self.assertEqual(out["selected_utility"], 2)
            self.assertEqual(out["fractional_coordinates"], 5)

    def test_original_degree(self):
        W = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
        out = supported_diversity(W, [1, 1, 1], budget=3, min_degree=2)
        self.assertEqual(out["selected_indices"], [0])

    def test_empty_cases(self):
        for W, r, b in [
            (np.zeros((2, 2)), [1, 1], 2),
            (np.ones((2, 2)) - np.eye(2), [1, 1], 0),
            (np.ones((2, 2)) - np.eye(2), [0, 0], 2),
        ]:
            out = supported_diversity(W, r, budget=b)
            self.assertEqual(out["selected_indices"], [])
            self.assertEqual(out["upper_bound"], 0)

    def test_invalid_inputs(self):
        W = np.ones((2, 2)) - np.eye(2)
        for r in ([-1, 1], [np.nan, 1], [1]):
            with self.assertRaises(ValueError):
                supported_diversity(W, r, budget=1)
        for b in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                supported_diversity(W, [1, 1], budget=b)
        with self.assertRaises(ValueError):
            supported_diversity([[0, 1], [0, 0]], [1, 1], budget=1)
        with self.assertRaises(ValueError):
            supported_diversity(W, [1, 1], budget=1, solver="x")

    def test_exact_small_graph_comparison(self):
        rng = np.random.default_rng(42)
        for n in range(2, 11):
            for _ in range(5):
                W = np.triu(rng.random((n, n)) < 0.35, 1).astype(float)
                W += W.T
                r = rng.uniform(0.1, 2, n)
                budget = max(1, n // 2)
                eligible = np.flatnonzero(W.sum(axis=1) > 0)
                optimum = max(
                    (
                        float(r[list(s)].sum())
                        for k in range(min(budget, len(eligible)) + 1)
                        for s in itertools.combinations(eligible, k)
                        if not np.any(W[np.ix_(s, s)])
                    ),
                    default=0,
                )
                exact = supported_diversity(W, r, budget=budget)
                heur = supported_diversity(W, r, budget=budget, solver="lp_greedy")
                self.assertAlmostEqual(exact["selected_utility"], optimum, places=7)
                for out in (exact, heur):
                    sel = out["selected_indices"]
                    self.assertLessEqual(len(sel), budget)
                    self.assertFalse(np.any(W[np.ix_(sel, sel)]))
                    self.assertLessEqual(out["selected_utility"], optimum + 1e-7)
                    self.assertGreaterEqual(out["upper_bound"] + 1e-7, optimum)
                    self.assertGreaterEqual(out["selected_utility"], out["utility_greedy_value"])
                covered = {
                    tuple(sorted(e))
                    for c in exact["clique_constraints"]
                    for e in itertools.combinations(c, 2)
                }
                self.assertEqual(covered, set(zip(*np.where(np.triu(W > 0, 1)), strict=True)))

    def test_sparse_and_repeatability(self):
        W = np.ones((4, 4)) - np.eye(4)
        self.assertEqual(
            supported_diversity(W, [1] * 4, budget=2),
            supported_diversity(sparse.csr_matrix(W), [1] * 4, budget=2),
        )

    def test_utf8_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "chunks.json")
            Path(p).write_text(
                json.dumps(["Économie", "réassurance"], ensure_ascii=False), encoding="utf-8"
            )
            self.assertEqual(_read_chunks(p), ["Économie", "réassurance"])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("adjacency", nargs="?")
    parser.add_argument("utilities", nargs="?")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--min-degree", type=float, default=0.0)
    parser.add_argument("--relaxation", choices=["cliques", "edges"], default="cliques")
    parser.add_argument("--solver", choices=["milp", "lp_greedy"], default="milp")
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--chunks")
    parser.add_argument("--output")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
        raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
    if args.demo:
        result = _demo()
    else:
        if args.adjacency is None or args.utilities is None or args.budget is None:
            parser.error("provide W.npy utilities.npy and --budget, or use --demo/--self-test")
        chunks = _read_chunks(args.chunks) if args.chunks else None
        result = supported_diversity(
            np.load(args.adjacency, allow_pickle=False),
            np.load(args.utilities, allow_pickle=False),
            budget=args.budget,
            chunks=chunks,
            min_degree=args.min_degree,
            relaxation=args.relaxation,
            solver=args.solver,
            time_limit=args.time_limit,
        )
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
