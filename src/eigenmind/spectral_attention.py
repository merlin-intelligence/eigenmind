"""Spectral Attention: thematic representatives and spectral vertices/medoids.

An Eigenmind module for two contexts: structural selection within a query's
candidate pool, or prompt-independent attention points from a corpus graph.
"Attention" means directing a reader's attention, not Transformer attention.
The graph is supplied by the caller; this module does not certify relevance.

Version 2.0 (package 1.8). Default selection reads the retained eigenvectors
jointly, as a spectral embedding of each component, and returns
  * spectral vertices: extreme points found by the Successive Projection
    Algorithm (SPA), at most m+1 per component for m retained directions;
  * optionally spectral medoids: typical points of a k-means partition of the
    row-normalized embedding (Ng-Jordan-Weiss).
Both rules depend only on the geometry of the embedded point cloud and are
therefore invariant under any orthonormal change of basis of the retained
subspace. The v1.x rule (two tails per simple eigenvector, row norms for a
repeated eigenspace) is kept as selection="tails" for ablation.

Requires Python >=3.10, NumPy and SciPy. Dense reference implementation.
Run: python spectral_attention.py --self-test
Run: python spectral_attention.py --demo
Run: python spectral_attention.py W.npy --chunks chunks.json --output result.json
W.npy stores an adjacency matrix, not a cosine Gram matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.cluster.vq import kmeans2
from scipy.linalg import block_diag
from scipy.sparse.csgraph import connected_components

VERSION = "2.0"
SELECTIONS = ("vertices", "medoids", "tails")


# ---------------------------------------------------------------- inputs
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
    W = np.where(C > tau, C, 0.0)
    np.fill_diagonal(W, 0.0)
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
    scale = np.max(W, initial=0.0)
    if scale and not np.allclose(W / scale, W.T / scale, atol=1e-12, rtol=1e-10):
        raise ValueError("Directed W is unsupported: choose symmetrization explicitly")
    return 0.5 * W + 0.5 * W.T


# ---------------------------------------------------------------- spectra
def component_spectrum(A, laplacian="normalized", degree_regularization=0.0):
    """Eigen-decomposition of one connected component.

    Returns (lam, U, deg) with lam ascending, U Euclidean-orthonormal, and deg
    the (possibly regularized) degrees used for degree adjustment. For the
    normalized Laplacian the block is first divided by its maximum weight,
    which leaves the spectrum unchanged. Regularization (Qin-Rohe) replaces
    D by D + t I in the normalization, with t = degree_regularization times
    the mean degree, so it is invariant to weight scale. Column 0 is the
    trivial direction (exactly the kernel when t = 0).
    """
    A = np.asarray(A, dtype=float)
    if laplacian == "normalized":
        A = A / A.max()
        deg = A.sum(1)
        deg = deg + degree_regularization * deg.mean()
        inv = 1 / np.sqrt(deg)
        L = np.eye(len(A)) - inv[:, None] * A * inv[None, :]
    else:
        deg = A.sum(1)
        L = np.diag(deg) - A
    lam, U = np.linalg.eigh(L)
    return lam, U, deg


def _groups(values, atol, rtol):
    """Sorted values; compare with the group anchor to avoid tolerance chaining."""
    groups = []
    for i, val in enumerate(values):
        if not groups or abs(val - values[groups[-1][0]]) > atol + rtol * max(
            abs(val), abs(values[groups[-1][0]])
        ):
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


def _retain(values, count, atol, rtol, min_gap, max_extension):
    """Choose leading groups of sorted values.

    1. Never cut a numerically repeated group (tolerance grouping).
    2. Then, while the relative gap to the next value is below min_gap,
       extend by whole groups, at most max_extension extra directions.
    Returns (groups, diagnostics).
    """
    groups = _groups(values, atol, rtol)
    kept, n_kept = [], 0
    for g in groups:
        if n_kept >= count:
            break
        kept.append(g)
        n_kept += len(g)
    base = n_kept

    def rel_gap(k):
        if k >= len(groups):
            return None
        last, nxt = values[groups[k - 1][-1]], values[groups[k][0]]
        return float((nxt - last) / nxt) if nxt > 0 else 0.0

    while kept and min_gap > 0 and len(kept) < len(groups):
        gap = rel_gap(len(kept))
        nxt = len(groups[len(kept)])
        if gap is None or gap >= min_gap or n_kept + nxt - base > max_extension:
            break
        kept.append(groups[len(kept)])
        n_kept += nxt
    gap = rel_gap(len(kept)) if kept else None
    return kept, {
        "requested": int(count),
        "retained": int(n_kept),
        "gap_extension": int(n_kept - base),
        "relative_gap_after_cutoff": gap,
        "gap_below_threshold": bool(gap is not None and min_gap > 0 and gap < min_gap),
    }


# ---------------------------------------------------------------- selectors
def spa(Y, max_points=None, *, candidates=None, rtol=1e-8, tie_rtol=1e-9):
    """Successive Projection Algorithm on the rows of Y.

    Repeatedly pick the candidate row with the largest residual norm and
    project all rows onto the orthogonal complement of that row. Depends only
    on inner products of rows, hence invariant under Y -> Y Q for orthogonal Q.
    Stops after max_points (default: number of columns) or when the residual
    norm falls below rtol times the first pick's norm. Ties (relative tie_rtol)
    are broken by lowest row index and counted.
    Returns list of (row, relative_residual_norm, tie_count).
    """
    R = np.array(Y, dtype=float, copy=True)
    n, p = R.shape
    max_points = p if max_points is None else min(max_points, p)
    cand = np.ones(n, bool) if candidates is None else np.asarray(candidates, bool).copy()
    picks, first = [], None
    for _ in range(max_points):
        norms = np.einsum("ij,ij->i", R, R)
        norms[~cand] = -1.0
        best = norms.max()
        if best <= 0:
            break
        first = best if first is None else first
        if best <= rtol**2 * first:
            break
        tied = np.flatnonzero(norms >= best * (1 - tie_rtol))
        i = int(tied[0])
        picks.append((i, float(math.sqrt(best / first)), int(len(tied))))
        u = R[i] / math.sqrt(best)
        R -= np.outer(R @ u, u)
        cand[i] = False
    return picks


def medoids(Z, k, *, seed=0, restarts=10):
    """k-means on rows of Z (row-normalized), best of `restarts` seeded runs;
    return one medoid (member nearest its centroid) per nonempty cluster."""
    n = len(Z)
    if n <= k:
        return [(i, 0.0, 1) for i in range(n)]
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = Z / np.maximum(norms, 1e-300)
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(restarts):
            cent, lab = kmeans2(Z, k, minit="++", seed=seed + t)
            inertia = float(((Z - cent[lab]) ** 2).sum())
            if best is None or inertia < best[0] - 1e-12:
                best = (inertia, cent, lab)
    _, cent, lab = best
    out = []
    for c in range(k):
        members = np.flatnonzero(lab == c)
        if len(members):
            dist = ((Z[members] - cent[c]) ** 2).sum(1)
            out.append((int(members[np.argmin(dist)]), float(dist.min()), int(len(members))))
    return sorted(out)


def _extreme(scores, fraction, high, ties, atol, rtol, cap):
    """Top/bottom selection with tie handling and an optional hard cap."""
    scores = np.asarray(scores)
    target = max(1, math.ceil(fraction * len(scores)))
    if cap is not None:
        target = min(target, cap)
    order = np.argsort(-scores if high else scores, kind="stable")
    cut = float(scores[order[target - 1]])
    close = np.abs(scores - cut) <= atol + rtol * np.maximum(np.abs(scores), abs(cut))
    beyond = scores > cut if high else scores < cut
    full = np.flatnonzero(beyond | close)
    if ties == "include" and (cap is None or len(full) <= cap):
        chosen = full
    else:
        chosen = np.sort(order[:target])
    return chosen, {
        "target_count": int(target),
        "selected_count": int(len(chosen)),
        "cutoff": cut,
        "cutoff_tie_count": int(close.sum()),
        "ties_split": bool(len(full) > len(chosen)),
        "cutoff_tied_local": np.flatnonzero(close).tolist() if len(full) > len(chosen) else [],
    }


# ---------------------------------------------------------------- main
def spectral_attention(
    W,
    chunks=None,
    *,
    positive_count=10,
    allocation="global",
    selection=("vertices",),
    laplacian="normalized",
    coordinates="degree_adjusted",
    degree_regularization=0.0,
    min_relative_gap=0.05,
    max_gap_extension=None,
    representative_fraction=0.05,
    max_per_selection=5,
    vertex_min_degree_quantile=0.0,
    medoid_seed=0,
    medoid_restarts=10,
    tail_fraction=0.05,
    eigenspace_fraction=None,
    select_isolates=False,
    eigen_atol=1e-10,
    eigen_rtol=1e-8,
    ties="include",
    score_atol=1e-12,
    score_rtol=1e-8,
    small_eigen_warning=1e-8,
):
    """Return JSON-serializable selections with all selection reasons.

    Retained directions: the first `positive_count` positive eigen-directions,
    ordered globally across components (allocation="global") or within each
    component (allocation="per_component"); complete repeated groups are never
    cut, and the cutoff is extended while the relative eigengap is below
    min_relative_gap (at most max_gap_extension extra directions, default
    positive_count). Each component with m >= 1 retained local directions is
    embedded in R^(m+1) (trivial direction included) and selections are made
    on that point cloud. Representatives are the highest weighted degrees.
    Every percentage-based selection is capped at max_per_selection
    (None: no cap); ties beyond the cap are split by row order and reported.
    Isolated chunks are reported separately; selected only if select_isolates.
    """
    W = _validate(W)
    n = len(W)
    sel = (selection,) if isinstance(selection, str) else tuple(selection)
    if not sel or any(s not in SELECTIONS for s in sel):
        raise ValueError(f"selection must be drawn from {SELECTIONS}")
    if not isinstance(positive_count, (int, np.integer)) or positive_count < 0:
        raise ValueError("positive_count must be a nonnegative integer")
    if allocation not in ("global", "per_component"):
        raise ValueError("allocation must be 'global' or 'per_component'")
    if not 0 < tail_fraction <= 0.5 or not 0 < representative_fraction <= 1:
        raise ValueError("Invalid selection fractions")
    eigenspace_fraction = (
        min(1.0, 2 * tail_fraction) if eigenspace_fraction is None else eigenspace_fraction
    )
    if not 0 < eigenspace_fraction <= 1:
        raise ValueError("eigenspace_fraction must be in (0,1]")
    if laplacian not in ("normalized", "unnormalized") or coordinates not in (
        "degree_adjusted",
        "raw",
    ):
        raise ValueError("Invalid Laplacian or coordinate choice")
    if ties not in ("include", "stable") or min(eigen_atol, eigen_rtol, score_atol, score_rtol) < 0:
        raise ValueError("Invalid tie policy or tolerances")
    if max_per_selection is not None and (
        not isinstance(max_per_selection, (int, np.integer)) or max_per_selection < 1
    ):
        raise ValueError("max_per_selection must be a positive integer or None")
    if not 0 <= degree_regularization or not math.isfinite(degree_regularization):
        raise ValueError("degree_regularization must be finite and nonnegative")
    if laplacian == "unnormalized" and degree_regularization:
        raise ValueError("degree_regularization applies to the normalized Laplacian only")
    if not 0 <= min_relative_gap < 1 or not 0 <= vertex_min_degree_quantile < 1:
        raise ValueError("min_relative_gap and vertex_min_degree_quantile must lie in [0,1)")
    if chunks is not None and (len(chunks) != n or not all(isinstance(s, str) for s in chunks)):
        raise ValueError("chunks must contain one string per row")
    max_gap_extension = positive_count if max_gap_extension is None else max_gap_extension
    coords = coordinates if laplacian == "normalized" else "raw"
    d = W.sum(axis=1)
    if not np.isfinite(d).all():
        raise ValueError("Degree overflow; rescale W")
    count, labels = connected_components(sparse.csr_matrix(W), directed=False)
    result = {
        "module": "Spectral Attention",
        "version": VERSION,
        "config": {
            "positive_count": int(positive_count),
            "allocation": allocation,
            "selection": list(sel),
            "laplacian": laplacian,
            "coordinates": coords,
            "degree_regularization": degree_regularization,
            "min_relative_gap": min_relative_gap,
            "max_gap_extension": int(max_gap_extension),
            "representative_fraction": representative_fraction,
            "max_per_selection": max_per_selection,
            "vertex_min_degree_quantile": vertex_min_degree_quantile,
            "medoid_seed": medoid_seed,
            "medoid_restarts": medoid_restarts,
            "tail_fraction": tail_fraction,
            "eigenspace_fraction": eigenspace_fraction,
            "select_isolates": select_isolates,
            "ties": ties,
            "eigen_atol": eigen_atol,
            "eigen_rtol": eigen_rtol,
            "score_atol": score_atol,
            "score_rtol": score_rtol,
        },
        "n_components": int(count),
        "kernel_dimension": int(count),
        "components": [],
        "isolated_chunks": [],
        "cutoff": {},
        "spectral_groups": [],
        "selections": [],
        "warnings": [],
    }
    reasons = [[] for _ in range(n)]

    def warn(msg):
        result["warnings"].append(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

    def add(kind, component, indices, scores, group=None, extra=None):
        sid = len(result["selections"])
        rec = {
            "id": sid,
            "kind": kind,
            "component": int(component),
            "spectral_group": group,
            "indices": [int(i) for i in indices],
        }
        rec.update(extra or {})
        result["selections"].append(rec)
        for rank, (i, s) in enumerate(zip(indices, scores, strict=True)):
            reasons[int(i)].append(
                {
                    "selection_id": sid,
                    "kind": kind,
                    "component": int(component),
                    "spectral_group": group,
                    "rank": rank,
                    "score": float(s),
                }
            )

    def extreme(idx, scores, fraction, high, kind, component, group=None):
        chosen, diag = _extreme(
            scores, fraction, high, ties, score_atol, score_rtol, max_per_selection
        )
        diag["cutoff_tied_indices"] = idx[diag.pop("cutoff_tied_local")].tolist()
        add(kind, component, idx[chosen], scores[chosen], group, diag)

    # --- per-component spectra
    comp = {}
    directions = []
    for c in range(count):
        idx = np.flatnonzero(labels == c)
        result["components"].append({"id": c, "indices": idx.tolist(), "isolated": len(idx) == 1})
        if len(idx) == 1:
            result["isolated_chunks"].append(int(idx[0]))
            if select_isolates:
                add("isolated_chunk", c, idx, [0.0])
            continue
        extreme(
            idx, d[idx] / d[idx].max(), representative_fraction, True, "component_representative", c
        )
        lam, U, deg = component_spectrum(W[np.ix_(idx, idx)], laplacian, degree_regularization)
        scale = max(1.0, float(np.max(np.abs(lam))))
        if np.any(lam[1:] <= small_eigen_warning * scale):
            warn(
                f"Component {c}: very small positive eigenvalues; the component is nearly "
                "disconnected and its leading eigenvectors may be ill-conditioned."
            )
        comp[c] = {"idx": idx, "lam": lam, "U": U, "deg": deg, "trivial_eigenvalue": float(lam[0])}
        for j in range(1, len(idx)):
            directions.append((max(0.0, float(lam[j])), c, j))

    # --- retained directions (global or per component)
    retained = {c: [] for c in comp}

    def retain(dirs, key):
        dirs = sorted(dirs)
        vals = [x[0] for x in dirs]
        groups, diag = (
            _retain(
                vals, positive_count, eigen_atol, eigen_rtol, min_relative_gap, max_gap_extension
            )
            if positive_count
            else (
                [],
                {
                    "requested": 0,
                    "retained": 0,
                    "gap_extension": 0,
                    "relative_gap_after_cutoff": None,
                    "gap_below_threshold": False,
                },
            )
        )
        result["cutoff"][key] = diag
        if diag["gap_below_threshold"]:
            warn(
                f"Cutoff {key}: relative eigengap {diag['relative_gap_after_cutoff']:.3g} below "
                f"{min_relative_gap}; the retained subspace may be unstable."
            )
        for g in groups:
            items = [dirs[i] for i in g]
            gid = len(result["spectral_groups"])
            local = {}
            for _, cc, j in items:
                local.setdefault(cc, []).append(j)
                retained[cc].append((j, gid))
            result["spectral_groups"].append(
                {
                    "id": gid,
                    "dimension": len(items),
                    "eigenvalue_min": items[0][0],
                    "eigenvalue_max": items[-1][0],
                    "local_dimensions": {str(k): len(v) for k, v in local.items()},
                }
            )

    if allocation == "global":
        retain(directions, "global")
    else:
        for c in comp:
            retain([x for x in directions if x[1] == c], f"component_{c}")
    result["positive_eigenvalues"] = sorted(x[0] for x in directions)
    result["available_positive_count"] = len(directions)
    result["retained_positive_count"] = sum(len(v) for v in retained.values())

    # --- selections on each component's embedding
    for c, info in comp.items():
        idx, lam, U, deg = info["idx"], info["lam"], info["U"], info["deg"]
        js = sorted(j for j, _ in retained[c])
        entry = result["components"][c]
        entry["retained_local_dimension"] = len(js)
        entry["trivial_eigenvalue"] = info["trivial_eigenvalue"]
        if not js:
            continue
        nxt = len(js) + 1
        entry["local_relative_gap"] = (
            float((lam[nxt] - lam[js[-1]]) / lam[nxt]) if nxt < len(lam) and lam[nxt] > 0 else None
        )
        cols = [0] + js
        Y = U[:, cols] / np.sqrt(deg)[:, None] if coords == "degree_adjusted" else U[:, cols].copy()
        Y /= np.linalg.norm(Y)  # one common scalar: rankings and invariance unchanged
        if "vertices" in sel:
            cand = None
            if vertex_min_degree_quantile > 0:
                cand = d[idx] >= np.quantile(d[idx], vertex_min_degree_quantile)
            picks = spa(Y, candidates=cand)
            add(
                "spectral_vertex",
                c,
                idx[[p[0] for p in picks]],
                [p[1] for p in picks],
                None,
                {
                    "embedding_dimension": len(cols),
                    "tie_counts": [p[2] for p in picks],
                    "score_meaning": "SPA residual norm relative to the first vertex",
                },
            )
        if "medoids" in sel:
            meds = medoids(U[:, cols], len(cols), seed=medoid_seed, restarts=medoid_restarts)
            add(
                "spectral_medoid",
                c,
                idx[[m[0] for m in meds]],
                [m[1] for m in meds],
                None,
                {
                    "n_clusters": len(cols),
                    "cluster_sizes": [m[2] for m in meds],
                    "score_meaning": "squared distance to centroid (row-normalized embedding)",
                },
            )
        if "tails" in sel:  # v1.x rule, kept for ablation
            by_group = {}
            for j, gid in retained[c]:
                by_group.setdefault(gid, []).append(j)
            for gid, gjs in sorted(by_group.items()):
                F = (
                    U[:, gjs] / np.sqrt(deg)[:, None]
                    if coords == "degree_adjusted"
                    else U[:, gjs].copy()
                )
                F /= np.linalg.norm(F)
                if len(gjs) == 1:
                    f = F[:, 0]
                    if f[np.argmax(np.abs(f))] < 0:
                        f = -f
                    extreme(idx, f, tail_fraction, False, "lower_tail", c, gid)
                    extreme(idx, f, tail_fraction, True, "upper_tail", c, gid)
                else:
                    extreme(
                        idx,
                        np.linalg.norm(F, axis=1),
                        eigenspace_fraction,
                        True,
                        "eigenspace_expression",
                        c,
                        gid,
                    )
    result["selected_chunks"] = [
        {
            "index": i,
            "component": int(labels[i]),
            "weighted_degree": float(d[i]),
            "text": chunks[i] if chunks is not None else None,
            "reasons": why,
        }
        for i, why in enumerate(reasons)
        if why
    ]
    return result


select_chunks = spectral_attention  # compatibility alias


# ---------------------------------------------------------------- tests
def _unit(deg):
    a = np.deg2rad(deg)
    return np.column_stack((np.cos(a), np.sin(a)))


def _planted(blocks=4, size=6, link=0.2):
    """Cliques of `size` joined in a ring by one weak edge each."""
    n = blocks * size
    W = np.zeros((n, n))
    for b in range(blocks):
        s = slice(b * size, (b + 1) * size)
        W[s, s] = 1.0
        i, j = b * size, ((b + 1) % blocks) * size + 1
        W[i, j] = W[j, i] = link
    np.fill_diagonal(W, 0.0)
    return W


def _kinds(r, kind):
    return [i for s in r["selections"] if s["kind"] == kind for i in s["indices"]]


class SelectionTests(unittest.TestCase):
    def test_report_example_spectral(self):
        W = adjacency_from_cosine(
            _unit([-35, -25, -15, 15, 25, 35]) @ _unit([-35, -25, -15, 15, 25, 35]).T, 0.85
        )
        for mode in ("vertices", "tails"):
            r = spectral_attention(W, positive_count=1, selection=mode, min_relative_gap=0)
            self.assertAlmostEqual(r["positive_eigenvalues"][0], 0.189986, places=5)
            self.assertEqual(sorted(c["index"] for c in r["selected_chunks"]), [0, 2, 3, 5])

    def test_triangle_group_completion(self):
        W = np.ones((3, 3)) - np.eye(3)
        r = spectral_attention(W, positive_count=1)
        self.assertEqual(r["retained_positive_count"], 2)
        self.assertTrue(np.allclose(r["positive_eigenvalues"], [1.5, 1.5]))
        self.assertEqual(sorted(_kinds(r, "spectral_vertex")), [0, 1, 2])

    def test_isolates_reported_not_selected(self):
        W = block_diag(np.ones((3, 3)) - np.eye(3), [[0, 2], [2, 0]], [[0]])
        r = spectral_attention(W, positive_count=10)
        self.assertEqual(r["kernel_dimension"], 3)
        self.assertEqual(r["isolated_chunks"], [5])
        self.assertNotIn(5, [c["index"] for c in r["selected_chunks"]])
        r = spectral_attention(W, positive_count=10, select_isolates=True)
        self.assertIn(5, [c["index"] for c in r["selected_chunks"]])

    def test_spa_basis_invariance(self):
        rng = np.random.default_rng(3)
        Y = rng.normal(size=(40, 5))
        Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        self.assertEqual([p[0] for p in spa(Y)], [p[0] for p in spa(Y @ Q)])

    def test_planted_themes_vertices_and_medoids(self):
        W = _planted()
        r = spectral_attention(W, positive_count=3, selection=("vertices", "medoids"))
        for kind in ("spectral_vertex", "spectral_medoid"):
            picks = _kinds(r, kind)
            self.assertEqual(len(picks), 4)
            self.assertEqual(len({i // 6 for i in picks}), 4, kind)

    def test_rescaling_and_permutation(self):
        rng = np.random.default_rng(7)
        A = rng.uniform(0.1, 2, (12, 12))
        W0 = (A + A.T) / 2
        np.fill_diagonal(W0, 0)
        B = _planted() * rng.uniform(0.5, 2, (24, 24))
        W1 = (B + B.T) / 2
        # medoids rely on k-means: test them where clusters exist (W1).
        for mode, W in (("vertices", W0), ("tails", W0), ("vertices", W1), ("medoids", W1)):

            def selected(X, mode=mode):
                return {
                    c["index"]
                    for c in spectral_attention(X, positive_count=3, selection=mode)[
                        "selected_chunks"
                    ]
                }

            p = rng.permutation(len(W))
            self.assertEqual(selected(W), selected(100 * W), mode)
            self.assertEqual(selected(W), {int(p[i]) for i in selected(W[np.ix_(p, p)])}, mode)

    def test_gap_extension(self):
        P = np.diag(np.ones(4), 1)
        P = P + P.T
        P2 = P.copy()
        P2[0, 1] = P2[1, 0] = 1.001
        W = block_diag(P, P2)
        self.assertEqual(
            spectral_attention(W, positive_count=1, min_relative_gap=0)["retained_positive_count"],
            1,
        )
        r = spectral_attention(W, positive_count=1, min_relative_gap=0.05)
        self.assertEqual(r["retained_positive_count"], 2)
        self.assertEqual(r["cutoff"]["global"]["gap_extension"], 1)

    def test_cap_and_reported_ties(self):
        W = np.ones((40, 40)) - np.eye(40)
        r = spectral_attention(W, positive_count=0, representative_fraction=0.5)
        rep = r["selections"][0]
        self.assertEqual(rep["selected_count"], 5)
        self.assertTrue(rep["ties_split"])
        self.assertEqual(len(rep["cutoff_tied_indices"]), 40)
        r = spectral_attention(
            W, positive_count=0, representative_fraction=0.5, max_per_selection=None
        )
        self.assertEqual(r["selections"][0]["selected_count"], 40)

    def test_generalized_eigenvectors_and_regularization(self):
        W = np.array([[0.0, 1, 0], [1, 0, 4], [0, 4, 0]])
        lam, U, deg = component_spectrum(W)
        F = U / np.sqrt(deg)[:, None]
        L = np.diag(deg) - W / W.max()
        self.assertTrue(np.allclose(L @ F, deg[:, None] * F * lam))
        self.assertAlmostEqual(lam[0], 0.0, places=12)
        lam_r, _, _ = component_spectrum(W, degree_regularization=0.5)
        self.assertGreater(lam_r[0], 1e-6)
        r = spectral_attention(_planted(), positive_count=3, degree_regularization=0.5)
        self.assertEqual(len({i // 6 for i in _kinds(r, "spectral_vertex")}), 4)

    def test_per_component_allocation(self):
        W = block_diag(_planted(3, 5), np.ones((4, 4)) - np.eye(4))
        r = spectral_attention(W, positive_count=2, allocation="per_component", min_relative_gap=0)
        self.assertEqual(
            {c["id"]: c["retained_local_dimension"] for c in r["components"]}, {0: 2, 1: 3}
        )

    def test_validation(self):
        for W in ([[0, 1], [0, 0]], [[0, -1], [-1, 0]], [[1]]):
            with self.assertRaises(ValueError):
                spectral_attention(W)
        with self.assertRaises(ValueError):
            spectral_attention(np.zeros((2, 2)), selection="other")
        with self.assertRaises(ValueError):
            spectral_attention(
                np.zeros((2, 2)), laplacian="unnormalized", degree_regularization=1.0
            )

    def test_zero_request_and_cosine(self):
        C = np.array([[1.0, 0.8, 0.2], [0.8, 1, 0.5], [0.2, 0.5, 1]])
        W = adjacency_from_cosine(C, 0.5)
        self.assertEqual(np.count_nonzero(W), 2)
        self.assertEqual(spectral_attention(W, positive_count=0)["retained_positive_count"], 0)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("matrix", nargs="?")
    parser.add_argument("--chunks")
    parser.add_argument("--output")
    parser.add_argument(
        "--positive-count", type=int, default=None, help="default 10 (3 for --demo)"
    )
    parser.add_argument("--allocation", choices=["global", "per_component"], default="global")
    parser.add_argument("--selection", nargs="+", choices=list(SELECTIONS), default=["vertices"])
    parser.add_argument("--laplacian", choices=["normalized", "unnormalized"], default="normalized")
    parser.add_argument(
        "--coordinates", choices=["degree_adjusted", "raw"], default="degree_adjusted"
    )
    parser.add_argument("--degree-regularization", type=float, default=0.0)
    parser.add_argument("--min-relative-gap", type=float, default=0.05)
    parser.add_argument("--max-per-selection", type=int, default=5)
    parser.add_argument("--select-isolates", action="store_true")
    parser.add_argument("--ties", choices=["include", "stable"], default="include")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelectionTests)
        raise SystemExit(
            0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
        )
    if args.demo:
        W = _planted()
        W = block_diag(W, [[0]])
        chunks = [f"Example chunk {i}" for i in range(len(W))]
    elif args.matrix:
        W = np.load(args.matrix, allow_pickle=False)
        chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8")) if args.chunks else None
    else:
        parser.error("Supply W.npy, --demo, or --self-test")
    count = args.positive_count if args.positive_count is not None else (3 if args.demo else 10)
    result = spectral_attention(
        W,
        chunks,
        positive_count=count,
        allocation=args.allocation,
        selection=args.selection,
        laplacian=args.laplacian,
        coordinates=args.coordinates,
        degree_regularization=args.degree_regularization,
        min_relative_gap=args.min_relative_gap,
        max_per_selection=args.max_per_selection if args.max_per_selection > 0 else None,
        select_isolates=args.select_isolates,
        ties=args.ties,
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
