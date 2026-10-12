"""Reproduce the synthetic scale and stability check of report Section 10.

Usage:
    python scale_check.py                       # v1.8 figures only
    python scale_check.py --v17 PATH_TO_V1_7    # also v1.7 columns

Synthetic data: 1,500 unit vectors in R^64 from 30 Gaussian themes with a
shared common direction; threshold graphs at off-diagonal cosine quantiles.
A diagnostic of mechanics, not a semantic evaluation.
"""
import argparse
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v17", help="folder containing the v1.7 scripts (optional)")
    a = p.parse_args()
    warnings.simplefilter("ignore")
    sa = load(HERE / "spectral_attention.py", "sa")
    sb = load(HERE / "semantic_bridges.py", "sb")
    sd = load(HERE / "supported_diversity.py", "sd")
    gc = load(HERE / "graph_construction.py", "gc")
    old = None
    if a.v17:
        old = (load(Path(a.v17) / "spectral_attention.py", "sa0"), load(Path(a.v17) / "semantic_bridges.py", "sb0"))

    rng = np.random.default_rng(0)
    n, k, dim = 1500, 30, 64
    centers = rng.normal(size=(k, dim))
    lab = rng.integers(0, k, n)
    X0 = centers[lab] + 1.2 * rng.normal(size=(n, dim)) + 3 * np.ones(dim) / np.sqrt(dim) * rng.normal(1, .3, (n, 1))

    def graph(X, q=.99):
        C = gc.cosine_matrix(gc.normalize_embeddings(X))
        return C, gc.threshold_graph(C, gc.quantile_threshold(C, q))

    def chosen(r, kinds=None):
        return {c["index"] for c in r["selected_chunks"]
                if kinds is None or any(x["kind"] in kinds for x in c["reasons"])}

    print("== Output size by edge-density quantile")
    for q in (.99, .995, .999):
        C, W = graph(X0, q)
        s = gc.graph_summary(W)
        line = f"q={q}: components={s['components']} isolates={s['isolates']}"
        if old:
            r0, b0 = old[0].spectral_attention(W), old[1].semantic_bridges(W, C)
            line += f" | v1.7 SA={len(chosen(r0))} SB pairs={len(b0['component_bridges'])} SB chunks={len(chosen(b0))}"
        r1, b1 = sa.spectral_attention(W), sb.semantic_bridges(W, C)
        t = time.time(); d = sd.supported_diversity(W, np.ones(n), budget=24); t = time.time() - t
        line += (f" | v1.8 SA={len(chosen(r1))} SB pairs={len(b1['component_bridges'])} SB chunks={len(chosen(b1))}"
                 f" | SD {d['status']} value={d['selected_utility']} ({t:.2f}s)")
        print(line)

    print("== Theme coverage and stability (q=0.99, 3% noise, 5 seeds)")
    C, W = graph(X0)
    variants = {
        "tails capped 1/tail": (lambda W: sa.spectral_attention(W, selection="tails", tail_fraction=1e-4,
                                                                max_per_selection=1), ("lower_tail", "upper_tail")),
        "spectral vertices": (lambda W: sa.spectral_attention(W), ("spectral_vertex",)),
        "vertices, degree q>=.25": (lambda W: sa.spectral_attention(W, vertex_min_degree_quantile=.25), ("spectral_vertex",)),
        "spectral medoids": (lambda W: sa.spectral_attention(W, selection="medoids"), ("spectral_medoid",)),
    }
    if old:
        variants = {"v1.7 tails": (lambda W: old[0].spectral_attention(W), ("lower_tail", "upper_tail")), **variants}
    for name, (fn, kinds) in variants.items():
        base = chosen(fn(W), kinds)
        jc, jt = [], []
        for seed in range(5):
            noise = np.random.default_rng(seed + 10).normal(size=X0.shape)
            Xp = X0 + .03 * np.linalg.norm(X0, axis=1, keepdims=True) / np.sqrt(dim) * noise
            got = chosen(fn(graph(Xp)[1]), kinds)
            jc.append(len(got & base) / len(got | base))
            tb, tg = set(lab[list(base)]), set(lab[list(got)])
            jt.append(len(tb & tg) / len(tb | tg))
        print(f"{name:26s} chunks={len(base):4d} themes={len(set(lab[list(base)])):2d} "
              f"chunk_stability={np.mean(jc):.2f} theme_stability={np.mean(jt):.2f}")


if __name__ == "__main__":
    main()
