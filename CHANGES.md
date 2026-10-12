# Changes from 1.7 to 1.8

## Spectral Attention 1.3 → 2.0
- **Joint reading of eigenvectors.** Each component is embedded with its retained directions (trivial direction included). Default selection = spectral vertices by the Successive Projection Algorithm (at most m+1 per component); optional spectral medoids (k-means on the row-normalized embedding). Both are invariant under any orthonormal change of basis of the retained subspace, so the simple/repeated split is no longer needed. The v1.x rule remains as `selection="tails"`.
- **Bounded outputs.** Percentage-based selections are capped (`max_per_selection=5`); ties beyond the cap are split by row order, flagged and listed in full. Isolated chunks are reported in `isolated_chunks` and selected only with `select_isolates=True`.
- **Eigengap extension.** After group completion, the cutoff is extended while the relative eigengap is below `min_relative_gap` (default 0.05), at most `max_gap_extension` directions; final and per-component gaps are reported, with warnings.
- **New options:** `allocation="per_component"`, `degree_regularization` (Qin–Rohe, scale-invariant), `vertex_min_degree_quantile`.
- Near-zero eigenvalue warning threshold raised to 1e-8 (relative), so it fires for nearly disconnected components.
- Tests now exercise `spectral_attention` itself, including the report example and planted themes.

## Semantic Bridges 1.3 → 2.0
- **Spanning-tree bridges (default).** k−1 bridges forming a maximum spanning tree of single-linkage component affinities (Kruskal), with merge ranks. Alternatives: `top_m`, `all_pairs` (v1.x).
- Components smaller than `min_component_size` (default 2, i.e. isolates) are reported and not bridged.
- Optional `min_similarity` and `max_bridges` filters.
- Vectorized component affinities (O(n²) NumPy) replace the per-pair Python loop.

## Supported Diversity 1.0 → 2.0
- **Exact solver.** `solver="milp"` (default) solves the clique-strengthened integer problem with HiGHS through `scipy.optimize.milp` under `time_limit`; falls back to the best feasible candidate with the MILP dual bound. `solver="lp_greedy"` reproduces v1.0. Output adds `status`, `upper_bound`, `mip`, `candidate_values`.
- **Bug fix:** `--chunks` is read as UTF-8 (previously the platform default, which fails on Windows for non-ASCII text).
- Clique cover runs in a single pass over sorted edges (previously a repeated minimum over the uncovered set).
- Symmetry check is scale-relative, as in the other modules.

## New modules
- `graph_construction.py`: centring, all-but-the-top PC removal, quantile thresholds, mutual k-NN graphs, graph summaries.
- `selection_baselines.py`: MMR, facility location (lazy greedy), k-DPP greedy MAP, farthest-first.
- `scale_check.py`: reproduces report Section 10.

## Report 1.8
- New sections: graph construction (anisotropy, centring, k-NN), spectral background with eigengap extension and regularization, related work, behaviour at scale (synthetic output-size and stability check).
- Spectral Attention rewritten around the spectral embedding, with a basis-invariance proposition; Semantic Bridges around the single-linkage/MST proposition; Supported Diversity with the MILP solver and soft alternatives.
- Limits consolidated in one section instead of repeated disclaimers; internal notes and change-log sentences removed from the body.
- Testing plan removed from the report for now; the restructured draft (Stage 0 offline diagnostic, new arms) is kept in `testing_plan_draft_v1_8.tex`.
- References expanded from 4 to 28, including the smoothness of eigenvalue-cluster projectors (Oustry 1999) alongside Davis–Kahan.
