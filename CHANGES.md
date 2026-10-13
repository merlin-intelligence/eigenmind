# Changelog

## 1.8.0

Version numbers now follow the Eigenmind report (1.8); the previous release was 0.1.1.

### Added
- **Spectral Attention** (`spectral_attention`): component representatives, spectral vertices (SPA) and spectral medoids on the spectral embedding of each component; legacy per-eigenvector tails as `selection="tails"`. Complete eigenvalue groups, eigengap extension, optional per-component allocation and degree regularization, capped selections with full tie reporting; isolates reported separately.
- **Semantic Bridges** (`semantic_bridges`): maximum spanning tree of single-linkage component affinities in the original cosine matrix, with `top_m` and `all_pairs` modes; isolates excluded by default.
- **Supported Diversity** (`supported_diversity`): budgeted stable set among supported chunks, solved exactly by MILP (HiGHS) with an LP bound and greedy comparators; `solver="lp_greedy"` heuristic.
- `graph_construction`: `normalize_embeddings` (centring, top-PC removal), `cosine_matrix`, `threshold_graph`, `quantile_threshold`, `knn_graph`, `graph_summary`.
- `coverage`: Gaussian-kernel facility-location coverage (`build_gaussian_kernel`, `diverse_coverage`).
- `selection_baselines`: `mmr`, `facility_location`, `kdpp_greedy`, `farthest_first`.
- `SimilarityGraph`: `summary`, `spectral_attention`, `semantic_bridges`, `supported_diversity`, `diverse_coverage`, `kernel`, baselines, `component_spectra`; `center` and `remove_top` options.
- CLI entry points `eigenmind-spectral`, `eigenmind-bridges`, `eigenmind-diversity`, `eigenmind-scale-check`.
- `eigenmind.__version__`.

### Changed
- `SimilarityGraph.selection_tags` now returns the tags `SpectralAttention`, `SemanticBridge`, `SupportedDiversity` (and `DiverseCoverage`), and accepts `utilities`. The pre-1.8 Singular/Hinge/Theta tags are available from the deprecated `legacy_selection_tags`. **Code that matches on tag names must be updated.**
- `SimilarityGraph` normalizes input vectors itself (pre-1.8 it raised on unnormalized vectors) and builds `W` with a strict threshold (`C_ij > threshold`; pre-1.8 kept `C_ij = threshold`). The default threshold is unchanged at 0.65.
- Results of `SimilarityGraph` methods are memoized per full set of arguments; chunk text defaults to `payload["text"]` when present.

### Deprecated (removal in 2.0)
- `find_singular_chunks`, `analyze_laplacian_eigenvectors`, `build_similarity_matrix`, `normalized_laplacian`, `compute_linf_connectivity_optimizer`, `rank_relevant_but_not_obvious_chunks`, `build_conflict_mask`, `theta_subgradient_approximation`, `theta_diversity_picker`, and the `SimilarityGraph` methods `singular_chunks`, `eigenvalue_analysis`, `linf_optimizer`, `hinge_ranking`, `conflict_mask`, `theta_factor`, `theta_diversity`, `legacy_selection_tags`. They keep their pre-1.8 behaviour and emit a `DeprecationWarning`. See the README for replacements.

### Fixed
- Supported Diversity CLI reads `--chunks` as UTF-8 on every platform.

### Packaging
- SPDX license expression (`license = "MIT"`), setuptools ≥ 77; Python 3.13 added to the CI matrix and classifiers.
- `pytest` now also runs the self-tests embedded in each module.

## 0.1.1
- Version bump.

## 0.1.0
- PyPI packaging, CI workflows, threshold-validation fix.
