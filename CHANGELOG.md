# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/): while the version is 0.x, a minor release may
contain breaking changes.

## [Unreleased]

## [0.2.0] — 2026-10-15

This release replaces the 0.1 selection strategies with three new mechanisms. It contains
breaking changes; see [Upgrading from 0.1](README.md#upgrading-from-01).

### Added

- **Spectral Attention** (`spectral_attention`): component representatives, spectral vertices
  (Successive Projection Algorithm) and optional spectral medoids on the spectral embedding of
  each connected component. `selection="tails"` selects the two tails of each retained
  eigenvector instead.
- **Semantic Bridges** (`semantic_bridges`): maximum spanning tree of component affinities in
  the cosine matrix, with `top_m` and `all_pairs` modes. Isolated chunks are not bridged by
  default.
- **Supported Diversity** (`supported_diversity`): budgeted stable set among chunks with
  positive degree, solved exactly by MILP (HiGHS) under a time limit, with an LP upper bound and
  greedy fallbacks (`solver="lp_greedy"` skips the MILP).
- Graph construction helpers: `normalize_embeddings` (optional mean-centring and removal of top
  principal directions), `cosine_matrix`, `threshold_graph`, `adjacency_from_cosine`,
  `quantile_threshold`, `knn_graph`, `graph_summary`.
- Gaussian-kernel diverse coverage: `build_gaussian_kernel`, `diverse_coverage`.
- Evaluation baselines: `mmr`, `facility_location`, `kdpp_greedy`, `farthest_first`.
- `SimilarityGraph` methods `summary`, `spectral_attention`, `semantic_bridges`,
  `supported_diversity`, `diverse_coverage`, `kernel`, `component_spectra` and the four
  baselines; constructor options `center` and `remove_top`.
- Command-line tools `eigenmind-spectral`, `eigenmind-bridges`, `eigenmind-diversity` and
  `eigenmind-scale-check`.
- `eigenmind.__version__`; results of the three mechanisms report the package version.

### Changed

- **Breaking:** `SimilarityGraph.selection_tags` returns the tags `SpectralAttention`,
  `SemanticBridge` and `SupportedDiversity` (plus `DiverseCoverage` with
  `include_coverage=True`) instead of `Singular`, `Hinge` and `Theta`. It also accepts
  `utilities`.
- **Breaking:** `SimilarityGraph` keeps an edge only when the cosine is strictly above the
  threshold (`C_ij > threshold`); 0.1 also kept `C_ij == threshold`. The default threshold is
  still 0.65.
- **Breaking:** the `SimilarityGraph` threshold must lie in [0, 1); 0.1 accepted any value.
- **Breaking:** `SimilarityGraph` raises a `ValueError` when two points share an id; 0.1 accepted
  them silently.
- `SimilarityGraph` normalizes input vectors itself instead of rejecting unnormalized ones.
- `SimilarityGraph` results are cached per full set of arguments and returned as copies, so
  modifying a result does not affect later calls.
- Chunk text defaults to `payload["text"]` when every point has one; payloads that are not
  dicts are ignored.

### Removed

- **Breaking:** the 0.1 API: `build_similarity_matrix`, `normalized_laplacian`,
  `find_singular_chunks`, `analyze_laplacian_eigenvectors`,
  `compute_linf_connectivity_optimizer`, `rank_relevant_but_not_obvious_chunks`,
  `build_conflict_mask`, `theta_subgradient_approximation`, `theta_diversity_picker`, and the
  `SimilarityGraph` methods `singular_chunks`, `eigenvalue_analysis`, `linf_optimizer`,
  `hinge_ranking`, `conflict_mask`, `theta_factor` and `theta_diversity`. Pin
  `eigenmind==0.1.1` to keep them.
- The `viz` extra (matplotlib), used only by `analyze_laplacian_eigenvectors`.

### Fixed

- Centring or top-PC removal no longer fails on a single point, identical points, or too few
  points for `remove_top`: rows are then only normalized, with a `RuntimeWarning`.

### Packaging

- Requires `scipy>=1.9`; build requires `setuptools>=77`.
- License declared as an SPDX expression (`MIT`).
- Python 3.13 added to the CI matrix and classifiers; development status set to Beta.
- `pytest` also runs the self-tests embedded in each module.

## [0.1.1] — 2026-09-08

### Changed

- Version bump.

## [0.1.0] — 2026-09-08

### Added

- First release on PyPI: `SimilarityGraph` with singular chunks, hinge ranking and θ diversity.
- CI and publishing workflows.

### Fixed

- Similarity threshold validation.
