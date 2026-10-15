# eigenmind

Eigenmind is an eigenvalue-based graph engine for semantic selection: given embedded chunks, it proposes structurally motivated *attention points* — chunks worth reading, with the reason each was selected.

Eigenmind provides three selection mechanisms:

1. **Spectral Attention** — embeds each connected component with its leading Laplacian eigenvectors and selects component representatives (highest weighted degree), **spectral vertices** (extreme points, by the Successive Projection Algorithm) and optionally **spectral medoids** (typical points, $k$-means on the row-normalized embedding). Both rules are invariant under any change of basis of the retained eigenspace; the cutoff never splits a repeated eigenvalue and is extended while the eigengap is small. `selection="tails"` instead selects the two tails of each retained eigenvector.
2. **Semantic Bridges** — links connected components through a maximum spanning tree of their single-linkage affinities in the original cosine matrix ($k-1$ bridges for $k$ components, each the merge that lowering the graph threshold would produce first). `top_m` and `all_pairs` modes are available for comparison.
3. **Supported Diversity** — selects, within a budget, chunks that have positive weighted degree in the graph (supported) and share no edge with each other (a stable set), maximizing an explicit utility. Solved exactly by mixed-integer programming (HiGHS via `scipy.optimize.milp`, clique-strengthened formulation) under a time limit, with an LP bound and greedy comparators.

Also included:

- **Graph construction** (`graph_construction.py`) — mean-centring, removal of top principal directions ("all-but-the-top"), quantile thresholds, mutual $k$-NN graphs and graph summaries.
- **Diverse coverage** (`coverage.py`) — facility-location coverage over a Gaussian kernel with relevance, $(1-1/e)$ greedy guarantee and an online upper bound.
- **Evaluation baselines** (`selection_baselines.py`) — MMR, facility location on clipped cosines, $k$-DPP greedy MAP and farthest-first, to compare against the three mechanisms.
- **Scale check** (`scale_check.py`) — synthetic diagnostic of output size and stability.

Graph structure does not certify relevance or truth; usefulness must be evaluated on real collections.

---

## Install

```bash
pip install eigenmind
```

Optional extra: `eigenmind[dev]` (pytest, ruff). To track an unreleased change:

```bash
pip install git+https://github.com/merlin-intelligence/eigenmind.git
```

Requirements: Python ≥ 3.10, NumPy, SciPy ≥ 1.9.

---

## High-level API: `SimilarityGraph`

```python
from eigenmind import SimilarityGraph

graph = SimilarityGraph(retrieved_points, threshold=0.65, center=False)

print(graph.summary())  # components, isolates, sizes: log this first
sa = graph.spectral_attention()  # representatives + spectral vertices
sb = graph.semantic_bridges()  # spanning-tree bridges between components
sd = graph.supported_diversity(relevance, budget=12)  # explicit utilities
tags = graph.selection_tags(top_k=10, utilities=relevance)
# {chunk_id: ["SpectralAttention", "SemanticBridge", "SupportedDiversity"]}
```

Each mechanism returns a JSON-serializable dict that records why each chunk was selected. Results are cached per full set of arguments, and every call returns a copy.

### Expected input

`retrieved_points` is a list of objects (duck-typed on Qdrant points; a `types.SimpleNamespace` is enough) exposing:

| Attribute | Type | Constraint |
|---|---|---|
| `.id` | hashable | unique across the list (a `ValueError` is raised otherwise) |
| `.vector` | sequence of floats | same length for every point |
| `.payload` | `dict` (optional) | `payload["text"]` is used as chunk text when every point has one; non-dict payloads are ignored |

`SimilarityGraph` normalizes the vectors itself (optionally after mean-centring, `center=True`, and removal of the top principal directions, `remove_top=k`), builds the cosine matrix `C` and the threshold graph `W` (`C_ij > threshold`, zero diagonal).

### Main methods

| Method | Purpose |
|---|---|
| `summary()` | components, isolates and size distribution of `W` |
| `spectral_attention(...)` | Spectral Attention on `W` |
| `semantic_bridges(...)` | Semantic Bridges on `W` and `C` |
| `supported_diversity(utilities, budget, ...)` | Supported Diversity on `W`; `utilities=None` means unit utilities (a cardinality criterion with no relevance signal) |
| `diverse_coverage(budget, relevance, ...)` | Gaussian-kernel facility-location coverage |
| `mmr`, `facility_location`, `kdpp_greedy`, `farthest_first` | evaluation baselines on `C` |
| `laplacian()` | symmetric normalized Laplacian of `W` |
| `component_spectra()` | eigen-decomposition of each nontrivial component |
| `selection_tags(top_k, include_coverage, utilities)` | tags from all mechanisms; `top_k` is the budget of the budgeted ones |

## Direct functional API

```python
from eigenmind import (
    normalize_embeddings,
    cosine_matrix,
    quantile_threshold,
    threshold_graph,
    graph_summary,
    spectral_attention,
    semantic_bridges,
    supported_diversity,
)

U = normalize_embeddings(V, center=True)
C = cosine_matrix(U)
W = threshold_graph(C, quantile_threshold(C, 0.99))
print(graph_summary(W))

spectral = spectral_attention(W, chunks=texts, selection=("vertices", "medoids"))
bridges = semantic_bridges(W, C, chunks=texts)
diversity = supported_diversity(W, utilities, budget=12, chunks=texts)
print(diversity["status"], diversity["selected_utility"], diversity["upper_bound"])
```

## Configuration

Defaults live in [`defaults.py`](src/eigenmind/defaults.py); every value can be overridden per call. The `SimilarityGraph` defaults equal those of the underlying functions.

| Constant | Default | Meaning |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.65` | cosine threshold of `SimilarityGraph` |
| `SPECTRAL_POSITIVE_COUNT` | `10` | positive eigen-directions retained (completed to whole eigenvalue groups) |
| `SPECTRAL_SELECTION` | `"vertices"` | `"vertices"`, `"medoids"`, `"tails"` or a tuple |
| `SPECTRAL_ALLOCATION` | `"global"` | order directions across components or `"per_component"` |
| `SPECTRAL_LAPLACIAN` | `"normalized"` | or `"unnormalized"` |
| `SPECTRAL_MIN_RELATIVE_GAP` | `0.05` | extend the cutoff while the relative eigengap is smaller |
| `SPECTRAL_MAX_GAP_EXTENSION` | `None` | at most `positive_count` extra directions |
| `SPECTRAL_MAX_PER_SELECTION` | `5` | cap on each percentage-based selection (ties beyond it are reported) |
| `SPECTRAL_DEGREE_REGULARIZATION` | `0.0` | Qin–Rohe regularization, as a fraction of the mean degree |
| `BRIDGES_MODE` | `"spanning_tree"` | or `"top_m"`, `"all_pairs"` |
| `BRIDGES_MIN_COMPONENT_SIZE` | `2` | isolates are reported, not bridged |
| `BRIDGES_MIN_SIMILARITY` | `None` | no similarity filter |
| `DIVERSITY_SOLVER` | `"milp"` | or `"lp_greedy"` |
| `DIVERSITY_TIME_LIMIT` | `10.0` | seconds for the MILP |
| `COVERAGE_BANDWIDTH`, `COVERAGE_RELEVANCE_WEIGHT` | `0.5`, `0.25` | Gaussian kernel width and relevance weight |

## Command line

Each mechanism has a CLI with `--help`, `--self-test` and `--demo`:

```bash
eigenmind-spectral W.npy --chunks chunks.json --output spectral.json
eigenmind-bridges W.npy C.npy --chunks chunks.json --output bridges.json
eigenmind-diversity W.npy utilities.npy --budget 12 --chunks chunks.json --output diversity.json
eigenmind-scale-check
```

`chunks.json` is a UTF-8 JSON list with one string per row of `W`.

## Upgrading from 0.1

The 0.1 API was removed in 0.2. The new mechanisms are different algorithms, not drop-in replacements; to keep the 0.1 behaviour, pin `eigenmind==0.1.1`.

| Removed in 0.2 | Use instead |
|---|---|
| `find_singular_chunks`, `SimilarityGraph.singular_chunks` | `spectral_attention` |
| `analyze_laplacian_eigenvectors`, `SimilarityGraph.eigenvalue_analysis` | `spectral_attention`, `SimilarityGraph.component_spectra` |
| `build_similarity_matrix` | `cosine_matrix` + `threshold_graph` |
| `normalized_laplacian` | `SimilarityGraph.laplacian`, `component_spectrum` |
| `compute_linf_connectivity_optimizer`, `rank_relevant_but_not_obvious_chunks`, `SimilarityGraph.linf_optimizer`, `SimilarityGraph.hinge_ranking` | no direct replacement; `semantic_bridges` covers connections between components |
| `build_conflict_mask`, `theta_subgradient_approximation`, `theta_diversity_picker`, `SimilarityGraph.conflict_mask`, `theta_factor`, `theta_diversity` | `supported_diversity` |
| `SimilarityGraph.selection_tags` tags Singular/Hinge/Theta | `SimilarityGraph.selection_tags` now returns SpectralAttention/SemanticBridge/SupportedDiversity |

## Tests

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

`pytest` runs the package tests and the self-tests embedded in each module (worked examples, exact-optimum checks, invariance properties).

## Authors and license

François Oustry and Marion Schaeffer (Merlin Intelligence). MIT License.
