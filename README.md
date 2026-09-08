# eigenmind
Eigenmind is a decision-intelligence engine that turns tacit expert knowledge into reusable decision frameworks, ideal for data-scarce settings. Its core uses eigenvalue-based optimisation on semantic graphs.

## Install

```bash
pip install eigenmind
```

Need the eigenvalue-spectrum plotting helper (`eigenvalue_analysis`)? Install
the `viz` extra too:

```bash
pip install "eigenmind[viz]"
```

To track an unreleased change instead of the latest PyPI release, install
straight from GitHub:

```bash
pip install git+https://github.com/merlin-intelligence/eigenmind.git
```

## Usage

```python
from eigenmind import SimilarityGraph

# retrieved_points: objects with .id, .vector, .payload (e.g. Qdrant points)
graph = SimilarityGraph(retrieved_points)
tags = graph.selection_tags(top_k=10)  # {chunk_id: ["Singular", "Hinge", "Theta"]}
```

### Expected input data

`retrieved_points` is a plain list of objects — any class works (duck-typing,
modeled after Qdrant's point objects), each exposing:

| Attribute | Type | Constraint |
|---|---|---|
| `.id` | hashable (`int`, `str`, `UUID`, ...) | unique across the list — used as a dict key |
| `.vector` | sequence of floats (`list` or `np.ndarray`) | **same length for every point**, and **unit-normalized (L2 norm ≈ 1)** |
| `.payload` | `dict` | only read by `eigenvalue_analysis`, via `.get("text")`, `.get("chunk_number")`, `.get("filename")` |

The library computes cosine similarity as a plain dot product between
vectors (`embedding_matrix @ embedding_matrix.T`) — it does **not** normalize
your vectors for you. Passing unnormalized vectors raises a `ValueError`
(from `build_similarity_matrix`, used internally by `SimilarityGraph` and
`find_singular_chunks`); normalize each vector yourself first, e.g.
`v / np.linalg.norm(v)`.

A `types.SimpleNamespace(id=..., vector=..., payload={...})` is enough — no
dependency on Qdrant is required, as shown in the test suite. Because the
interface is duck-typed on these three attributes, results from any
vector store (Pinecone, Weaviate, Milvus, pgvector, Chroma, ...) work as-is,
either directly or after a thin adapter mapping their result objects to
`.id` / `.vector` / `.payload`.

`SimilarityGraph` memoizes the similarity matrix and exposes four ranking
strategies over a set of embedded chunks:

- **Singular chunks** — points at the +/- poles of the normalized Laplacian's
  first eigenvectors (`singular.py`).
- **Hinge ranking** — "relevant but not obvious" chunks via an ℓ∞-connectivity
  optimizer (`connectivity.py`).
- **Theta diversity** — a diverse subset picked via a Lovász θ approximation on
  the conflict graph (`theta.py`).
- **Eigenvalue analysis** — spectrum plot + most expressive chunks per
  eigenvector, for exploratory/debugging use. Requires the `viz` extra
  (`pip install eigenmind[viz]`).


## Configuration

Every tunable is a plain keyword argument with a default from
[`defaults.py`](src/eigenmind/defaults.py) — override it per call/instance,
there is no global state to mutate.

| Constant | Default | Meaning | Override via |
|---|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.65` | Minimum cosine similarity kept as an edge weight; pairs below it are zeroed out of `W`. | `SimilarityGraph(points, threshold=...)`, `build_similarity_matrix(..., threshold=...)` |
| `THETA_CONFLICT_THRESHOLD` | `= SIMILARITY_THRESHOLD` | Similarity level above which two chunks are considered "conflicting" (same info) for the θ diversity graph. | `SimilarityGraph.conflict_mask(threshold=...)`, `build_conflict_mask(..., threshold=...)` |
| `THETA_RANK` | `24` | Rank of the low-rank factor `Y` (`X ≈ Y Yᵀ`) used by the diversity picker — caps how many effective directions the θ approximation keeps. | `SimilarityGraph.theta_factor(rank=...)`, `theta_subgradient_approximation(..., rank=...)` |
| `THETA_MAX_ITERS` | `400` | Number of subgradient-descent iterations for the Lovász θ approximation. | `SimilarityGraph.theta_factor(max_iters=...)`, `theta_subgradient_approximation(..., max_iters=...)` |
| `THETA_STEP0` | `0.25` | Initial subgradient step size, decayed as `step0 / sqrt(k)` over iterations. | `SimilarityGraph.theta_factor(step0=...)`, `theta_subgradient_approximation(..., step0=...)` |
| `THETA_DIAG_SHIFT` | `1e-2` | Diagonal regularization added when the dual matrix isn't Cholesky-decomposable (numerical stabilizer). | `SimilarityGraph.theta_factor(diag_shift=...)`, `theta_subgradient_approximation(..., diag_shift=...)` |

These defaults are the values the algorithms were originally tuned against;
adjust them if your embeddings' similarity distribution differs
significantly (e.g. a different embedding model or normalization scheme).

## Tests

```bash
pip install -e ".[dev]"
pytest
```
