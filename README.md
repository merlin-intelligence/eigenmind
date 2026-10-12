# eigenmind (v1.8)

Eigenmind is an eigenvalue-based graph intelligence engine for semantic selection and chunk retrieval in high-dimensional embedding spaces.

Package **v1.8** provides three mathematically grounded selection mechanisms, upstream graph construction helpers, and evaluation baselines:

1. **Spectral Attention (2.0)** — Structural thematic representatives and extreme points (spectral vertices via Successive Projection Algorithm / SPA, and spectral medoids on row-normalized Laplacian embeddings). Replaces legacy `singular.py`.
2. **Semantic Bridges (2.0)** — Cross-component bridging points selected as a Maximum Spanning Tree (MST) of single-linkage cosine affinities (Kruskal) between connected components. Replaces legacy `connectivity.py`.
3. **Supported Diversity (2.0)** — Exact budget-constrained stable set under neighborhood support constraints ($Ax \ge 1$), solved via Mixed-Integer Linear Programming (MILP with HiGHS / `scipy.optimize.milp`) and LP-greedy fallback. Replaces legacy `theta.py`.

Upstream and downstream utilities included:
- **Graph Construction** (`graph_construction.py`) — Embedding centring, top-PC removal ("all-but-the-top"), quantile thresholding, mutual $k$-NN graphs, and topology summaries.
- **Selection Baselines** (`selection_baselines.py`) — MMR, Submodular Facility Location, $k$-DPP Greedy MAP, and Farthest-First geometric traversal.
- **Scale and Stability Check** (`scale_check.py`) — Diagnostic verification of mechanics at scale across density quantiles and noise perturbations.

---

## Installation

```bash
pip install -e .
```

For development (running tests and linter):
```bash
pip install -e ".[dev]"
```

Requirements: Python $\ge$ 3.10, NumPy, SciPy $\ge$ 1.9.

---

## Python API Usage

### High-Level API (`SimilarityGraph`)

`SimilarityGraph` holds the underlying cosine matrix $C$ and adjacency matrix $W$, memoizes intermediate decompositions, and orchestrates the selection strategies:

```python
from eigenmind import SimilarityGraph

# retrieved_points: list of objects exposing .id, .vector, and optional .payload
graph = SimilarityGraph(retrieved_points, threshold=0.5, center=True)

# 1. Topology summary
print(graph.summary())
# {'n': 250, 'edges': 1840, 'components': 8, 'isolates': 12, ...}

# 2. Spectral Attention
sa = graph.spectral_attention(positive_count=10, selection="vertices")
# Selected extreme vertices across spectral embeddings

# 3. Semantic Bridges
sb = graph.semantic_bridges(mode="spanning_tree")
# Bridges connecting separate thematic components

# 4. Supported Diversity
sd = graph.supported_diversity(budget=12, solver="milp")
# Diverse supported subset with conflict avoidance

# 5. Consolidated Selection Tags
tags = graph.selection_tags(top_k=10)
# {chunk_id: ['SpectralAttention', 'SemanticBridge', 'SupportedDiversity']}
```

### Direct Module API

All mathematical modules are directly importable and usable on standard NumPy matrices:

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
    mmr,
    facility_location,
    kdpp_greedy,
    farthest_first,
)

# Upstream graph construction
U = normalize_embeddings(V, center=True, remove_top=1)
C = cosine_matrix(U)
tau = quantile_threshold(C, q=0.99)
W = threshold_graph(C, tau)

# The Three Selection Mechanisms
sa_res = spectral_attention(W, chunks=texts, selection=("vertices", "medoids"))
sb_res = semantic_bridges(W, C, chunks=texts, mode="spanning_tree")
sd_res = supported_diversity(W, utilities, budget=12, chunks=texts, solver="milp")

# Comparison Baselines
selected_mmr = mmr(relevance, C, budget=10, lam=0.5)
selected_fl  = facility_location(C, budget=10)
selected_dpp = kdpp_greedy(C, budget=10)
```

---

## Command-Line Usage

Each module exposes a standalone CLI with `--self-test`, `--demo`, and file input:

```bash
# Run self-tests
python -m eigenmind.spectral_attention --self-test
python -m eigenmind.semantic_bridges --self-test
python -m eigenmind.supported_diversity --self-test
python -m eigenmind.graph_construction --self-test
python -m eigenmind.selection_baselines --self-test

# Run scale and stability benchmark
python -m eigenmind.scale_check

# Process adjacency matrices directly
python -m eigenmind.spectral_attention W.npy --chunks chunks.json --output spectral.json
python -m eigenmind.semantic_bridges W.npy C.npy --chunks chunks.json --output bridges.json
python -m eigenmind.supported_diversity W.npy utilities.npy --budget 12 --chunks chunks.json --output diversity.json
```

Or via installed CLI entry points:
```bash
eigenmind-spectral --self-test
eigenmind-bridges --self-test
eigenmind-diversity --self-test
eigenmind-scale-check
```

---

## Testing

Run the full pytest suite:

```bash
pytest tests/ -v
```

---

## Authors & License

- Authors: François Oustry & Marion Schaeffer (Merlin Intelligence / ORSON)
- License: MIT
