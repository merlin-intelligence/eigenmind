"""Tests of the SimilarityGraph wrapper: caching, defaults and edge cases."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from eigenmind import (
    SimilarityGraph,
    defaults,
    semantic_bridges,
    spectral_attention,
    supported_diversity,
)


def _points_with_isolates(seed: int = 0):
    """Three tight clusters of six chunks plus two unrelated chunks."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, 16))
    V = np.vstack(
        [centers[i] + 0.15 * rng.normal(size=(6, 16)) for i in range(3)]
        + [rng.normal(size=(2, 16))]
    )
    return [
        SimpleNamespace(id=f"c{i}", vector=v.tolist(), payload={"text": f"chunk {i}"})
        for i, v in enumerate(V)
    ]


@pytest.fixture
def graph():
    return SimilarityGraph(_points_with_isolates(), threshold=0.6)


def test_graph_has_isolates(graph):
    summary = graph.summary()
    assert summary["isolates"] == 2
    assert summary["components"] == 5


def test_spectral_cache_respects_every_argument(graph):
    capped = graph.spectral_attention(max_per_selection=1)
    uncapped = graph.spectral_attention(max_per_selection=None)
    assert capped["config"]["max_per_selection"] == 1
    assert uncapped["config"]["max_per_selection"] is None
    assert graph.spectral_attention(min_relative_gap=0.0)["config"]["min_relative_gap"] == 0.0
    cached = len(graph._cache)
    assert graph.spectral_attention(max_per_selection=1) == capped
    assert len(graph._cache) == cached


def test_cached_results_are_isolated_from_callers(graph):
    graph.spectral_attention()["selected_chunks"].clear()
    graph.semantic_bridges()["selected_chunks"].clear()
    graph.supported_diversity(budget=3)["selected_indices"].clear()
    graph.kernel()[:] = 0.0
    assert graph.spectral_attention()["selected_chunks"]
    assert graph.semantic_bridges()["selected_chunks"]
    assert graph.supported_diversity(budget=3)["selected_indices"]
    assert np.all(np.diag(graph.kernel()) == 1.0)
    assert any("SpectralAttention" in t for t in graph.selection_tags().values())


def test_bridges_cache_respects_min_component_size(graph):
    with_isolates = graph.semantic_bridges(min_component_size=1)
    without = graph.semantic_bridges()
    assert without["excluded_components"] == [3, 4]
    assert with_isolates["excluded_components"] == []
    tagged = {pid for pid, t in graph.selection_tags().items() if "SemanticBridge" in t}
    assert not tagged & {"c18", "c19"}


def test_supported_diversity_cache_respects_utilities(graph):
    a = graph.supported_diversity(budget=3)
    b = graph.supported_diversity(np.arange(graph.n, dtype=float), budget=3)
    assert a is not b
    assert b["selected_utility"] > a["selected_utility"]


def test_laplacian_with_isolates(graph):
    L = graph.laplacian()
    assert L.shape == (graph.n, graph.n)
    assert np.isfinite(L).all()
    assert np.allclose(L, L.T)
    assert L[18, 18] == 1.0 and L[19, 19] == 1.0  # isolates
    spectra = graph.component_spectra()
    assert [len(s["indices"]) for s in spectra] == [6, 6, 6]
    for s in spectra:
        assert abs(s["eigenvalues"][0]) < 1e-10


def test_kernel_memoized_per_bandwidth(graph):
    K = graph.kernel(bandwidth=0.5)
    cached = len(graph._cache)
    assert np.array_equal(graph.kernel(bandwidth=0.5), K)
    assert len(graph._cache) == cached
    assert not np.allclose(graph.kernel(bandwidth=0.5), graph.kernel(bandwidth=1.0))


def test_payload_text_used_as_chunk_text(graph):
    res = graph.spectral_attention()
    assert all(c["text"].startswith("chunk ") for c in res["selected_chunks"])


def test_wrapper_defaults_match_module_defaults():
    pairs = [
        (SimilarityGraph.spectral_attention, spectral_attention),
        (SimilarityGraph.semantic_bridges, semantic_bridges),
        (SimilarityGraph.supported_diversity, supported_diversity),
    ]
    for wrapper, function in pairs:
        wrapped = inspect.signature(wrapper).parameters
        direct = inspect.signature(function).parameters
        for name, param in wrapped.items():
            if name in direct and name not in (
                "self",
                "chunks",
                "utilities",
                "budget",
                "selection",
            ):
                assert param.default == direct[name].default, (function.__name__, name)
    assert defaults.SIMILARITY_THRESHOLD == 0.65


def test_centering_degenerate_pools():
    single = [SimpleNamespace(id="a", vector=[1.0, 2.0, 3.0])]
    duplicates = [SimpleNamespace(id=i, vector=[1.0, 2.0, 3.0]) for i in ("a", "b")]
    with pytest.warns(RuntimeWarning, match="plain row normalization"):
        assert SimilarityGraph(single, center=True).n == 1
    with pytest.warns(RuntimeWarning, match="plain row normalization"):
        g = SimilarityGraph(duplicates, center=True)
    assert g.W[0, 1] == pytest.approx(1.0)


def test_empty_graph():
    g = SimilarityGraph([])
    assert g.selection_tags() == {}
    assert g.component_spectra() == []
    full = SimilarityGraph(_points_with_isolates())
    for method, kwargs in (
        ("spectral_attention", {}),
        ("semantic_bridges", {}),
        ("supported_diversity", {"budget": 3}),
    ):
        empty = getattr(g, method)(**kwargs)
        assert set(empty) == set(getattr(full, method)(**kwargs)), method
        json.dumps(empty, allow_nan=False)
    assert g.spectral_attention()["selected_chunks"] == []
    assert g.supported_diversity()["selected_indices"] == []


def test_duplicate_ids_rejected():
    points = [
        SimpleNamespace(id="a", vector=[1.0, 0.0]),
        SimpleNamespace(id="a", vector=[0.0, 1.0]),
    ]
    with pytest.raises(ValueError, match="unique"):
        SimilarityGraph(points)


def test_non_dict_payload_ignored():
    points = [
        SimpleNamespace(id=i, vector=v, payload="not a dict")
        for i, v in enumerate([[1.0, 0.0], [1.0, 0.1]])
    ]
    g = SimilarityGraph(points)
    assert all(c["text"] is None for c in g.spectral_attention()["selected_chunks"])
    assert g.selection_tags()


def test_results_are_json_serializable(graph):
    for result in (
        graph.spectral_attention(),
        graph.semantic_bridges(),
        graph.supported_diversity(budget=3),
        graph.diverse_coverage(budget=3),
    ):
        json.dumps(result, allow_nan=False)
    assert [c["eligible"] for c in graph.semantic_bridges()["components"]] == [
        True,
        True,
        True,
        False,
        False,
    ]
