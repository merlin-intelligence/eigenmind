"""Tests of the SimilarityGraph wrapper: caching, defaults and edge cases."""

from __future__ import annotations

import inspect
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
    assert graph.spectral_attention(max_per_selection=1) is capped


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
    spectra = graph.component_spectra()
    assert [len(s["indices"]) for s in spectra] == [6, 6, 6]
    for s in spectra:
        assert abs(s["eigenvalues"][0]) < 1e-10


def test_kernel_memoized_per_bandwidth(graph):
    assert graph.kernel(bandwidth=0.5) is graph.kernel(bandwidth=0.5)
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


def test_empty_graph():
    g = SimilarityGraph([])
    assert g.selection_tags() == {}
    assert g.spectral_attention()["selected_chunks"] == []
    assert g.component_spectra() == []
