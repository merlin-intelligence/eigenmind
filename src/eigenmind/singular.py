"""Singular chunks via eigenvector poles of the symmetric normalized Laplacian."""

from __future__ import annotations

import textwrap

import numpy as np
from scipy.linalg import eigh

from eigenmind.defaults import SIMILARITY_THRESHOLD


def build_similarity_matrix(
    embedding_matrix: np.ndarray, threshold: float = SIMILARITY_THRESHOLD
) -> np.ndarray:
    """Cosine-similarity matrix with sub-threshold values zeroed and zero diagonal.

    ``embedding_matrix`` rows must already be unit-normalized (L2 norm ≈ 1):
    the similarity is a plain row-wise dot product, not a normalized cosine,
    so unnormalized vectors would silently produce a meaningless matrix.
    """
    if embedding_matrix.size:
        norms = np.linalg.norm(embedding_matrix, axis=1)
        bad = np.where(np.abs(norms - 1.0) > 1e-3)[0]
        if len(bad):
            raise ValueError(
                "embedding_matrix rows must be unit-normalized (L2 norm ≈ 1); "
                f"{len(bad)} row(s) are not, e.g. row {int(bad[0])} has norm "
                f"{norms[bad[0]]:.4f}. Normalize each vector (v / np.linalg.norm(v)) "
                "before building a SimilarityGraph."
            )
    W = embedding_matrix @ embedding_matrix.T
    W[W < threshold] = 0
    np.fill_diagonal(W, 0)
    return W


def normalized_laplacian(W: np.ndarray) -> np.ndarray:
    degrees = np.sum(W, axis=1)
    d_inv_sqrt = np.zeros_like(degrees)
    nz = degrees > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(degrees[nz])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return np.identity(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt


def find_singular_chunks(
    retrieved_points: list,
    W: np.ndarray | None = None,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list:
    """Return points lying at the +/- poles of the first k Laplacian eigenvectors.

    Pass ``W`` (e.g. ``SimilarityGraph.W``) to reuse an already-built similarity
    matrix instead of recomputing one from ``threshold``.
    """
    if len(retrieved_points) < 2:
        return []

    id_to_point = {p.id: p for p in retrieved_points}
    ordered_ids = [p.id for p in retrieved_points]

    if W is None:
        embedding_matrix = np.array([p.vector for p in retrieved_points])
        W = build_similarity_matrix(embedding_matrix, threshold)

    L_sym = normalized_laplacian(W)
    eigenvalues, eigenvectors = eigh(L_sym)

    zero_threshold = 1e-3
    k = int(np.sum(eigenvalues < zero_threshold))

    out: list = []
    seen: set = set()
    for i in range(1, min(k + 1, len(eigenvalues))):
        ev = eigenvectors[:, i]
        for idx in (int(np.argmax(ev)), int(np.argmin(ev))):
            pid = ordered_ids[idx]
            if pid not in seen:
                seen.add(pid)
                out.append(id_to_point[pid])
    return out


def analyze_laplacian_eigenvectors(
    similarity_matrix: np.ndarray,
    id_to_point: dict,
    ordered_ids: list,
    output_filename_base: str,
):
    """Compute the Laplacian spectrum and identify the most expressive chunks per eigenvector.

    Returns (plot_path, summary_text, singular_info) where singular_info maps
    chunk-index → list of tags like "EV3(+)".

    Requires the ``matplotlib`` extra (``pip install eigenmind[viz]``).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "analyze_laplacian_eigenvectors requires matplotlib. "
            "Install it with: pip install eigenmind[viz]"
        ) from e

    summary: list[str] = []
    singular_info: dict[int, list[str]] = {}

    L_sym = normalized_laplacian(similarity_matrix)
    eigenvalues, eigenvectors = eigh(L_sym)

    zero_threshold = 1e-3
    zero_multiplicity = int(np.sum(eigenvalues < zero_threshold))
    summary.append(
        f"Found {zero_multiplicity} connected components (eigenvalues < {zero_threshold})."
    )

    n_to_plot = min(20, len(eigenvalues))
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_to_plot + 1), eigenvalues[:n_to_plot], "o-")
    plt.title("Eigenvalues of the Graph Laplacian (Spectrum)")
    plt.xlabel(f"Eigenvalue Index (Sorted) - First {zero_multiplicity} are near-zero")
    plt.ylabel("Eigenvalue")
    plt.xticks(range(1, n_to_plot + 1))
    plt.grid(True)
    plot_filename = f"{output_filename_base}_eigenvalues.png"
    try:
        plt.savefig(plot_filename)
        summary.append(f"Eigenvalue plot saved to '{plot_filename}'")
    except Exception as e:  # noqa: BLE001
        summary.append(f"Could not save eigenvalue plot. Error: {e}")
    plt.close()

    summary.append("\nMost expressive chunks along the primary semantic axes (eigenvectors):")
    for i in range(1, min(zero_multiplicity + 5, len(eigenvalues))):
        ev = eigenvectors[:, i]
        pos_idx = np.argsort(ev)[-3:]
        neg_idx = np.argsort(ev)[:3]

        for idx in pos_idx:
            singular_info.setdefault(int(idx), []).append(f"EV{i + 1}(+)")
        for idx in neg_idx:
            singular_info.setdefault(int(idx), []).append(f"EV{i + 1}(-)")

        summary.append(f"\n--- EIGENVECTOR {i + 1} (Eigenvalue: {eigenvalues[i]:.4f}) ---")
        summary.append("  (+) Positive Pole Chunks:")
        for idx in reversed(pos_idx):
            point = id_to_point[ordered_ids[idx]]
            text = " ".join(point.payload.get("text", "N/A").split())
            wrapped = textwrap.fill(
                text, width=120, initial_indent=" " * 6, subsequent_indent=" " * 6
            )
            summary.append(
                f"    - (Score: {ev[idx]:.3f}) "
                f"[{point.payload.get('chunk_number', 'N/A')} | {point.payload.get('filename', 'N/A')}]"
            )
            summary.append(f'      "{wrapped.strip()}"')

        summary.append("\n  (-) Negative Pole Chunks:")
        for idx in neg_idx:
            point = id_to_point[ordered_ids[idx]]
            text = " ".join(point.payload.get("text", "N/A").split())
            wrapped = textwrap.fill(
                text, width=120, initial_indent=" " * 6, subsequent_indent=" " * 6
            )
            summary.append(
                f"    - (Score: {ev[idx]:.3f}) "
                f"[{point.payload.get('chunk_number', 'N/A')} | {point.payload.get('filename', 'N/A')}]"
            )
            summary.append(f'      "{wrapped.strip()}"')

    return plot_filename, "\n".join(summary), singular_info
