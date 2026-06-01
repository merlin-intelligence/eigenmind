"""Pure analytics for an ingested corpus: stats, wordcloud, topic modeling, similarity.

Topic modeling and similarity run on the **embeddings** already stored in Qdrant
(mean of chunk vectors per file). TF-IDF is only used *after* clustering to
extract human-readable keywords per cluster — embeddings give semantic structure,
TF-IDF gives labels for it.

This module is UI-agnostic — it returns plain dataclasses / DataFrames / numpy arrays,
leaving the Streamlit page to handle rendering. It mirrors the workflow originally
prototyped in ``corpus_analysis.ipynb``.
"""
from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
except ImportError:  # pragma: no cover
    nltk = None
    nltk_stopwords = None

try:
    from wordcloud import WordCloud
except ImportError:  # pragma: no cover
    WordCloud = None


# ── Document representation ────────────────────────────────────────────

@dataclass
class CorpusDocument:
    """One reconstructed document: title, raw text, type and (optional) mean embedding."""
    title: str
    content: str
    doc_type: str
    vector: np.ndarray | None = None

    @classmethod
    def from_filename(
        cls,
        filename: str,
        content: str,
        vector: np.ndarray | None = None,
    ) -> CorpusDocument:
        title, ext = os.path.splitext(filename)
        return cls(
            title=title,
            content=content,
            doc_type=(ext.lstrip(".").lower() or "unknown"),
            vector=vector,
        )


def documents_from_qdrant(records: dict[str, dict]) -> list[CorpusDocument]:
    """Build :class:`CorpusDocument` list from :meth:`QdrantStore.all_documents`.

    Expected shape: ``{filename: {"text": str, "vector": np.ndarray | None}}``.
    """
    return [
        CorpusDocument.from_filename(fn, r.get("text", ""), r.get("vector"))
        for fn, r in records.items()
    ]


def to_dataframe(docs: list[CorpusDocument]) -> pd.DataFrame:
    """Flatten a doc list to a DataFrame with computed character length.

    Text is normalised to Unicode NFC so that combining-accent diacritics extracted
    by some PDF/DOCX loaders (e.g. ``e`` + ``◌́``) are recomposed to single codepoints
    (``é``) — otherwise the ``\\w`` regex used by WordCloud and scikit-learn drops
    the leading vowel and yields tokens like ``mocratie`` instead of ``démocratie``.
    """
    df = pd.DataFrame({
        "title": [d.title for d in docs],
        "content": [unicodedata.normalize("NFC", d.content or "") for d in docs],
        "type": [d.doc_type for d in docs],
    })
    df["length"] = df["content"].str.len()
    return df


def _stack_vectors(docs: list[CorpusDocument]) -> np.ndarray | None:
    """Return an ``(n_docs, dim)`` matrix of L2-normalised vectors, or ``None``."""
    vecs = [d.vector for d in docs if d.vector is not None]
    if len(vecs) != len(docs) or not vecs:
        return None
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms > 0, norms, 1.0)


# ── Stopwords ───────────────────────────────────────────────────────────

def load_stopwords(languages: tuple[str, ...] = ("french", "english")) -> set[str]:
    """Return a union stopword set for the requested NLTK languages.

    Falls back to an empty set if NLTK is unavailable. Downloads the corpus on demand.
    """
    if nltk is None or nltk_stopwords is None:
        return set()
    try:
        nltk_stopwords.words(languages[0])
    except LookupError:
        nltk.download("stopwords", quiet=True)
    out: set[str] = set()
    for lang in languages:
        try:
            out.update(nltk_stopwords.words(lang))
        except OSError:
            continue
    return out


# ── Stats ───────────────────────────────────────────────────────────────

def type_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count documents per type, including a ``Total`` row at the top."""
    by_type = df["type"].value_counts().sort_index()
    rows = [{"Type": "Total", "Count": int(len(df))}]
    rows.extend({"Type": t.upper(), "Count": int(n)} for t, n in by_type.items())
    return pd.DataFrame(rows)


def length_stats(df: pd.DataFrame) -> dict[str, float]:
    return {
        "mean": float(df["length"].mean()) if len(df) else 0.0,
        "std": float(df["length"].std()) if len(df) > 1 else 0.0,
        "min": float(df["length"].min()) if len(df) else 0.0,
        "max": float(df["length"].max()) if len(df) else 0.0,
    }


# ── Wordcloud ───────────────────────────────────────────────────────────

def build_wordcloud(
    df: pd.DataFrame,
    stopwords: set[str],
    *,
    width: int = 1400,
    height: int = 600,
    max_words: int = 200,
    colormap: str = "OrRd",
):
    """Generate a ``WordCloud`` object from the concatenated corpus text.

    Returns ``None`` if the ``wordcloud`` library is missing or the corpus is empty.
    """
    if WordCloud is None:
        return None
    text = " ".join(df["content"].astype(str))
    if not text.strip():
        return None
    return WordCloud(
        width=width,
        height=height,
        background_color="#faf6f0",
        max_words=max_words,
        stopwords=stopwords,
        colormap=colormap,
        prefer_horizontal=0.9,
    ).generate(text)


# ── Topic modeling ──────────────────────────────────────────────────────

@dataclass
class TopicModel:
    """Bundle of everything produced by :func:`topic_model`."""
    best_k: int
    silhouette: float
    labels: np.ndarray
    coords: np.ndarray
    cluster_terms: dict[int, list[str]]
    cluster_label: dict[int, str]


def _label_clusters_with_tfidf(
    df: pd.DataFrame,
    labels: np.ndarray,
    stopwords: set[str],
    *,
    max_features: int,
    top_terms: int,
) -> tuple[dict[int, list[str]], dict[int, str]]:
    """For each cluster, fit a global TF-IDF then take the top terms of the cluster mean.

    Embeddings give the partition; TF-IDF gives the human-readable names.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=list(stopwords))
    tfidf = vectorizer.fit_transform(df["content"].fillna(""))
    terms = vectorizer.get_feature_names_out()

    cluster_terms: dict[int, list[str]] = {}
    cluster_label: dict[int, str] = {}
    for i in sorted({int(x) for x in labels}):
        mask = labels == i
        if not mask.any():
            continue
        cluster_mean = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        top_idx = cluster_mean.argsort()[-top_terms:][::-1]
        words = [terms[j] for j in top_idx if cluster_mean[j] > 0]
        cluster_terms[i] = words
        cluster_label[i] = ", ".join(words) if words else f"cluster {i + 1}"
    return cluster_terms, cluster_label


def topic_model(
    docs: list[CorpusDocument],
    df: pd.DataFrame,
    stopwords: set[str],
    *,
    k_min: int = 2,
    k_max: int = 20,
    max_features: int = 1000,
    random_state: int = 42,
    top_terms: int = 5,
) -> TopicModel | None:
    """Cluster documents on their **embeddings**, pick ``k`` by silhouette (cosine).

    The elbow / silhouette curve is computed internally but not returned — callers
    get only the resulting clusters. Cluster names are extracted post-hoc by
    TF-IDF on the original text, so we keep the readable keyword labels while
    benefiting from semantic clustering.

    Returns ``None`` when fewer than ``k_min`` documents have a usable vector.
    """
    vectors = _stack_vectors(docs)
    if vectors is None or len(vectors) < k_min:
        return None

    n_docs = len(vectors)
    effective_kmax = max(k_min, min(k_max, n_docs - 1))

    scores: dict[int, float] = {}
    for k in range(k_min, effective_kmax + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels_k = km.fit_predict(vectors)
        if len(set(labels_k)) < 2:
            continue
        scores[k] = float(silhouette_score(vectors, labels_k, metric="cosine"))

    if not scores:
        return None

    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels = km.fit_predict(vectors)

    n_components = 2 if n_docs >= 2 else 1
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(vectors)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros_like(coords)])

    cluster_terms, cluster_label = _label_clusters_with_tfidf(
        df, labels, stopwords, max_features=max_features, top_terms=top_terms,
    )

    return TopicModel(
        best_k=best_k,
        silhouette=scores[best_k],
        labels=labels,
        coords=coords,
        cluster_terms=cluster_terms,
        cluster_label=cluster_label,
    )


# ── Similarity ──────────────────────────────────────────────────────────

@dataclass
class SimilarityPair:
    a: str
    b: str
    score: float


def similar_pairs(
    docs: list[CorpusDocument],
    *,
    threshold: float = 0.95,
) -> list[SimilarityPair]:
    """All distinct document pairs whose **embedding** cosine similarity is ≥ ``threshold``.

    Vectors are pre-normalised, so ``v_i · v_j`` directly yields cosine similarity.
    Pairs are sorted by descending score. Returns ``[]`` if any doc lacks a vector.
    """
    vectors = _stack_vectors(docs)
    if vectors is None or len(vectors) < 2:
        return []
    sim = vectors @ vectors.T
    titles = [d.title for d in docs]
    pairs: list[SimilarityPair] = []
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            score = float(sim[i, j])
            if score >= threshold:
                pairs.append(SimilarityPair(a=titles[i], b=titles[j], score=score))
    pairs.sort(key=lambda p: p.score, reverse=True)
    return pairs
