"""Corpus Analysis page — type counts, sizes, wordcloud, topics, similarity."""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from eigenmind.config import DOCUMENT_COLORS
from eigenmind.core.corpus_analysis import (
    build_wordcloud,
    documents_from_qdrant,
    length_stats,
    load_stopwords,
    similar_pairs,
    to_dataframe,
    topic_model,
    type_counts,
)
from eigenmind.ui.auth import (
    check_password,
    list_visible_collections,
)
from eigenmind.ui.components import empty_state, render_sidebar, section_header
from eigenmind.ui.styles import apply_global_styles
from eigenmind.vectordb.store import QdrantStore

apply_global_styles()
if not check_password():
    st.stop()
sb = render_sidebar()

section_header("/analyze corpus/", "explore the shape of your corpus")

if not sb.is_connected:
    empty_state("⚡", "Qdrant offline.")
    st.stop()

store = QdrantStore(sb.qdrant_host, sb.qdrant_port)
_visible = list_visible_collections(store.list_collections())
col_labels = [label for label, _ in _visible]
col_map = {label: qdrant_name for label, qdrant_name in _visible}

if not col_labels:
    empty_state("📂", "No collections found. Ingest documents via <strong>/enrich corpus/</strong> first.")
    st.stop()

collection_name = st.selectbox("collection", col_labels)
qdrant_col = col_map[collection_name]

with st.spinner("reconstructing documents from chunks…"):
    records = store.all_documents(qdrant_col)

if not records:
    empty_state("💨", "This collection is empty.")
    st.stop()

docs = documents_from_qdrant(records)
df = to_dataframe(docs)
stopwords = load_stopwords(("french", "english"))

missing_vectors = sum(1 for d in docs if d.vector is None)
if missing_vectors:
    st.warning(
        f"{missing_vectors} document(s) have no stored embedding — they will be excluded "
        "from topic modeling and similarity analysis."
    )

# ── Color palette aligned with the rust/earth design system ───────────
PALETTE = [
    "#c44a28", "#3d7a4a", "#4a7fa8", "#c08030", "#8a3a1e",
    "#6a5040", "#9a6010", "#2e6b3a", "#a82020", "#1e6090",
    *DOCUMENT_COLORS,
]

# Match plot aesthetic to the page background
plt.rcParams.update({
    "figure.facecolor": "#faf6f0",
    "axes.facecolor": "#faf6f0",
    "axes.edgecolor": "#c0b4a8",
    "axes.labelcolor": "#2a1f18",
    "xtick.color": "#8a6a50",
    "ytick.color": "#8a6a50",
    "axes.titlecolor": "#2a1f18",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


# ─── Section 1 — Document counts by type ──────────────────────────────
st.markdown("### document inventory")
counts_df = type_counts(df)
col_table, col_metric = st.columns([1.2, 1])
with col_table:
    st.dataframe(counts_df, use_container_width=True, hide_index=True)
with col_metric:
    st.markdown(
        f'<div class="metric-card"><div class="label">documents</div>'
        f'<div class="value">{len(df)}</div>'
        f'<div class="sub">{df["type"].nunique()} distinct format(s)</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")


# ─── Section 2 — Size distribution ────────────────────────────────────
st.markdown("### document sizes")
stats = length_stats(df)
m1, m2, m3, m4 = st.columns(4)
for col, (label, key, suffix) in zip(
    (m1, m2, m3, m4),
    [("mean", "mean", "chars"), ("std dev", "std", "chars"),
     ("min", "min", "chars"), ("max", "max", "chars")],
    strict=True,
):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="value">{int(stats[key]):,}</div>'
            f'<div class="sub">{suffix}</div></div>',
            unsafe_allow_html=True,
        )

fig, ax = plt.subplots(figsize=(14, 5))
df_sorted = df.sort_values("length", ascending=False).reset_index(drop=True)
ax.bar(range(len(df_sorted)), df_sorted["length"], color="#c44a28", alpha=0.85, edgecolor="#8a3a1e")
ax.axhline(stats["mean"], color="#3d7a4a", linestyle="--", linewidth=1.5,
           label=f"mean: {int(stats['mean']):,} chars")
ax.set_xticks(range(len(df_sorted)))
ax.set_xticklabels(df_sorted["title"], rotation=80, ha="right", fontsize=7)
ax.set_ylabel("characters")
ax.legend(loc="upper right", frameon=False)
plt.tight_layout()
st.pyplot(fig, clear_figure=True)

st.markdown("---")


# ─── Section 3 — Wordcloud ────────────────────────────────────────────
st.markdown("### wordcloud")
with st.spinner("generating wordcloud…"):
    wc = build_wordcloud(df, stopwords)
if wc is None:
    st.info("Wordcloud unavailable — install the `wordcloud` package or check that the corpus has text.")
else:
    fig_wc, ax_wc = plt.subplots(figsize=(14, 6))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    plt.tight_layout()
    st.pyplot(fig_wc, clear_figure=True)

st.markdown("---")


# ─── Section 4 — Topic modeling ───────────────────────────────────────
st.markdown("### topic modeling")
st.markdown(
    '<p style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#8a6a50">'
    'embedding-based KMeans · k chosen by silhouette (cosine) over 2..20 · '
    'cluster labels extracted post-hoc by TF-IDF</p>',
    unsafe_allow_html=True,
)

with st.spinner("clustering documents on embeddings…"):
    model = topic_model(docs, df, stopwords)

if model is None:
    st.info("Not enough documents with embeddings to cluster (need at least 2).")
else:
    st.markdown(
        f'<div class="info-box"><strong>{model.best_k} clusters</strong> '
        f'· silhouette score <strong>{model.silhouette:.3f}</strong></div>',
        unsafe_allow_html=True,
    )

    df_clustered = df.copy()
    df_clustered["cluster"] = model.labels

    col_map, col_pie = st.columns([1.4, 1])

    with col_map:
        st.markdown("**document map**")
        fig_map, ax_map = plt.subplots(figsize=(9, 7))
        for i in range(model.best_k):
            mask = model.labels == i
            ax_map.scatter(
                model.coords[mask, 0],
                model.coords[mask, 1],
                color=PALETTE[i % len(PALETTE)],
                s=120,
                alpha=0.8,
                edgecolors="#2a1f18",
                linewidths=0.6,
                label=f"#{i + 1} · {model.cluster_label[i]}",
            )
        ax_map.set_xlabel("PCA 1 (embedding space)")
        ax_map.set_ylabel("PCA 2 (embedding space)")
        ax_map.legend(fontsize=7, loc="best", framealpha=0.9)
        plt.tight_layout()
        st.pyplot(fig_map, clear_figure=True)

    with col_pie:
        st.markdown("**distribution**")
        share = df_clustered["cluster"].value_counts().sort_index()
        fig_pie, ax_pie = plt.subplots(figsize=(7, 7))
        colors = [PALETTE[i % len(PALETTE)] for i in share.index]
        wedges, _, autotexts = ax_pie.pie(
            share.values,
            labels=[f"#{i + 1}" for i in share.index],
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            wedgeprops={"edgecolor": "#faf6f0", "linewidth": 2},
            textprops={"fontsize": 9, "color": "#2a1f18"},
        )
        for at in autotexts:
            at.set_color("#fff8f4")
            at.set_fontweight("bold")
        plt.tight_layout()
        st.pyplot(fig_pie, clear_figure=True)

    st.markdown("**cluster contents**")
    for i in range(model.best_k):
        titles = df_clustered[df_clustered["cluster"] == i]["title"].tolist()
        color = PALETTE[i % len(PALETTE)]
        header = (
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            f'background:{color};margin-right:8px;vertical-align:middle"></span>'
            f'<strong>Cluster #{i + 1}</strong> · '
            f'<span style="color:#8a6a50">{model.cluster_label[i]}</span> · '
            f'<span style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#8a6a50">'
            f'{len(titles)} doc(s)</span>'
        )
        with st.expander(f"Cluster #{i + 1} — {model.cluster_label[i]}  ({len(titles)} doc(s))"):
            st.markdown(header, unsafe_allow_html=True)
            for t in titles:
                st.markdown(f"• {t}")

st.markdown("---")


# ─── Section 5 — Near-duplicates (scores ≥ 0.99) ──────────────────────
NEAR_DUPLICATE_THRESHOLD = 0.99

st.markdown("### document similarity")
st.markdown(
    '<p style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#8a6a50">'
    f'cosine similarity between document embeddings · only pairs ≥ {NEAR_DUPLICATE_THRESHOLD} '
    'are shown</p>',
    unsafe_allow_html=True,
)

with st.spinner("computing cosine similarity…"):
    pairs = similar_pairs(docs, threshold=NEAR_DUPLICATE_THRESHOLD)

if not pairs:
    empty_state("✨", f"No near-duplicate pairs (≥ {NEAR_DUPLICATE_THRESHOLD}) — your corpus is well differentiated.")
else:
    st.markdown(
        f'<p style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#8a6a50">'
        f'{len(pairs)} near-duplicate pair(s) detected</p>',
        unsafe_allow_html=True,
    )

    bar_color = "#a82020"
    tag = "near-duplicate"
    tag_bg = "#f5e0e0"
    for p in pairs:
        bar_width = int(8 + 92 * (p.score - NEAR_DUPLICATE_THRESHOLD) / max(1e-6, 1.0 - NEAR_DUPLICATE_THRESHOLD))
        st.markdown(
            f"""
            <div style="background:#faf6f0;border:1px solid #d8d0c6;border-left:4px solid {bar_color};
                        border-radius:10px;padding:0.8rem 1rem;margin-bottom:0.5rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;
                            margin-bottom:0.4rem;">
                    <div style="font-family:'Inter',sans-serif;font-size:0.92rem;color:#2a1f18">
                        <strong>{p.a}</strong>
                        <span style="color:#8a6a50;margin:0 0.4rem">↔</span>
                        <strong>{p.b}</strong>
                    </div>
                    <div>
                        <span style="background:{tag_bg};color:{bar_color};
                                     font-family:'DM Mono',monospace;font-size:0.65rem;
                                     letter-spacing:0.06em;text-transform:uppercase;
                                     padding:2px 8px;border-radius:4px;font-weight:700;">
                            {tag}
                        </span>
                        <span style="font-family:'Playfair Display',serif;font-weight:800;
                                     color:{bar_color};font-size:1.2rem;margin-left:0.6rem">
                            {p.score:.3f}
                        </span>
                    </div>
                </div>
                <div style="background:#ece6de;border-radius:4px;height:6px;overflow:hidden">
                    <div style="background:linear-gradient(90deg,{bar_color},#e07040);
                                width:{bar_width}%;height:100%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
