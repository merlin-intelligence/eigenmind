"""Chat page — hybrid RAG (similarity search + graph) over a Qdrant collection."""
from __future__ import annotations

import re
import tempfile

import streamlit as st

from eigenmind.pipelines.rag import GraphExplorer
from eigenmind.ui.auth import (
    check_password,
    display_name_from,
    qdrant_collection_for,
)
from eigenmind.ui.components import (
    NebiusClient,
    empty_state,
    get_embedder,
    load_nlp,
    render_sidebar,
    section_header,
)
from eigenmind.ui.styles import apply_global_styles, badge
from eigenmind.vectordb.store import QdrantStore

apply_global_styles()
if not check_password():
    st.stop()
sb = render_sidebar()

section_header("/ask/", "hybrid RAG · graph + similarity")

if not sb.is_connected:
    empty_state("⚡", "Qdrant offline.")
    st.stop()

store = QdrantStore(sb.qdrant_host, sb.qdrant_port)
existing_cols = sorted(c for c in (display_name_from(c) for c in store.list_collections()) if c)

if not existing_cols:
    empty_state("📂", "No collections found. Ingest documents in <strong>/enrich corpus/</strong> first.")
    st.stop()

col_q1, col_q2 = st.columns([1, 2])
with col_q1:
    collection_name = st.selectbox("collection", existing_cols)
    pt_count, _ = store.collection_stats(qdrant_collection_for(collection_name))
    if pt_count:
        st.markdown(
            f'<p style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#8a6a50">'
            f'{pt_count:,} vectors</p>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<p style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
        'color:#8a6a50;margin-top:1rem">retrieval settings</p>',
        unsafe_allow_html=True,
    )
    num_similar = st.slider("similarity chunks", 0, 20, 5)
    num_singular = st.slider("singular chunks", 0, 20, 5)

    short_model = sb.nebius_model.split("/")[-1]
    st.markdown(
        f'<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;color:#8a6a50">'
        f'model: {short_model}</p>',
        unsafe_allow_html=True,
    )

with col_q2:
    prompt = st.text_area(
        "your question",
        "what is intellectual humility?",
        height=120,
        help="Ask anything about your corpus. The system blends semantic search with graph analysis.",
    )

    if not sb.nebius_api_key:
        st.markdown(
            '<div class="info-box" style="border-left-color:#a82020;background:#f5e8e8">'
            '⚠ AI Hub API key is not configured. Please contact your administrator.</div>',
            unsafe_allow_html=True,
        )

    ask_btn = st.button("▶ get answer", type="primary", disabled=(not sb.nebius_api_key))


def _run_query() -> None:
    with st.spinner("retrieving context and generating…"):
        try:
            qdrant_col = qdrant_collection_for(collection_name)

            embedder = get_embedder(sb.selected_device)

            # 1. Similarity retrieval
            query_vec = embedder.encode_query(prompt).tolist()
            sim_chunks = store.similarity_search(qdrant_col, query_vec, limit=num_similar)
            for c in sim_chunks:
                c["source_type"] = "Similarity Search"
            retrieved = list(sim_chunks)

            # 2. Graph retrieval (singular/hinge/theta)
            with tempfile.TemporaryDirectory() as tmp_dir:
                explorer = GraphExplorer(load_nlp(), store=store, embedder=embedder)
                artifacts = explorer.explore(qdrant_col, prompt, output_dir=tmp_dir)

                added = 0
                for ch in artifacts.ranked_chunks:
                    if added >= num_singular: break
                    if not any(c["text"] == ch["text"] for c in retrieved):
                        retrieved.append({
                            "text": ch["text"],
                            "filename": ch["filename"],
                            "chunk_number": ch["chunk_id"],
                            "source_type": f"Singular ({', '.join(ch['methods'])})",
                            "score": None,
                        })
                        added += 1

            # 3. Build context, call LLM
            context = ""
            for i, c in enumerate(retrieved):
                context += f"--- Chunk {i + 1} ({c['source_type']}) ---\n{c['text']}\n\n"

            llm = NebiusClient(model=sb.nebius_model, api_key=sb.nebius_api_key)
            answer = llm.chat(
                system_prompt=(
                    "You are a helpful assistant. Answer the question based on the provided context. "
                    "Explicitly cite the chunk number (e.g. [Chunk 1]) for every piece of information you use."
                ),
                user_content=f"Context:\n{context}\n\nQuestion: {prompt}",
            )

            st.markdown("---")
            st.markdown(
                '<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
                'letter-spacing:0.15em;color:#8a6a50;text-transform:uppercase;'
                'margin-bottom:0.5rem">⬡ answer</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#ffffff;border:1px solid #c0b4a8;border-left:3px solid #c44a28;'
                f'border-radius:10px;padding:1.4rem 1.6rem;font-size:0.95rem;line-height:1.75;color:#2a1f18">'
                f'{answer}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown(
                '<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
                'letter-spacing:0.15em;color:#8a6a50;text-transform:uppercase;'
                'margin-bottom:0.5rem">references</p>',
                unsafe_allow_html=True,
            )
            for i, c in enumerate(retrieved):
                src = c["source_type"]
                is_graph = any(k in src for k in ("Graph", "Singular", "Hinge", "Theta"))
                badge_html = badge("graph" if is_graph else "sim")
                score_str = f" · score {c['score']}" if c.get("score") else ""
                with st.expander(f"[{i + 1}] {c['filename']} · chunk {c['chunk_number']}{score_str}"):
                    st.markdown(
                        f'<p style="margin-bottom:0.5rem">{badge_html} '
                        f'<span style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
                        f'color:#8a6a50;margin-left:0.5rem">{src}</span></p>',
                        unsafe_allow_html=True,
                    )
                    st.write(c["text"])

            content_parts = [
                f"Question: {prompt}\n\n{'=' * 40}\nLLM Answer:\n{'=' * 40}\n\n{answer}\n\n"
                f"{'=' * 40}\nReferences:\n{'=' * 40}\n\n"
            ]
            for i, c in enumerate(retrieved):
                content_parts.append(
                    f"[{i + 1}] {c['filename']} (Chunk {c['chunk_number']}) — {c['source_type']}\n"
                    f"---\n{c['text']}\n---\n\n"
                )
            sanitized = re.sub(r"[^\w\s-]", "", prompt).strip().replace(" ", "_")
            st.download_button(
                "⬇ download answer + references",
                "".join(content_parts).encode("utf-8"),
                file_name=f"answer_{sanitized[:30]}.txt",
                mime="text/plain",
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to generate answer: {e}")
            st.exception(e)


if ask_btn and collection_name and prompt and sb.nebius_api_key:
    _run_query()
elif ask_btn and not sb.nebius_api_key:
    st.warning("AI Hub API key is not configured. Please contact your administrator.")
elif ask_btn:
    st.warning("Provide a collection and a question.")
