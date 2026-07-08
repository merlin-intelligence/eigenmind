"""Manage page — list, delete documents per ingestion date."""
from __future__ import annotations

import datetime

import pandas as pd
import streamlit as st

from eigenmind.ui.auth import (
    check_password,
    current_user,
    is_admin,
    list_visible_collections,
)
from eigenmind.ui.components import empty_state, render_sidebar, section_header
from eigenmind.ui.styles import apply_global_styles
from eigenmind.vectordb.store import QdrantStore

apply_global_styles()
if not check_password():
    st.stop()
sb = render_sidebar()

section_header("/manage/", "list and delete embedded docs")

if not sb.is_connected:
    empty_state("⚡", "Qdrant offline.")
    st.stop()

store = QdrantStore(sb.qdrant_host, sb.qdrant_port)
_visible = list_visible_collections(store.list_collections())
col_labels = [label for label, _ in _visible]
col_map = {label: qdrant_name for label, qdrant_name in _visible}

if not col_labels:
    empty_state("📂", "No collections found.")
    st.stop()

col_sel, col_date = st.columns([1, 1])
with col_sel:
    collection_name = st.selectbox("Select Collection", col_labels)
with col_date:
    selected_date: datetime.date = st.date_input("Select Ingestion Date", datetime.datetime.today())

qdrant_col = col_map[collection_name]
_is_public = collection_name.startswith("[public] ")
_can_modify = is_admin() if _is_public else qdrant_col.startswith(f"{current_user()}_")

with st.expander("⚠ danger zone — delete entire collection"):
    if not _can_modify:
        st.warning(
            "You do not have permission to delete this collection. "
            + ("Only admins can delete public collections." if _is_public else "")
        )
    else:
        pt_count, _ = store.collection_stats(qdrant_col)
        if pt_count is not None:
            st.markdown(
                f'<p style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#a82020">'
                f'This will permanently delete collection <strong>{collection_name}</strong> '
                f'and all its {pt_count:,} vectors.</p>',
                unsafe_allow_html=True,
            )
        if st.checkbox(f"I confirm permanent deletion of '{collection_name}'", key="confirm_del_col"):
            if st.button("delete collection now", type="primary"):
                try:
                    store.delete_collection(qdrant_col)
                    st.success(f"'{collection_name}' deleted.")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error: {e}")

st.info(f"Viewing documents in **{collection_name}** ingested on **{selected_date.isoformat()}**")
with st.spinner("fetching documents..."):
    docs = store.documents_for_date(qdrant_col, selected_date)

if not docs:
    empty_state("💨", "No documents found for this date.")
else:
    st.dataframe(
        pd.DataFrame([{"Filename": k, "Chunks": v} for k, v in docs.items()]),
        use_container_width=True,
    )

    if not _can_modify:
        st.info("Read-only view — you do not have permission to delete documents from this collection.")
    else:
        st.markdown("---")
        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.75rem;'
            'color:#a82020;text-transform:uppercase;letter-spacing:0.1em">⚠️ danger zone</p>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button(f"Delete ALL for {selected_date.isoformat()}", type="primary"):
                try:
                    store.delete_for_date(qdrant_col, selected_date)
                    st.success(f"Deleted all points for {selected_date.isoformat()} in {collection_name}")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error: {e}")

        with c2:
            doc_to_delete = st.selectbox("Select specific document to delete", [""] + sorted(docs.keys()))
            if doc_to_delete and st.button(f"Delete '{doc_to_delete}'"):
                try:
                    store.delete_document_on_date(qdrant_col, doc_to_delete, selected_date)
                    st.success(f"Deleted '{doc_to_delete}'")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error: {e}")
