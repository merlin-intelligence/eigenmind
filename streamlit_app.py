"""Streamlit entry point: ``streamlit run streamlit_app.py``.

Streamlit auto-discovers the ``pages/`` folder next to this file and builds the sidebar
navigation from it. Authentication and global styling are applied here once.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="eigenmind · accelerate clarity",
    page_icon="⬡",
    initial_sidebar_state="expanded",
)

from eigenmind.ui.auth import check_password
from eigenmind.ui.styles import apply_global_styles

apply_global_styles()

if not check_password():
    st.stop()

st.markdown(
    '<h1 style="font-family:\'Playfair Display\',Georgia,serif;font-weight:900;font-size:2.4rem;'
    'letter-spacing:-0.03em;color:#2a1f18;margin-bottom:0">eigenmind</h1>',
    unsafe_allow_html=True,
)
st.markdown('<div class="hero-tagline">/ accelerate clarity /</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    Welcome. Use the sidebar to navigate:
    <ul style="margin-top:0.5rem">
      <li><strong>Ingest</strong> — build your corpus from local folders, Google Drive or SharePoint.</li>
      <li><strong>Chat</strong> — ask questions over your corpus with hybrid graph + similarity retrieval.</li>
      <li><strong>Graph Explorer</strong> — visualize the knowledge graph around a prompt.</li>
      <li><strong>Manage</strong> — list and delete embedded documents by date.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
