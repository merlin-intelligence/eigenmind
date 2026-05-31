"""Streamlit entry point: ``streamlit run streamlit_app.py``.

Authentication and global styling are applied here once.
Custom navigation structure defined for the sidebar.
"""
from __future__ import annotations

import streamlit as st
from eigenmind.ui.auth import check_password
from eigenmind.ui.styles import apply_global_styles

def home_page():
    """Main welcome page content."""
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
          <li><strong>/enrich corpus/</strong> — build your corpus from local folders, Google Drive or SharePoint.</li>
          <li><strong>/analyze corpus/</strong> — inventory, topics and similarities across your corpus.</li>
          <li><strong>/ask/</strong> — ask questions over your corpus with hybrid graph + similarity retrieval.</li>
          <li><strong>/explore graphs/</strong> — visualize the knowledge graph around a prompt.</li>
          <li><strong>/manage/</strong> — list and delete embedded documents by date.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Page config must be first
st.set_page_config(
    layout="wide",
    page_title="eigenmind · accelerate clarity",
    page_icon="⬡",
    initial_sidebar_state="expanded",
)

apply_global_styles()

if not check_password():
    st.stop()

# Define navigation structure with custom labels as requested
pg = st.navigation([
    st.Page(home_page, title="Eigenmind Cognitive Maps", icon="🧠", default=True),
    st.Page("pages/1_Ingest.py", title="/enrich corpus/", icon="📥"),
    st.Page("pages/2_Corpus_Analysis.py", title="/analyze corpus/", icon="📊"),
    st.Page("pages/3_Chat.py", title="/ask/", icon="💬"),
    st.Page("pages/4_Graph_Explorer.py", title="/explore graphs/", icon="🕸️"),
    st.Page("pages/5_Manage.py", title="/manage/", icon="⚙️"),
])

pg.run()
