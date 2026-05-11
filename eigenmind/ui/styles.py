"""Global CSS injected into every page via :func:`apply_global_styles`."""
from __future__ import annotations

import streamlit as st

_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400;1,700&family=DM+Mono:ital,wght@0,400;0,500;1,400&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg:#f2ede6; --bg-card:#ffffff; --bg-hover:#ece6de; --bg-sidebar:#ece6de;
    --border:#c0b4a8; --border-soft:#d8d0c6; --border-glow:#c44a2830;
    --rust:#c44a28; --rust-dim:#c44a2866; --rust-pale:#c44a2812;
    --brown:#8a6a50; --green:#3d7a4a; --red:#a82020;
    --text:#2a1f18; --text-muted:#8a6a50;
    --font-head:'Playfair Display',Georgia,serif;
    --font-mono:'DM Mono',monospace;
    --font-body:'Inter',sans-serif;
    --radius:10px; --radius-lg:16px;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}
[data-testid="stAppViewContainer"] > .main { background-color: var(--bg) !important; }
[data-testid="stSidebar"] { background: var(--bg-sidebar) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] *:not([class*="material"]):not([class*="Material"]):not([class*="icon"]):not([class*="Icon"]):not([data-testid*="Icon"]) { font-family: var(--font-body) !important; }
[data-testid="stIconMaterial"],
[data-testid*="Icon"],
[class*="material-icons"],
[class*="material-symbols"],
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapsedControl"] span,
[data-testid="baseButton-headerNoPadding"] span {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
}

h1, h2, h3, h4 { font-family: var(--font-head) !important; color: var(--text) !important; letter-spacing: -0.02em !important; }
h1 { font-size: 2.2rem !important; font-weight: 800 !important; }
h2 { font-weight: 700 !important; color: var(--text-muted) !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; font-size: 0.75rem !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; }

p, li, label, [data-testid="stMarkdownContainer"] p {
    font-family: var(--font-body) !important; color: var(--text) !important; line-height: 1.65 !important;
}

.stButton > button {
    background: transparent !important;
    border: 1.5px solid var(--rust) !important; color: var(--rust) !important;
    font-family: var(--font-mono) !important; font-size: 0.78rem !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    border-radius: var(--radius) !important; padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { background: var(--rust) !important; color: #fff8f4 !important; }
.stButton > button[kind="primary"] {
    background: var(--rust) !important; color: #fff8f4 !important;
    font-weight: 700 !important; border-color: var(--rust) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #a83c1e !important; transform: translateY(-1px) !important;
    box-shadow: 0 3px 12px var(--rust-dim) !important;
}

.stTextInput > div > div > input, .stTextArea > div > div > textarea,
.stNumberInput > div > div > input, .stSelectbox > div > div > div {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; color: var(--text) !important;
    font-family: var(--font-body) !important; transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
    border-color: var(--rust) !important; box-shadow: 0 0 0 2px var(--border-glow) !important;
}

[data-testid="stRadio"] > div { gap: 0.5rem !important; }
[data-testid="stRadio"] label {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; padding: 0.4rem 1rem !important;
    font-family: var(--font-mono) !important; font-size: 0.78rem !important;
    color: var(--text-muted) !important; cursor: pointer !important; transition: all 0.15s !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    border-color: var(--rust) !important; color: var(--rust) !important; background: var(--rust-pale) !important;
}

[data-testid="stAlert"] { border-radius: var(--radius) !important; border: none !important; }
[data-baseweb="notification"][kind="info"]    { background: #e8eff5 !important; border-left: 3px solid #4a7fa8 !important; }
[data-baseweb="notification"][kind="success"] { background: #e6f0e8 !important; border-left: 3px solid var(--green) !important; }
[data-baseweb="notification"][kind="warning"] { background: #f5ede0 !important; border-left: 3px solid #c08030 !important; }
[data-baseweb="notification"][kind="error"]   { background: #f5e0e0 !important; border-left: 3px solid var(--red) !important; }

[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, var(--rust), #e07040) !important; border-radius: 4px !important; }
[data-testid="stProgressBar"] > div { background: var(--border-soft) !important; border-radius: 4px !important; }

[data-testid="stExpander"] {
    background: var(--bg-card) !important; border: 1px solid var(--border-soft) !important;
    border-radius: var(--radius) !important; margin-bottom: 0.5rem !important; overflow: hidden !important;
}
[data-testid="stExpander"]:hover { border-color: var(--rust-dim) !important; }
[data-testid="stExpander"] summary {
    font-family: var(--font-mono) !important; font-size: 0.82rem !important;
    color: var(--text) !important; padding: 0.7rem 1rem !important;
}

[data-testid="stFileUploader"] {
    background: var(--bg-card) !important; border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius-lg) !important; padding: 1rem !important; transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--rust-dim) !important; }

hr { border: none !important; border-top: 1px solid var(--border-soft) !important; margin: 2rem 0 !important; }

code, pre {
    font-family: var(--font-mono) !important; background: #ece6de !important;
    color: var(--rust) !important; border-radius: 6px !important; border: 1px solid var(--border-soft) !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--brown); }

.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px;
    font-family: var(--font-mono); font-size: 0.72rem;
    letter-spacing: 0.06em; font-weight: 700; text-transform: uppercase;
}
.status-pill.online  { background: #e6f0e8; color: #2e6b3a; border: 1px solid #3d7a4a; }
.status-pill.offline { background: #f5e0e0; color: var(--red);  border: 1px solid var(--red); }
.status-pill::before { content: '●'; font-size: 0.6rem; }

.metric-card {
    background: var(--bg-card); border: 1px solid var(--border-soft);
    border-radius: var(--radius); padding: 1.2rem 1.4rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.metric-card:hover { border-color: var(--rust-dim); box-shadow: 0 2px 16px #c44a2810; }
.metric-card .label {
    font-family: var(--font-mono); font-size: 0.68rem;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-muted); margin-bottom: 0.4rem;
}
.metric-card .value { font-family: var(--font-head); font-size: 2rem; font-weight: 800; color: var(--rust); line-height: 1; }
.metric-card .sub { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.3rem; }

.section-header {
    display: flex; align-items: baseline; gap: 1rem;
    margin-bottom: 1.5rem; padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-soft);
}
.section-header .title { font-family: var(--font-head); font-size: 1.6rem; font-weight: 800; color: var(--text); }
.section-header .tag { font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-muted); letter-spacing: 0.1em; text-transform: uppercase; }

.badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-family: var(--font-mono); font-size: 0.68rem;
    font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; margin-right: 4px;
}
.badge-singular { background: #fdf0e0; color: #9a6010; border: 1px solid #c4901844; }
.badge-hinge    { background: #e8f4ea; color: #2e6b3a; border: 1px solid #3d7a4a44; }
.badge-theta    { background: #f5ede8; color: #8a3a1e; border: 1px solid #c44a2844; }
.badge-sim      { background: #f0ece8; color: #6a5040; border: 1px solid #8a6a5044; }

.hero-tagline {
    font-family: var(--font-mono); font-size: 0.78rem;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--text-muted); margin-top: -0.5rem; margin-bottom: 1rem;
}

.info-box {
    background: #faf6f0; border: 1px solid var(--border-soft);
    border-left: 3px solid var(--rust); border-radius: var(--radius);
    padding: 1rem 1.2rem; margin: 1rem 0;
    font-size: 0.88rem; color: var(--text-muted);
}

.empty-state {
    text-align: center; padding: 3rem 2rem;
    border: 1.5px dashed var(--border); border-radius: var(--radius-lg);
    color: var(--text-muted); font-family: var(--font-mono);
    font-size: 0.82rem; letter-spacing: 0.04em; background: var(--bg-card);
}
.empty-state .icon { font-size: 2.5rem; margin-bottom: 1rem; display: block; }
</style>
"""


def apply_global_styles() -> None:
    """Inject the global CSS. Safe to call multiple times — Streamlit dedupes markdown."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def badge(label: str, text: str | None = None) -> str:
    """Return an HTML span for a method badge.

    ``label`` selects the visual variant (Singular / Hinge / Theta / sim / graph).
    ``text`` overrides the displayed text — useful when reusing a variant's color
    scheme with arbitrary content (e.g. status indicators).
    """
    cls = {
        "Singular": "badge-singular",
        "Hinge":    "badge-hinge",
        "Theta":    "badge-theta",
    }.get(label, "badge-sim")
    return f'<span class="badge {cls}">{text if text is not None else label}</span>'
