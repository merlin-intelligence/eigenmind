"""Reusable Streamlit components: sidebar, NLP loader, Nebius/AI Hub adapter."""
from __future__ import annotations

import os
from dataclasses import dataclass

import nltk
import requests
import spacy
import streamlit as st
import torch
from spacy.cli import download as spacy_download

from eigenmind.config import nebius_api_key as env_nebius_key
from eigenmind.core.embeddings import EmbeddingModel
from eigenmind.ui.auth import current_user, hash_password, load_user_db, save_user_db
from eigenmind.vectordb.store import QdrantStore


@dataclass
class SidebarState:
    qdrant_host: str
    qdrant_port: int
    is_connected: bool
    selected_device: str
    nebius_api_key: str
    nebius_model: str


@st.cache_resource
def load_nlp():
    """Load (and download on demand) the spaCy English model used by the graph explorer."""
    try:
        model = spacy.load("en_core_web_sm")
    except OSError:
        spacy_download("en_core_web_sm")
        model = spacy.load("en_core_web_sm")
    nltk.download("stopwords", quiet=True)
    return model


@st.cache_resource
def get_embedder(device: str) -> EmbeddingModel:
    """Process-wide cached embedding model.

    Streamlit's ``cache_resource`` keeps a single instance per (device,) key for the
    lifetime of the server process — shared across all sessions and users. Do not
    ``release()`` the returned instance; let the cache own its lifecycle.
    """
    return EmbeddingModel(device=device)


def _detect_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")
    return devices


def render_sidebar() -> SidebarState:
    """Render the shared sidebar (host/port, device, LLM model, profile) and return its state."""
    with st.sidebar:
        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
            'letter-spacing:0.15em;color:#8a6a50;text-transform:uppercase;'
            'margin-bottom:1.2rem">⬡ navigation</p>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
            'letter-spacing:0.15em;color:#8a6a50;text-transform:uppercase;'
            'margin-bottom:0.8rem">⚙ corpus settings</p>',
            unsafe_allow_html=True,
        )

        host = st.text_input("host", os.getenv("QDRANT_HOST", "localhost"))
        port = st.number_input("port", min_value=1, max_value=65535,
                               value=int(os.getenv("QDRANT_PORT", "6333")))

        is_connected = QdrantStore.is_reachable(host, port)
        cls, txt = ("online", "qdrant online") if is_connected else ("offline", "qdrant offline")
        st.markdown(f'<div class="status-pill {cls}">{txt}</div>', unsafe_allow_html=True)

        st.markdown("---")
        device = st.selectbox("embedding device", _detect_devices())

        st.markdown("---")
        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
            'letter-spacing:0.15em;color:#8a6a50;text-transform:uppercase;'
            'margin-bottom:0.8rem">🤖 llm settings</p>',
            unsafe_allow_html=True,
        )

        api_key = env_nebius_key()
        try:
            if "NEBIUS_API_KEY" in st.secrets:
                api_key = st.secrets["NEBIUS_API_KEY"]
        except Exception:
            pass

        model = st.selectbox(
            "model",
            ("meta-llama/Llama-3.3-70B-Instruct", "moonshotai/Kimi-K2.5-fast", "openai/gpt-oss-120b"),
            format_func=lambda x: x.split("/")[-1],
            help="Model used in the Chat page.",
        )
        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.6rem;'
            'color:#8a6a50;text-align:center;margin-top:-0.5rem">Powered by AI Hub</p>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        user = current_user()
        st.markdown(
            f'<p style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
            f'color:var(--rust);text-align:center;text-transform:uppercase;">● USER: {user}</p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace; font-size:0.7rem; '
            'color:var(--text-muted); text-align:center; margin-top:1rem;">PROFILE SETTINGS</p>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown(
                '<p style="font-family:\'DM Mono\',monospace; font-size:0.6rem; '
                'color:var(--text-muted); margin-bottom:0.5rem; text-align:center;">CHANGE PASSWORD</p>',
                unsafe_allow_html=True,
            )
            old_p = st.text_input("current password", type="password", key="cp_old",
                                  label_visibility="collapsed", placeholder="current password")
            new_p = st.text_input("new password", type="password", key="cp_new",
                                  label_visibility="collapsed", placeholder="new password")
            confirm_p = st.text_input("confirm new password", type="password", key="cp_confirm",
                                      label_visibility="collapsed", placeholder="confirm new password")
            if st.button("update password", use_container_width=True):
                if not old_p or not new_p or not confirm_p:
                    st.error("Fill all fields.")
                elif new_p != confirm_p:
                    st.error("New passwords don't match.")
                else:
                    db = load_user_db()
                    if db.get(user) == hash_password(old_p):
                        db[user] = hash_password(new_p)
                        save_user_db(db)
                        st.success("Password updated!")
                    else:
                        st.error("Incorrect current password.")

        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.6rem;'
            'color:#a09080;text-align:center;margin-top:1rem;">© 2025 Prax Value Eurl</p>',
            unsafe_allow_html=True,
        )

    return SidebarState(host, int(port), is_connected, device, api_key, model)


class NebiusClient:
    """Client for the Nebius / AI Hub Chat Completions endpoints.

    The endpoint is selected automatically based on the model name (TokenFactory for
    Kimi/GPT-OSS, Studio for the rest).
    """

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        if "moonshotai" in model or "openai" in model:
            self.api_url = "https://api.tokenfactory.nebius.com/v1/chat/completions"
            self.vendor = "AI Hub"
            self._user_content_is_blocks = True
        else:
            self.api_url = "https://api.studio.nebius.ai/v1/chat/completions"
            self.vendor = "AI Hub Studio"
            self._user_content_is_blocks = False

    def chat(self, system_prompt: str, user_content: str) -> str:
        """Single-turn chat completion. Raises on non-200 responses or empty content."""
        user_content_payload = (
            [{"type": "text", "text": user_content}]
            if self._user_content_is_blocks
            else user_content
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content_payload},
            ],
            "max_tokens": 2048,
            "temperature": 0.7,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"{self.vendor} API {response.status_code}: {response.text}")

        result = response.json()
        if not (result.get("choices") and result["choices"][0].get("message")):
            raise RuntimeError(f"Unexpected response format: {result}")

        msg = result["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")

        answer = ""
        if reasoning:
            answer += f"**[Reasoning Process]**\n{reasoning}\n\n---\n\n"
        if content:
            answer += content

        if not answer.strip():
            raise RuntimeError(f"Empty content in response: {result}")
        return answer


def empty_state(icon: str, message: str) -> None:
    st.markdown(
        f'<div class="empty-state"><span class="icon">{icon}</span>{message}</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, tag: str) -> None:
    st.markdown(
        f'<div class="section-header"><span class="title">{title}</span>'
        f'<span class="tag">→ {tag}</span></div>',
        unsafe_allow_html=True,
    )


def info_box(html: str) -> None:
    st.markdown(f'<div class="info-box">{html}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "", *, value_style: str = "") -> str:
    """Return the HTML for a metric card. Pair with ``st.markdown(..., unsafe_allow_html=True)``."""
    style_attr = f' style="{value_style}"' if value_style else ""
    return (
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value"{style_attr}>{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )
