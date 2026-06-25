"""Reusable Streamlit components: sidebar, NLP loader, Nebius/AI Hub adapter."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

import nltk
import requests
import spacy
import streamlit as st
import torch
from spacy.cli import download as spacy_download

from eigenmind.config import (
    NEBIUS_MODELS,
    llm_provider,
    ollama_host,
    ollama_models,
)
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
    llm_provider: str
    llm_model: str
    nebius_api_key: str
    ollama_host: str
    llm_ready: bool


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


def list_ollama_models(host: str) -> list[str]:
    """Return the models installed on a local Ollama server.

    Queries ``GET /api/tags``. Returns an empty list if the server is unreachable
    (which the caller treats as "Ollama offline").
    """
    try:
        resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=2)
        resp.raise_for_status()
        models = sorted(m["name"] for m in resp.json().get("models", []))
        logger.debug("Ollama models at %s: %s", host, models)
        return models
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning("Ollama unreachable at %s: %s", host, e)
        return []


def _render_llm_settings() -> tuple[str, str, str, str, bool]:
    """Render the 'llm settings' sidebar block for the active provider.

    Returns ``(provider, model, nebius_api_key, ollama_host_url, llm_ready)``.
    """
    st.markdown(
        '<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
        'letter-spacing:0.15em;color:#8a6a50;text-transform:uppercase;'
        'margin-bottom:0.8rem">🤖 llm settings</p>',
        unsafe_allow_html=True,
    )

    providers = ["nebius", "ollama"]
    labels = {"nebius": "Nebius (cloud)", "ollama": "Ollama (local)"}
    default_provider = llm_provider()
    provider = st.radio(
        "backend",
        providers,
        index=providers.index(default_provider) if default_provider in providers else 0,
        format_func=lambda p: labels[p],
        horizontal=True,
        help="Where /ask/ generates answers. LLM_PROVIDER sets the initial choice.",
    )

    if provider == "ollama":
        host_url = ollama_host()
        installed = list_ollama_models(host_url)
        cls, txt = ("online", "ollama online") if installed else ("offline", "ollama offline")
        st.markdown(f'<div class="status-pill {cls}">{txt}</div>', unsafe_allow_html=True)

        options = installed or ollama_models()
        if options:
            model = st.selectbox(
                "model", options,
                help="Local Ollama model used in the Chat page.",
            )
        else:
            model = ""
            st.caption("No models found. Pull one first, e.g. `ollama pull qwen2.5:7b`.")
        ready = bool(installed) and bool(model)
        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.6rem;'
            'color:#8a6a50;text-align:center;margin-top:-0.5rem">Powered by Ollama (local)</p>',
            unsafe_allow_html=True,
        )
        return provider, model, "", host_url, ready

    # Default provider: Nebius / AI Hub cloud
    api_key = env_nebius_key()
    try:
        if "NEBIUS_API_KEY" in st.secrets:
            api_key = st.secrets["NEBIUS_API_KEY"]
    except Exception:
        pass

    model = st.selectbox(
        "model",
        NEBIUS_MODELS,
        format_func=lambda x: x.split("/")[-1],
        help="Model used in the Chat page.",
    )
    st.markdown(
        '<p style="font-family:\'DM Mono\',monospace;font-size:0.6rem;'
        'color:#8a6a50;text-align:center;margin-top:-0.5rem">Powered by AI Hub</p>',
        unsafe_allow_html=True,
    )
    return provider, model, api_key, ollama_host(), bool(api_key)


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
        provider, model, api_key, ollama_host_url, llm_ready = _render_llm_settings()

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
                    if db["users"].get(user) == hash_password(old_p):
                        db["users"][user] = hash_password(new_p)
                        save_user_db(db)
                        st.success("Password updated!")
                    else:
                        st.error("Incorrect current password.")

        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.6rem;'
            'color:#a09080;text-align:center;margin-top:1rem;">© 2025 Prax Value Eurl</p>',
            unsafe_allow_html=True,
        )

    return SidebarState(
        qdrant_host=host,
        qdrant_port=int(port),
        is_connected=is_connected,
        selected_device=device,
        llm_provider=provider,
        llm_model=model,
        nebius_api_key=api_key,
        ollama_host=ollama_host_url,
        llm_ready=llm_ready,
    )


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
        logger.debug("NebiusClient: vendor=%s model=%s", self.vendor, self.model)

    def chat(self, system_prompt: str, user_content: str) -> str:
        """Single-turn chat completion. Raises on non-200 responses or empty content."""
        logger.info("Nebius chat: model=%s prompt=%r", self.model, user_content[:80])
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
            logger.error("Nebius API error %s: %s", response.status_code, response.text[:200])
            raise RuntimeError(f"{self.vendor} API {response.status_code}: {response.text}")

        result = response.json()
        if not (result.get("choices") and result["choices"][0].get("message")):
            logger.error("Nebius unexpected response format: %s", result)
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
            logger.error("Nebius empty content in response: %s", result)
            raise RuntimeError(f"Empty content in response: {result}")
        logger.debug("Nebius chat done: answer_len=%d", len(answer))
        return answer


class OllamaClient:
    """Client for a local Ollama server's native chat endpoint (``/api/chat``).

    Exposes the same ``chat(system_prompt, user_content)`` interface as
    :class:`NebiusClient`, so the Chat page stays provider-agnostic. No API key
    is required — generation happens entirely on the local machine.
    """

    def __init__(self, model: str, host: str):
        self.model = model
        self.host = host.rstrip("/")
        self.api_url = f"{self.host}/api/chat"
        self.vendor = "Ollama"
        logger.debug("OllamaClient: host=%s model=%s", self.host, self.model)

    def chat(self, system_prompt: str, user_content: str) -> str:
        """Single-turn chat completion. Raises on transport/HTTP errors or empty content."""
        logger.info("Ollama chat: model=%s prompt=%r", self.model, user_content[:80])
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=600)
        except requests.RequestException as e:
            logger.error("Cannot reach Ollama at %s: %s", self.host, e)
            raise RuntimeError(f"Cannot reach Ollama at {self.host}: {e}") from e
        if response.status_code != 200:
            logger.error("Ollama API error %s: %s", response.status_code, response.text[:200])
            raise RuntimeError(f"{self.vendor} API {response.status_code}: {response.text}")

        result = response.json()
        content = (result.get("message") or {}).get("content", "") or ""
        if not content.strip():
            logger.error("Ollama empty content in response: %s", result)
            raise RuntimeError(f"Empty content in response: {result}")
        logger.debug("Ollama chat done: model=%s answer_len=%d", self.model, len(content))
        return content


def build_llm_client(state: SidebarState):
    """Construct the chat client for the provider selected in the sidebar."""
    logger.info("Building LLM client: provider=%s model=%s", state.llm_provider, state.llm_model)
    if state.llm_provider == "ollama":
        return OllamaClient(model=state.llm_model, host=state.ollama_host)
    return NebiusClient(model=state.llm_model, api_key=state.nebius_api_key)


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
