"""Unit tests for the pluggable Chat-page LLM backend (Nebius cloud / local Ollama)."""
from __future__ import annotations

from eigenmind import config
from eigenmind.ui.components import (
    NebiusClient,
    OllamaClient,
    SidebarState,
    build_llm_client,
)


def _state(**overrides) -> SidebarState:
    base = {
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "is_connected": True,
        "selected_device": "cpu",
        "llm_provider": "nebius",
        "llm_model": "meta-llama/Llama-3.3-70B-Instruct",
        "nebius_api_key": "key",
        "ollama_host": "http://localhost:11434",
        "llm_ready": True,
    }
    base.update(overrides)
    return SidebarState(**base)


# ── config.llm_provider ──

def test_llm_provider_defaults_to_nebius(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert config.llm_provider() == "nebius"


def test_llm_provider_normalises_case_and_whitespace(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "  Ollama  ")
    assert config.llm_provider() == "ollama"


def test_llm_provider_empty_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "")
    assert config.llm_provider() == config.DEFAULT_LLM_PROVIDER


# ── config.ollama_host / ollama_models ──

def test_ollama_host_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert config.ollama_host() == config.DEFAULT_OLLAMA_HOST


def test_ollama_host_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")
    assert config.ollama_host() == "http://gpu-box:11434"


def test_ollama_models_parsing(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODELS", " qwen2.5:7b , llama3.1 ,, ")
    assert config.ollama_models() == ["qwen2.5:7b", "llama3.1"]


def test_ollama_models_empty(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    assert config.ollama_models() == []


# ── OllamaClient ──

def test_ollama_client_builds_native_chat_url():
    client = OllamaClient(model="qwen2.5:7b", host="http://localhost:11434/")
    assert client.api_url == "http://localhost:11434/api/chat"
    assert client.model == "qwen2.5:7b"


# ── build_llm_client factory ──

def test_build_llm_client_selects_ollama():
    client = build_llm_client(_state(llm_provider="ollama", llm_model="qwen2.5:7b"))
    assert isinstance(client, OllamaClient)
    assert client.model == "qwen2.5:7b"


def test_build_llm_client_defaults_to_nebius():
    client = build_llm_client(_state(llm_provider="nebius"))
    assert isinstance(client, NebiusClient)
