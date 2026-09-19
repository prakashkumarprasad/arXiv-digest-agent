"""Shared test fixtures and configuration."""

import sys
import os
import socket
from unittest.mock import MagicMock

os.environ["HF_HUB_OFFLINE"] = "1"

# Create a mock sentence_transformers module before any real imports.
# sentence_transformers takes ~11s to initialize on first import.
class _MockSentenceTransformer:
    def __init__(self, model_name=None):
        self.model_name = model_name
    def encode(self, docs, show_progress_bar=False, **kw):
        import numpy as np
        return np.array([[0.0] * 384] * len(docs))

class _MockSTModule:
    SentenceTransformer = _MockSentenceTransformer

sys.modules["sentence_transformers"] = _MockSTModule()

import pytest
from agent.config import reset_settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Point all settings caches at tmp_path and reset between tests."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "sessions.sqlite"))
    monkeypatch.setenv("PDF_CACHE_DIR", str(tmp_path / "pdfs"))
    reset_settings()
    yield
    reset_settings()


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Block all non-localhost socket connections in unit tests."""
    real_connect = socket.socket.connect

    def guarded(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise RuntimeError(f"Network blocked in unit tests: {address!r}")
        return real_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.fixture(autouse=True)
def _mock_embedding_model(monkeypatch):
    """Prevent sentence-transformers model from loading in unit tests."""
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.0] * 384]
    monkeypatch.setattr("agent.services.chunker._get_embedding_model", lambda: mock_model)
    monkeypatch.setattr("agent.services.vectorstore._get_client", lambda: MagicMock())


@pytest.fixture
def sample_sections():
    """Return a list of section dicts for chunker/parser tests."""
    return [
        {"title": "Abstract", "text": "This is the abstract text about the paper. It has some content for testing.", "page_start": 0},
        {"title": "Introduction", "text": "The introduction discusses the background and motivation. More text here for testing chunk boundaries.", "page_start": 1},
        {"title": "Method", "text": "The method section describes the approach. We include enough text to trigger multiple chunks.", "page_start": 2},
    ]