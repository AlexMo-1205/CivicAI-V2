"""Test config: stub external services before any project imports."""
import os
import sys
from unittest.mock import MagicMock

import pytest


# Stub env so module-level client constructors don't blow up.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")


@pytest.fixture
def stub_external_modules(monkeypatch):
    """Patch chromadb / sentence_transformers / anthropic / tavily in sys.modules.

    Used by tests that need to (re)import project modules without touching real
    services, real models, or real DBs.
    """
    chromadb_mock = MagicMock()
    st_mock = MagicMock()
    anthropic_mock = MagicMock()
    tavily_mock = MagicMock()

    monkeypatch.setitem(sys.modules, "chromadb", chromadb_mock)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_mock)
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_mock)
    monkeypatch.setitem(sys.modules, "tavily", tavily_mock)

    yield {
        "chromadb": chromadb_mock,
        "sentence_transformers": st_mock,
        "anthropic": anthropic_mock,
        "tavily": tavily_mock,
    }
