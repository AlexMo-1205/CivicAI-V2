"""API smoke test: /chat returns the right schema and the fallback path fires."""
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from civicai.agent import graph as graph_mod
from civicai.agent import nodes as nodes_mod
from civicai.api.app import app
from civicai.tools import search_docs as sd_mod
from civicai.tools import web_search as ws_mod


class _Block:
    def __init__(self, type, text=None, name=None, input=None, id=None):
        self.type = type
        if text is not None:
            self.text = text
        if name is not None:
            self.name = name
        if input is not None:
            self.input = input
        if id is not None:
            self.id = id


class _Resp:
    def __init__(self, content):
        self.content = content


def test_chat_triggers_web_search_below_threshold(monkeypatch):
    # Local RAG returns low scores -> fallback message is returned to Claude
    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "documents": [["doc"]],
        "metadatas": [[{"source": "s.txt"}]],
        "distances": [[0.9]],  # score 0.1, avg 0.1 (< 0.5)
    }
    monkeypatch.setattr(sd_mod, "get_collection", lambda: fake_collection)
    embedder = MagicMock()
    embedder.encode.return_value.tolist.return_value = [0.0]
    monkeypatch.setattr(sd_mod, "get_embedder", lambda: embedder)

    fake_tavily = MagicMock()
    fake_tavily.search.return_value = {
        "results": [{"title": "Hit", "url": "https://t", "content": "info"}],
    }
    monkeypatch.setattr(ws_mod, "get_tavily", lambda: fake_tavily)

    # Scripted Claude turns: search_docs -> web_search -> final text
    turns = [
        _Resp([_Block("tool_use", name="search_docs",
                       input={"query": "visa"}, id="t1")]),
        _Resp([_Block("tool_use", name="web_search",
                       input={"query": "thailand visa"}, id="t2")]),
        _Resp([_Block("text", text="Final grounded answer.")]),
    ]
    fake_claude = MagicMock()
    fake_claude.messages.create.side_effect = turns
    monkeypatch.setattr(nodes_mod, "get_claude", lambda: fake_claude)

    # Make sure we re-compile the graph against the patched node closures
    graph_mod._compiled_app.cache_clear()

    client = TestClient(app)
    resp = client.post("/chat", json={"question": "I need a Thai visa"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) == {"answer"}
    assert body["answer"] == "Final grounded answer."

    fake_tavily.search.assert_called_once()
    assert fake_claude.messages.create.call_count == 3


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "CivicAI"}
