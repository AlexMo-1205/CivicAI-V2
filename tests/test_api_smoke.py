"""API smoke test: /chat returns correct schema and fallback path triggers Tavily."""
import importlib
from unittest.mock import MagicMock

from fastapi.testclient import TestClient


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


def _fresh_modules():
    for name in ("api", "agent"):
        if name in importlib.sys.modules:
            del importlib.sys.modules[name]
    agent = importlib.import_module("agent")
    api = importlib.import_module("api")
    return agent, api


def test_chat_endpoint_triggers_web_search_below_threshold(stub_external_modules, monkeypatch):
    agent, api = _fresh_modules()

    # Vector store returns low scores -> fallback path
    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "documents": [["doc"]],
        "metadatas": [[{"source": "s.txt"}]],
        "distances": [[0.9]],  # score 0.1 -> avg 0.1 < 0.5
    }
    monkeypatch.setattr(agent, "collection", fake_collection)

    fake_embedder = MagicMock()
    fake_embedder.encode.return_value.tolist.return_value = [0.0]
    monkeypatch.setattr(agent, "embedder", fake_embedder)

    # Tavily fallback
    fake_tavily = MagicMock()
    fake_tavily.search.return_value = {
        "results": [{"title": "Web Hit", "url": "https://t", "content": "info"}]
    }
    monkeypatch.setattr(agent, "tavily", fake_tavily)

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
    monkeypatch.setattr(agent, "claude", fake_claude)

    client = TestClient(api.app)
    resp = client.post("/chat", json={"question": "I need a Thai visa"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) == {"answer"}
    assert body["answer"] == "Final grounded answer."

    # The fallback actually fired
    fake_tavily.search.assert_called_once()
    # Claude was driven for all three scripted turns
    assert fake_claude.messages.create.call_count == 3


def test_health_endpoint(stub_external_modules):
    _, api = _fresh_modules()
    client = TestClient(api.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "CivicAI"}
