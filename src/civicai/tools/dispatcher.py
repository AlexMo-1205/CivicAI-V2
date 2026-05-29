"""Tool name -> handler dispatch."""
from __future__ import annotations

from typing import Callable

from civicai.tools.search_docs import search_docs
from civicai.tools.web_search import web_search


HANDLERS: dict[str, Callable[..., str]] = {
    "search_docs": search_docs,
    "web_search": web_search,
}


def run_tool(tool_name: str, tool_input: dict) -> str:
    handler = HANDLERS.get(tool_name)
    if handler is None:
        return f"Outil inconnu : {tool_name}"
    return handler(**tool_input)
