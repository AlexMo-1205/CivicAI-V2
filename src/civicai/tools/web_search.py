"""Tavily web-search tool."""
from __future__ import annotations

from functools import lru_cache

from tavily import TavilyClient

from civicai.config import SETTINGS


@lru_cache(maxsize=1)
def get_tavily() -> TavilyClient:
    return TavilyClient(api_key=SETTINGS.tavily_api_key)


def web_search(query: str) -> str:
    client = get_tavily()
    results = client.search(
        query=query,
        max_results=SETTINGS.tavily_max_results,
        include_raw_content=False,
    )
    formatted = [
        f"**{r['title']}**\nURL: {r['url']}\n{r['content']}"
        for r in results.get("results", [])
    ]
    return "\n\n---\n\n".join(formatted) if formatted else "Aucun résultat trouvé."
