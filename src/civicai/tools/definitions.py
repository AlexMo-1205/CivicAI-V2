"""Anthropic tool JSON schemas.

Kept in French to match the original behavior — these descriptions influence
Claude's routing decisions and must remain verbatim.
"""
from __future__ import annotations

TOOLS = [
    {
        "name": "search_docs",
        "description": (
            "Recherche dans la base de connaissance CivicAI sur les démarches "
            "administratives en Thaïlande (visas, permis de travail, résidence). "
            "Utilise cet outil EN PREMIER pour toute question administrative."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La question ou le concept à rechercher",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Nombre de résultats (défaut: 5)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Recherche des informations récentes sur le web. "
            "Utilise cet outil uniquement si search_docs ne trouve rien de pertinent, "
            "ou pour des informations très récentes (changements de loi, actualités)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La requête de recherche",
                },
            },
            "required": ["query"],
        },
    },
]
