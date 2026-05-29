"""
CivicAI — Agent LangGraph
Agent spécialisé dans les démarches administratives en Thaïlande.
"""

import os
from datetime import date
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv

load_dotenv()

import anthropic
from langgraph.graph import StateGraph, END
import chromadb
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient

# ── Clients ───────────────────────────────────────────────────────────────────
claude   = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
tavily   = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")
db       = chromadb.PersistentClient(path="chroma_db")
collection = db.get_collection("civicai")

MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = f"""Tu es CivicAI, un assistant spécialisé dans les démarches administratives en Thaïlande.
La date d'aujourd'hui est {date.today()}.

Tu aides les citoyens et expatriés à comprendre :
- Les visas et conditions de séjour
- Les permis de travail
- La résidence permanente
- Toute démarche administrative en Thaïlande

Règles :
- Utilise TOUJOURS search_docs en premier pour chercher dans ta base de connaissance
- Si search_docs ne trouve rien de pertinent (score < 0.5), utilise web_search
- Réponds de manière claire, structurée et bienveillante
- Cite toujours tes sources
- Si tu n'es pas sûr, dis-le et recommande de consulter un professionnel
- Réponds dans la langue de l'utilisateur (français ou anglais)
"""

# ── Outils ────────────────────────────────────────────────────────────────────
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
                    "description": "La question ou le concept à rechercher"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Nombre de résultats (défaut: 5)"
                }
            },
            "required": ["query"]
        }
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
                    "description": "La requête de recherche"
                }
            },
            "required": ["query"]
        }
    }
]


# ── Exécution des outils ──────────────────────────────────────────────────────
def run_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "search_docs":
        query     = tool_input["query"]
        n_results = tool_input.get("n_results", 5)

        query_embedding = embedder.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        formatted    = []
        total_score  = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            score = round(1 - dist, 3)
            total_score.append(score)
            formatted.append(f"[Source: {meta['source']} | Score: {score}]\n{doc}")

        average_score = sum(total_score) / len(total_score)

        if average_score < 0.5:
            return (
                f"Aucun document pertinent trouvé (score moyen: {average_score}). "
                "Utilise web_search pour répondre à cette question."
            )

        return "\n\n---\n\n".join(formatted)

    elif tool_name == "web_search":
        results = tavily.search(
            query=tool_input["query"],
            max_results=5,
            include_raw_content=False
        )
        formatted = []
        for r in results.get("results", []):
            formatted.append(f"**{r['title']}**\nURL: {r['url']}\n{r['content']}")
        return "\n\n---\n\n".join(formatted) if formatted else "Aucun résultat trouvé."

    return f"Outil inconnu : {tool_name}"


# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


# ── Nœuds ─────────────────────────────────────────────────────────────────────
def call_claude(state: AgentState) -> dict:
    response = claude.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=state["messages"]
    )
    return {
        "messages": [{"role": "assistant", "content": response.content}]
    }


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    for block in last_message["content"]:
        if hasattr(block, "type") and block.type == "tool_use":
            return "run_tools"
    return "end"


def run_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_results = []
    for block in last_message["content"]:
        if hasattr(block, "type") and block.type == "tool_use":
            result = run_tool(block.name, block.input)
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result
            })
    return {
        "messages": [{"role": "user", "content": tool_results}]
    }


# ── Graphe ────────────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("call_claude", call_claude)
    graph.add_node("run_tools",   run_tools)
    graph.set_entry_point("call_claude")
    graph.add_conditional_edges(
        "call_claude",
        should_continue,
        {"run_tools": "run_tools", "end": END}
    )
    graph.add_edge("run_tools", "call_claude")
    return graph.compile()


# ── Fonction principale (appelée par l'API) ───────────────────────────────────
def ask(question: str, history: list = []) -> str:
    """
    Point d'entrée principal.
    history : liste de messages précédents pour le contexte conversationnel
    """
    app = build_graph()

    messages = history + [{"role": "user", "content": question}]
    final_state = app.invoke({"messages": messages})

    for message in reversed(final_state["messages"]):
        if message["role"] == "assistant":
            for block in message["content"]:
                if hasattr(block, "text"):
                    return block.text

    return "Je n'ai pas pu générer une réponse."


# ── CLI (pour tester sans interface web) ──────────────────────────────────────
if __name__ == "__main__":
    print("🏛️  CivicAI — Assistant administratif Thaïlande")
    print("Tape 'quit' pour quitter.\n")

    history = []
    while True:
        question = input("Vous > ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        answer = ask(question, history)
        print(f"\nCivicAI > {answer}\n")

        # Garde l'historique pour le contexte conversationnel
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": answer})
