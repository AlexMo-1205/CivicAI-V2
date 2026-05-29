"""LangGraph orchestrator: wires nodes, exposes the ask() entry point."""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, StateGraph

from civicai.agent.nodes import call_claude, run_tools, should_continue
from civicai.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("call_claude", call_claude)
    graph.add_node("run_tools", run_tools)
    graph.set_entry_point("call_claude")
    graph.add_conditional_edges(
        "call_claude",
        should_continue,
        {"run_tools": "run_tools", "end": END},
    )
    graph.add_edge("run_tools", "call_claude")
    return graph.compile()


@lru_cache(maxsize=1)
def _compiled_app():
    return build_graph()


def ask(question: str, history: list | None = None) -> str:
    """Top-level entry: run the graph for one user turn, return assistant text."""
    history = history or []
    app = _compiled_app()

    messages = history + [{"role": "user", "content": question}]
    final_state = app.invoke({"messages": messages})

    for message in reversed(final_state["messages"]):
        if message["role"] == "assistant":
            for block in message["content"]:
                if hasattr(block, "text"):
                    return block.text

    return "Je n'ai pas pu générer une réponse."
