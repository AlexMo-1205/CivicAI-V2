"""LangGraph node functions: Claude call, tool exec, control-flow predicate."""
from __future__ import annotations

from civicai.agent.state import AgentState
from civicai.config import SETTINGS, build_system_prompt
from civicai.llm.client import get_claude
from civicai.tools.definitions import TOOLS
from civicai.tools.dispatcher import run_tool


def call_claude(state: AgentState) -> dict:
    claude = get_claude()
    response = claude.messages.create(
        model=SETTINGS.model,
        max_tokens=SETTINGS.max_tokens,
        system=build_system_prompt(),
        tools=TOOLS,
        messages=state["messages"],
    )
    return {"messages": [{"role": "assistant", "content": response.content}]}


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
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })
    return {"messages": [{"role": "user", "content": tool_results}]}
