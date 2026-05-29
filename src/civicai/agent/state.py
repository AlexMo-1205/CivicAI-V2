"""LangGraph state schema."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
