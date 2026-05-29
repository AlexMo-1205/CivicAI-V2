"""Pydantic request/response models for the HTTP layer."""
from __future__ import annotations

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []


class ChatResponse(BaseModel):
    answer: str
