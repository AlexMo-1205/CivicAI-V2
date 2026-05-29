"""HTTP routes: /, /chat, /health."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from civicai.agent.graph import ask
from civicai.api.schemas import ChatRequest, ChatResponse
from civicai.config import SETTINGS


router = APIRouter()


@router.get("/")
def root():
    """Serve the static chat UI."""
    return FileResponse(str(SETTINGS.static_index))


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in request.history]
    answer = ask(request.question, history)
    return ChatResponse(answer=answer)


@router.get("/health")
def health():
    return {"status": "ok", "service": "CivicAI"}
