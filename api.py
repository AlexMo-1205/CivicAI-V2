"""
CivicAI — API FastAPI
Expose l'agent LangGraph via HTTP.
Lance avec : uv run uvicorn api:app --reload
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agent import ask

app = FastAPI(title="CivicAI API")

# Sert les fichiers statiques (index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Modèles Pydantic ──────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str      # "user" ou "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    history:  list[Message] = []


class ChatResponse(BaseModel):
    answer: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Sert l'interface web."""
    return FileResponse("static/index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Point d'entrée principal.
    Reçoit une question + historique, retourne la réponse de l'agent.
    """
    # Convertit les messages Pydantic en dicts pour LangGraph
    history = [
        {"role": m.role, "content": m.content}
        for m in request.history
    ]

    answer = ask(request.question, history)
    return ChatResponse(answer=answer)


@app.get("/health")
def health():
    """Vérifie que l'API tourne."""
    return {"status": "ok", "service": "CivicAI"}
