# 🏛️ CivicAI — Thailand Administrative Assistant

Conversational assistant built on Claude (Anthropic) to help french expats navigate administrative procedures in Thailand.

## Demo

![CivicAI Interface](static/CivicAI_screenshot.png)

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Claude Sonnet (Anthropic) |
| Agent | LangGraph |
| RAG | ChromaDB + Sentence Transformers |
| Web search | Tavily API |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Docker + Docker Compose |

---

## Architecture

```
Browser (web interface)
        ↓ POST /chat
FastAPI (api.py)
        ↓
LangGraph Agent (agent.py)
        ├── search_docs  → ChromaDB (RAG on administrative docs)
        └── web_search   → Tavily (recent information)
```

### Routing Logic

1. The agent **always searches local docs** (`search_docs`) first
2. If the average similarity score is < 0.5 → falls back to `web_search`
3. Conversation history is maintained client-side and sent with each request

---

## Features

- **Hybrid RAG** — local knowledge base + web search as fallback
- **Confidence score** — automatic detection of out-of-domain questions
- **Conversation history** — Claude remembers the conversation context
- **Web interface** — chat with question suggestions
- **Deployable** — optimized multi-stage Dockerfile

---

## Local Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — package manager
- API keys: [Anthropic](https://console.anthropic.com) + [Tavily](https://tavily.com)

### Setup

```bash
git clone https://github.com/AlexMo-1205/civicai.git
cd civicai

# Install dependencies
uv sync

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Generate the vector database
uv run python ingest.py

# Start the server
uv run uvicorn api:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

---

## Docker Deployment

```bash
# Build and start
docker compose up --build

# Stop
docker compose down
```

The app is accessible at [http://localhost:8000](http://localhost:8000).

The ChromaDB vector database is persisted via a Docker volume — it is not regenerated on every restart.

---

## Project Structure

```
civicai/
├── docs/                    # Administrative documents
│   ├── visa_touriste.txt
│   ├── permis_travail.txt
│   └── carte_residence.txt
├── static/
│   └── index.html           # Web interface
├── agent.py                 # LangGraph agent
├── api.py                   # FastAPI backend
├── ingest.py                # RAG ingestion → ChromaDB
├── Dockerfile               # Multi-stage build
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Adding Documents

1. Add your `.txt` files to `docs/`
2. Re-run the ingestion:

```bash
uv run python ingest.py
# or in Docker:
docker compose up --build
```

---

## Environment Variables

Create a `.env` file from `.env.example`:

```
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

---

## Technical Choices

**Why LangGraph instead of LangChain?**
LangGraph models the agent as an explicit state graph — transitions are readable, debuggable, and extensible without modifying existing logic.

**Why ChromaDB?**
Local vector database, zero infrastructure. Perfect for an MVP — migratable to pgvector or Pinecone in production without changing the ingestion code.

**Why a confidence score?**
Without a relevance threshold, the agent can synthesize irrelevant chunks and hallucinate a confident answer. The automatic fallback to `web_search` when the score < 0.5 ensures responses are always sourced.

**Why vanilla JS for the frontend?**
Zero dependencies, zero build step. The interface is simple and statically deployable.

---

## Roadmap

- [ ] PDF support in addition to txt files
- [ ] Response streaming (WebSockets)
- [ ] GCP Cloud Run deployment
- [ ] Multilingual support (Thai, English, French)
- [ ] User authentication

---

## Author

**Alexis Monnier** — [@AlexMo-1205](https://github.com/AlexMo-1205)

ML/AI Engineer - Data Scientist | Bangkok, Thailand
