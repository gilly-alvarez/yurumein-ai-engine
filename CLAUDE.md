# Yurumein AI — Engine (AI Service)

This is the FastAPI/Python AI service for Yurumein AI, a platform dedicated to preserving and teaching the Garífuna language and culture. It provides chatbot (RAG), translation, and lesson generation capabilities.

## Hard Rules

- Do not deploy without explicit owner approval.
- Do not touch the production VPS, nginx, PM2, or Docker configuration.
- Do not expose, print, or commit secrets, tokens, or credentials.
- Do not create or commit real `.env` files (use `.env.example` for documentation only).
- Preserve current architecture.

## Ask Before Changing

- System prompts or user prompts (any file in `app/services/utils/RAG_pipeline/prompt/`)
- OpenAI model names or parameters (temperature, max_tokens, etc.)
- RAG pipeline logic (`workflow/`, `vectore_db/`, `processed_data/`)
- Vector database connection or collection configuration
- Document upload logic (`/upload_docs` endpoint)
- Authentication or access control on any endpoint
- Docker or deployment configuration (`Dockerfile`, `docker-compose.yml`)
- Any environment variable names or their usage in `app/core/config.py`

## Architecture

- Runtime: Python 3.11, FastAPI, Uvicorn (port 8085)
- Chatbot: LangGraph StateGraph → hybrid RAG (PGVector semantic + SQL keyword) → GPT-4o
- Translator: OpenAI sync client → GPT model (direct, no RAG)
- Learning Hub: LangChain ChatOpenAI → GPT model (direct, no RAG)
- Embeddings: OpenAI `text-embedding-3-small`
- Vector store: PGVector (`langchain-postgres`)
- Session persistence: PostgreSQL (SQLAlchemy async)
- Document processing: PDF/TXT/DOCX/XLS/CSV/images (OCR via pytesseract)

## Environment Variables

Copy `.env.example` to `.env` and fill in real values before running locally.
Never commit `.env` — it is protected by `.gitignore`.

## Mission Context

Yurumein AI exists to preserve Garífuna language and cultural heritage. This context is load-bearing — not a footnote.

- Do not generate, alter, or approve Garífuna vocabulary, grammar, or cultural content without RAG grounding or explicit owner review.
- The `DICCIONARIO GARIFUNA.pdf` in `data/new_docs/` is the primary cultural source. Treat it as authoritative.
- Linguistic accuracy in Garífuna is mission-critical. A wrong translation or lesson is not a minor bug — it can cause lasting cultural harm.
