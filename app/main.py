"""FastAPI 서버 — 데모용 웹 UI + /chat API.

실행:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.config import settings

app = FastAPI(title="stock_prod RAG 챗봇", version="1.0.0")

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    section: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    mode: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "embedding_provider": settings.embedding_provider,
        "llm_provider": settings.llm_provider if settings.has_openai else "extractive(키없음)",
        "collection": settings.collection_name,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # rag 모듈은 최초 호출 시 벡터스토어를 로드하므로 지연 임포트
    from app.rag import answer

    result = answer(req.question.strip())
    return ChatResponse(**result)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html = _WEB_DIR / "index.html"
    if html.exists():
        return html.read_text(encoding="utf-8")
    return "<h1>stock_prod RAG</h1><p>POST /chat 로 질문하세요.</p>"
