"""FastAPI 서버 — 데모용 웹 UI + 채팅 API(스트리밍 포함) + 지식 패널.

실행:
    uvicorn app.main:app --reload --port 8090
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """기동 시 임베딩 모델·Chroma 를 미리 적재(워밍업).

    지연 로딩이면 첫 질문이 ~18초 걸린다(HF 모델 로드). 미리 데워두면 1~2초.
    BM25 인덱스 구축도 여기서 함께 끝난다.
    """
    from app.retriever import search

    search("warmup")
    yield


app = FastAPI(title="stock_prod RAG 챗봇", version="2.0.0", lifespan=lifespan)

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# 지식원 내용을 반영한 시작 질문(칩). 사용자가 "무엇을 물어볼 수 있는지" 감을 잡게 함.
SUGGESTED_QUESTIONS = [
    "지금 운용 모드가 뭐야?",
    "5개 포트폴리오 모드 차이를 표로 정리해줘",
    "zombie recovery 로직이 뭐야?",
    "SL 은 어떻게 결정돼? 캡은 있어?",
    "2026-06-26 OCX 스톨/드리프트 사고 원인이 뭐야?",
    "broker hard_end 시각은 어떻게 정해져?",
]


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    section: str = ""
    snippet: str = ""   # 각주 [n] 을 펼치면 보이는 원문 발췌


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    mode: str
    retrieval: dict = {}   # 검색 진단(유사도·BM25·선택된 청크) — 데모에서 근거를 보여주는 용도


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "embedding_provider": settings.embedding_provider,
        "llm_provider": settings.active_llm,
        "configured_llm": settings.llm_provider,
        "collection": settings.collection_name,
    }


@app.get("/topics")
def topics() -> dict:
    """지식원 파일 목록 + 시작 질문. 프런트의 '이 봇이 아는 것' 패널·칩에 사용."""
    from app.loader import list_sources

    files = list_sources()
    return {
        "count": len(files),
        "files": files,
        "suggestions": SUGGESTED_QUESTIONS,
        "llm": settings.active_llm,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    from app.rag import answer

    return ChatResponse(**answer(req.question.strip()))


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """토큰 단위 스트리밍. NDJSON(줄바꿈 구분 JSON) 이벤트를 흘려보냄."""
    from app.rag import stream_answer

    question = req.question.strip()

    def gen():
        for ev in stream_answer(question):
            yield json.dumps(ev, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson; charset=utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html = _WEB_DIR / "index.html"
    if html.exists():
        return html.read_text(encoding="utf-8")
    return "<h1>stock_prod RAG</h1><p>POST /chat 로 질문하세요.</p>"
