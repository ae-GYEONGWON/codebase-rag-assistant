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
from app.profiles import active_profile


@asynccontextmanager
async def lifespan(app: FastAPI):
    """기동 시 임베딩 모델·Chroma 를 미리 적재(워밍업).

    지연 로딩이면 첫 질문이 ~18초 걸린다(HF 모델 로드). 미리 데워두면 1~2초.
    BM25 인덱스 구축도 여기서 함께 끝난다.
    """
    from app.retriever import search

    search("warmup")
    if settings.use_reranker:
        from app.reranker import warmup as rr_warmup

        rr_warmup()
    yield


app = FastAPI(title="Codebase RAG 어시스턴트", version="2.0.0", lifespan=lifespan)

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# 지식원 내용을 반영한 시작 질문(칩). 사용자가 "무엇을 물어볼 수 있는지" 감을 잡게 함.
# ※ 인덱싱한 코드베이스에 맞게 자유롭게 바꾸세요(문서 질문 / 코드 질문을 섞는 것을 권장).
# 프로필이 추천 질문을 갖고 있지 않을 때만 쓰는 범용 폴백.
# ★추천 질문은 코퍼스에 실제로 답이 있어야 한다 — 클릭했는데 못 찾으면 데모가 거기서 끝난다.
FALLBACK_QUESTIONS = [
    "이 프로젝트는 무엇을 하는 시스템이야?",
    "주요 컴포넌트 차이를 표로 정리해줘",
    "설정값은 어디서 바꿔?",
    "최근에 가장 크게 바뀐 부분이 뭐야?",
]


class HistoryTurn(BaseModel):
    role: str                # user | assistant
    content: str


class ChatRequest(BaseModel):
    question: str
    dev_mode: bool = False   # True 면 답변에 코드 본문을 ``` 블록으로 인용
    # 멀티턴 — 직전 대화. 후속 질문("그건 왜?")을 독립형으로 재작성해 검색한다.
    history: list[HistoryTurn] = []


class Source(BaseModel):
    source: str
    section: str = ""
    doc_type: str = "doc"   # doc | code
    snippet: str = ""       # 각주 [n] 을 펼치면 보이는 원문 발췌


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    mode: str
    retrieval: dict = {}   # 검색 진단(유사도·BM25·선택된 청크) — 데모에서 근거를 보여주는 용도
    rewrite: dict = {}     # 질의 재작성 결과(멀티턴). 무엇으로 검색했는지 UI 에 노출한다
    # 경로 선택(단발 RAG vs 에이전트)과 그 이유. ★ 여기에 필드를 안 두면 FastAPI 가
    # 응답 모델로 걸러 내어 값이 조용히 사라진다 — 실제로 그렇게 빠져 있었다.
    route: dict = {}
    trace: list = []       # 에이전트 경로일 때의 툴 호출 순서(진단용)
    llm_calls: int | None = None


class FeedbackRequest(BaseModel):
    question: str
    verdict: str           # up | down
    answer: str = ""
    mode: str = ""


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "embedding_provider": settings.embedding_provider,
        "llm_provider": settings.active_llm,
        "configured_llm": settings.llm_provider,
        "corpus_profile": active_profile().name,
        "collection": active_profile().collection_name,
    }


@app.get("/topics")
def topics() -> dict:
    """지식원 목록 + 시작 질문. 프런트의 '이 봇이 아는 것' 패널·칩에 사용."""
    from app.code_loader import list_code_sources
    from app.loader import list_sources

    files = list_sources()
    code = list_code_sources()
    return {
        "count": len(files),
        "files": files,
        "code_count": len(code),
        "suggestions": list(active_profile().suggestions) or FALLBACK_QUESTIONS,
        "llm": settings.active_llm,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    from app.rag import answer

    return ChatResponse(**answer(req.question.strip(), dev_mode=req.dev_mode,
                                 history=[t.model_dump() for t in req.history]))


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """토큰 단위 스트리밍. NDJSON(줄바꿈 구분 JSON) 이벤트를 흘려보냄."""
    from app.rag import stream_answer

    question = req.question.strip()
    dev_mode = req.dev_mode
    history = [t.model_dump() for t in req.history]

    def gen():
        for ev in stream_answer(question, dev_mode=dev_mode, history=history):
            yield json.dumps(ev, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson; charset=utf-8")


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    """답변에 대한 👍/👎. 👎 질문은 평가셋 확장 후보로 쌓인다."""
    from app.feedback import log_feedback

    log_feedback(req.model_dump())
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html = _WEB_DIR / "index.html"
    if html.exists():
        return html.read_text(encoding="utf-8")
    return "<h1>Codebase RAG</h1><p>POST /chat 로 질문하세요.</p>"
