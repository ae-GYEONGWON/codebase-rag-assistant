"""검색(Retrieval) + 근거 기반 답변 생성(RAG).

핵심 설계: **검색된 문서 context 에만 근거**해 답하고,
근거가 없으면 "문서에서 찾을 수 없다"고 답하도록 강제 → 환각 억제.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

from langchain_core.documents import Document

from app.config import settings
from app.ingest import get_vectorstore

SYSTEM_PROMPT = (
    "당신은 'stock_prod'(KOSPI200 선물·옵션 자동매매 시스템) 프로젝트의 문서 도우미입니다. "
    "반드시 아래에 주어진 [문서] 발췌 내용에만 근거해서 한국어로 답하세요.\n"
    "- 문서에 없는 내용은 절대 지어내지 말고 '제공된 문서에서 관련 내용을 찾을 수 없습니다.'라고 답하세요.\n"
    "- 가능하면 답변 끝에 근거가 된 파일명을 (출처: 파일 > 섹션) 형태로 표기하세요.\n"
    "- 코드·설정값·수치는 문서에 적힌 그대로 정확히 인용하세요."
)


def _format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "?")
        sec = d.metadata.get("section", "")
        head = f"[문서 {i}] {src}" + (f" > {sec}" if sec else "")
        blocks.append(f"{head}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def _sources(docs: List[Document]) -> List[Dict[str, str]]:
    seen, out = set(), []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("section"))
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": d.metadata.get("source", "?"), "section": d.metadata.get("section", "")})
    return out


@lru_cache(maxsize=1)
def _retriever():
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": settings.retrieval_k})


@lru_cache(maxsize=1)
def _llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


def answer(question: str) -> Dict[str, Any]:
    """질문 → {answer, sources, mode}."""
    docs = _retriever().invoke(question)

    if not docs:
        return {
            "answer": "제공된 문서에서 관련 내용을 찾을 수 없습니다.",
            "sources": [],
            "mode": "no_hit",
        }

    context = _format_context(docs)

    # LLM 키가 없거나 extractive 모드면 → 검색된 발췌만 반환 (키 없이 데모 가능)
    if settings.llm_provider.lower() == "extractive" or not settings.has_openai:
        preview = "\n\n".join(
            f"• {d.metadata.get('source','?')}"
            + (f" > {d.metadata.get('section')}" if d.metadata.get("section") else "")
            + f"\n{d.page_content.strip()[:500]}"
            for d in docs
        )
        return {
            "answer": "[extractive 모드 · LLM 미사용] 관련 문서 발췌:\n\n" + preview,
            "sources": _sources(docs),
            "mode": "extractive",
        }

    from langchain_core.messages import HumanMessage, SystemMessage

    resp = _llm().invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"[문서]\n{context}\n\n[질문]\n{question}"),
        ]
    )
    return {
        "answer": resp.content,
        "sources": _sources(docs),
        "mode": "openai",
    }
