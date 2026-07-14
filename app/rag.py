"""검색(Retrieval) + 근거 기반 답변 생성(RAG).

핵심 설계: **검색된 문서 context 에만 근거**해 답하고,
근거가 없으면 "문서에서 찾을 수 없다"고 답하도록 강제 → 환각 억제.
provider: gemini(무료) | openai | extractive(LLM 미사용, 발췌만).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterator, List

from langchain_core.documents import Document

from app.config import settings
from app.retriever import search, snippets_for

OUT_OF_SCOPE = (
    "제공된 문서에서 관련 내용을 찾을 수 없습니다. "
    "이 봇은 stock_prod 프로젝트 문서에 대해서만 답할 수 있습니다."
)

SYSTEM_PROMPT = (
    "당신은 'stock_prod'(KOSPI200 선물·옵션 자동매매 시스템) 프로젝트의 문서 어시스턴트입니다.\n"
    "규칙:\n"
    "1. 반드시 아래 [문서] 발췌에만 근거해 한국어로 답하세요. 없는 내용은 지어내지 말고 "
    "'제공된 문서에서 관련 내용을 찾을 수 없습니다.'라고 답하세요.\n"
    "2. 읽기 좋게 마크다운으로 구조화하세요: 핵심 결론을 한 문장으로 먼저, 그 뒤 필요하면 불릿/표로. 장황하지 않게.\n"
    "3. 근거로 쓴 문서를 해당 문장 끝에 [1], [2] 형태 각주로 표기하세요. 번호는 [문서 N]의 N과 일치시킵니다.\n"
    "4. 코드·설정값·수치는 문서에 적힌 그대로 정확히 인용하세요."
)


def _text_of(content: Any) -> str:
    """LLM 응답 content 정규화.

    Gemini 3.x 계열은 content 를 문자열이 아니라 파트 배열로 반환한다:
    `[{"type":"text","text":"..."}, {"extras":{"signature":...}}]`
    → text 파트만 이어붙이고 thinking/서명 등 비텍스트 파트는 버린다.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "".join(parts)
    return str(content) if content else ""


def _format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "?")
        sec = d.metadata.get("section", "")
        head = f"[문서 {i}] {src}" + (f" > {sec}" if sec else "")
        blocks.append(f"{head}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def _extractive_text(docs: List[Document]) -> str:
    preview = "\n\n".join(
        f"• {d.metadata.get('source','?')}"
        + (f" > {d.metadata.get('section')}" if d.metadata.get("section") else "")
        + f"\n{d.page_content.strip()[:500]}"
        for d in docs
    )
    return "**[extractive 모드 · LLM 미사용]** 관련 문서 발췌:\n\n" + preview


@lru_cache(maxsize=1)
def _llm():
    provider = settings.active_llm
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_chat_model, api_key=settings.openai_api_key, temperature=0)
    raise RuntimeError("LLM provider 가 extractive 인데 _llm() 이 호출됨")


def _messages(question: str, docs: List[Document]):
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"[문서]\n{_format_context(docs)}\n\n[질문]\n{question}"),
    ]


def answer(question: str) -> Dict[str, Any]:
    """질문 → {answer, sources, mode, retrieval} (비스트리밍)."""
    docs, debug = search(question)
    if not docs:
        return {"answer": OUT_OF_SCOPE, "sources": [], "mode": "no_hit", "retrieval": debug}

    sources = snippets_for(docs, question)  # 순서 = 프롬프트의 [문서 N] = 답변의 [n]
    provider = settings.active_llm
    if provider == "extractive":
        return {"answer": _extractive_text(docs), "sources": sources, "mode": "extractive", "retrieval": debug}

    resp = _llm().invoke(_messages(question, docs))
    return {"answer": _text_of(resp.content), "sources": sources, "mode": provider, "retrieval": debug}


def stream_answer(question: str) -> Iterator[Dict[str, Any]]:
    """질문 → 이벤트 스트림. 각 이벤트: {type: sources|token|done|error, ...}."""
    docs, debug = search(question)

    if not docs:
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": OUT_OF_SCOPE}
        yield {"type": "done", "mode": "no_hit", "retrieval": debug}
        return

    yield {"type": "sources", "sources": snippets_for(docs, question)}

    provider = settings.active_llm
    if provider == "extractive":
        yield {"type": "token", "text": _extractive_text(docs)}
        yield {"type": "done", "mode": "extractive", "retrieval": debug}
        return

    try:
        for chunk in _llm().stream(_messages(question, docs)):
            piece = _text_of(chunk.content)
            if piece:
                yield {"type": "token", "text": piece}
    except Exception as e:  # noqa: BLE001 — 사용자에게 오류를 그대로 표시
        yield {"type": "error", "message": f"LLM 호출 오류: {e}"}
        return
    yield {"type": "done", "mode": provider, "retrieval": debug}
