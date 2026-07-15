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
    "당신은 'stock_prod'(KOSPI200 선물·옵션 자동매매 시스템) 프로젝트의 어시스턴트입니다.\n"
    "[근거] 에는 **프로젝트 문서(.md)** 와 **실제 소스코드(.py)** 발췌가 섞여 들어옵니다.\n"
    "코드 발췌는 `# 파일: … / # 심볼: …` 헤더로 시작합니다.\n"
    "규칙:\n"
    "1. 반드시 아래 [근거] 에만 기반해 한국어로 답하세요. 없는 내용은 지어내지 말고 "
    "'제공된 문서에서 관련 내용을 찾을 수 없습니다.'라고 답하세요.\n"
    "2. 읽는 사람은 개발자가 아닐 수 있습니다. **핵심 결론을 평이한 한 문장으로 먼저** 쓰고, "
    "그 뒤에 필요하면 불릿/표로 풀어 쓰세요. 장황하지 않게.\n"
    "3. 근거로 쓴 발췌를 문장 끝에 [1], [2] 형태 각주로 표기하세요. 번호는 [근거 N]의 N과 일치시킵니다.\n"
    "4. 코드가 근거일 때는 파일·함수명을 밝히세요(예: `app/…/vix_strategy.py` 의 `apply_vix_sl`).\n"
    "5. **문서와 코드가 어긋나면 그 사실을 명시**하세요(예: 문서에는 남아 있으나 코드에서는 제거됨). "
    "실제 동작은 코드가 기준입니다.\n"
    "6. 설정값·수치는 근거에 적힌 그대로 정확히 인용하세요."
)

# 독자 층에 따른 코드 인용 방침. 개발자 모드 토글이 답변 형태를 실제로 바꾼다.
_DEV_ON = (
    "\n\n[개발자 모드 ON] 질문이 구현·코드에 관한 것이고 근거에 관련 함수/메서드 본문이 있으면, "
    "그 코드를 ```python 코드블록으로 인용한 뒤 핵심 라인을 짚어 설명하세요. "
    "근거에 코드 본문이 없으면 없다고 밝히고 파일·심볼 위치만 안내하세요."
)
_DEV_OFF = (
    "\n\n[개발자 모드 OFF] 독자는 비개발자입니다. 코드 본문(``` 블록)은 넣지 말고, "
    "그 코드가 무엇을 하는지 일상어로 설명하세요. 함수명·파일명은 언급해도 되지만 코드 자체는 보이지 마세요."
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
        kind = "코드" if d.metadata.get("doc_type") == "code" else "문서"
        head = f"[근거 {i}] ({kind}) {src}" + (f" > {sec}" if sec else "")
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


def _messages(question: str, docs: List[Document], dev_mode: bool = False):
    from langchain_core.messages import HumanMessage, SystemMessage

    system = SYSTEM_PROMPT + (_DEV_ON if dev_mode else _DEV_OFF)
    return [
        SystemMessage(content=system),
        HumanMessage(content=f"[근거]\n{_format_context(docs)}\n\n[질문]\n{question}"),
    ]


def answer(question: str, dev_mode: bool = False) -> Dict[str, Any]:
    """질문 → {answer, sources, mode, retrieval} (비스트리밍)."""
    docs, debug = search(question)
    if not docs:
        return {"answer": OUT_OF_SCOPE, "sources": [], "mode": "no_hit", "retrieval": debug}

    sources = snippets_for(docs, question)  # 순서 = 프롬프트의 [문서 N] = 답변의 [n]
    provider = settings.active_llm
    if provider == "extractive":
        return {"answer": _extractive_text(docs), "sources": sources, "mode": "extractive", "retrieval": debug}

    resp = _llm().invoke(_messages(question, docs, dev_mode))
    return {"answer": _text_of(resp.content), "sources": sources, "mode": provider, "retrieval": debug}


def stream_answer(question: str, dev_mode: bool = False) -> Iterator[Dict[str, Any]]:
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
        for chunk in _llm().stream(_messages(question, docs, dev_mode)):
            piece = _text_of(chunk.content)
            if piece:
                yield {"type": "token", "text": piece}
    except Exception as e:  # noqa: BLE001 — 사용자에게 오류를 그대로 표시
        yield {"type": "error", "message": f"LLM 호출 오류: {e}"}
        return
    yield {"type": "done", "mode": provider, "retrieval": debug}
