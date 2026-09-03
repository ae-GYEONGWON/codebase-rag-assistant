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
    "이 봇은 인덱싱된 프로젝트의 문서·코드에 대해서만 답할 수 있습니다."
)

SYSTEM_PROMPT = (
    "당신은 인덱싱된 소프트웨어 프로젝트(문서·소스코드·git 이력)의 어시스턴트입니다.\n"
    "[근거] 에는 **프로젝트 문서(.md)** 와 **실제 소스코드(.py)** 발췌가 섞여 들어옵니다.\n"
    "코드 발췌는 `# 파일: … / # 심볼: …` 헤더로 시작합니다.\n"
    "규칙:\n"
    "1. 반드시 아래 [근거] 에만 기반해 한국어로 답하세요. 없는 내용은 지어내지 말고 "
    "'제공된 문서에서 관련 내용을 찾을 수 없습니다.'라고 답하세요.\n"
    "2. 읽는 사람은 개발자가 아닐 수 있습니다. **핵심 결론을 평이한 한 문장으로 먼저** 쓰고, "
    "그 뒤에 필요하면 불릿/표로 풀어 쓰세요. 장황하지 않게.\n"
    "3. 근거로 쓴 발췌를 문장 끝에 [1], [2] 형태 각주로 표기하세요. 번호는 [근거 N]의 N과 일치시킵니다.\n"
    "4. 코드가 근거일 때는 파일·함수명을 밝히세요(예: `app/…/retry_strategy.py` 의 `apply_retry_policy`).\n"
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


def _messages(question: str, docs: List[Document], dev_mode: bool = False, turns=None):
    """생성용 메시지. **검색은 재작성된 질의로, 생성은 원문 대화로** 한다.

    검색기는 지시대명사를 못 풀지만 생성 모델은 앞 대화를 주면 자연스럽게 답한다 —
    둘의 요구가 달라 입력을 나눈다(app/conversation.py).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    system = SYSTEM_PROMPT + (_DEV_ON if dev_mode else _DEV_OFF)
    msgs = [SystemMessage(content=system)]
    if turns:
        from app.conversation import to_messages

        note = ("[대화 맥락] 아래에 이전 대화가 이어집니다. 후속 질문의 지시대명사는 "
                "그 맥락으로 해석하되, **답변 근거는 반드시 이번 [근거] 안에서만** 찾으세요.")
        msgs[0] = SystemMessage(content=system + chr(10) + chr(10) + note)
        msgs += to_messages(turns)
    msgs.append(HumanMessage(content=f"[근거]\n{_format_context(docs)}\n\n[질문]\n{question}"))
    return msgs

# 검색 범위를 사람이 직접 고를 때 쓰는 표. UI 의 세그먼트 값 → 검색 축.
#
# 왜 수동 선택을 두는가: 축 판별은 규칙 기반이라(app/intent.py) 표지어가 없는 질문에서는
# 전체를 뒤진다. 대개 맞지만 틀렸을 때 사용자가 손쓸 방법이 없었다 — "코드에서만 찾아줘"
# 를 자연어로 부탁해도 그건 다시 같은 판별기를 통과한다. 스위치를 주면 판별을 우회한다.
SCOPE_AXES: Dict[str, tuple] = {
    "doc": ("doc",),
    "code": ("code",),
    "commit": ("commit",),
}

SCOPE_LABEL = {"doc": "문서", "code": "코드", "commit": "커밋 이력"}


def _axis_of(scope: str | None) -> tuple | None:
    """`scope` → search() 의 doc_types. 'auto'·None·모르는 값은 전체 검색."""
    return SCOPE_AXES.get(scope or "")


def _scoped_route(chosen, scope: str | None):
    """범위를 고정하면 경로도 단발로 굳힌다.

    에이전트는 축을 **스스로 나눠** 여러 번 검색하는 경로다. 사용자가 축을 하나로 못
    박았는데 에이전트를 태우면, 축을 나누라고 만든 도구에 나눌 축이 하나뿐인 상태가 된다
    — LLM 3~5회를 쓰고 단발과 같은 답을 낸다. 범위 지정은 곧 단발 지정이다.
    """
    if not _axis_of(scope):
        return chosen
    from app.router import Route

    return Route("single",
                 f"검색 범위를 '{SCOPE_LABEL[scope]}' 하나로 고정함 — 축이 하나라 단발 검색")


def _no_hit_text(scope: str | None) -> str:
    """범위를 좁혀 못 찾은 것과 코퍼스에 아예 없는 것은 다른 사실이다.

    좁혀서 못 찾았는데 "아는 범위 밖"이라고만 하면, 사용자는 범위를 넓히면 답이 있다는
    것을 모른 채 질문을 포기한다.
    """
    if _axis_of(scope):
        return (f"'{SCOPE_LABEL[scope]}' 범위에서는 근거를 찾지 못했습니다. "
                "검색 범위를 '자동' 으로 두고 다시 물어보세요.")
    return OUT_OF_SCOPE


def _route_info(route) -> Dict[str, Any]:
    """경로 선택을 응답에 실어 보낸다 — 왜 느렸는지/빨랐는지가 화면에서 보여야 한다."""
    return {"mode": route.mode, "reason": route.reason, "axes": list(route.axes)}


def answer(question: str, dev_mode: bool = False, history=None,
           route: str | None = None, scope: str | None = None) -> Dict[str, Any]:
    """질문 → {answer, sources, mode, retrieval, rewrite, route} (비스트리밍).

    `route` 로 경로를 강제할 수 있다("single" | "agent"). 지정하지 않으면 라우터가
    질문을 보고 고른다(app/router.py) — 축을 둘 이상 물으면 에이전트, 아니면 단발.
    """
    from app.conversation import parse_history, rewrite_query
    from app.router import decide

    turns = parse_history(history)
    query, rw = rewrite_query(question, turns)

    # 재작성된 질의로 판단한다 — 후속 질문("그건 왜 바뀌었어?")은 원문만 보면
    # 축이 하나로 보이지만, 재작성하면 무엇을 묻는지가 드러난다.
    chosen = _scoped_route(decide(query, force=route), scope)
    if chosen.uses_agent:
        from app.agent import answer as agent_answer

        res = agent_answer(query, dev_mode)
        res.setdefault("retrieval", {})
        res["rewrite"] = rw
        res["route"] = {"mode": chosen.mode, "reason": chosen.reason,
                        "axes": list(chosen.axes)}
        return res

    docs, debug = search(query, doc_types=_axis_of(scope))
    if not docs:
        return {"answer": _no_hit_text(scope), "sources": [], "mode": "no_hit",
                "retrieval": debug, "rewrite": rw, "route": _route_info(chosen)}

    sources = snippets_for(docs, query)  # 순서 = 프롬프트의 [문서 N] = 답변의 [n]
    provider = settings.active_llm
    if provider == "extractive":
        return {"answer": _extractive_text(docs), "sources": sources, "mode": "extractive",
                "retrieval": debug, "rewrite": rw, "route": _route_info(chosen)}

    resp = _llm().invoke(_messages(question, docs, dev_mode, turns))
    return {"answer": _text_of(resp.content), "sources": sources, "mode": provider,
            "retrieval": debug, "rewrite": rw, "route": _route_info(chosen)}


def stream_answer(question: str, dev_mode: bool = False, history=None,
                  route: str | None = None,
                  scope: str | None = None) -> Iterator[Dict[str, Any]]:
    """질문 → 이벤트 스트림. 각 이벤트: {type: rewrite|sources|token|done|error, ...}."""
    from app.conversation import parse_history, rewrite_query

    from app.router import decide

    turns = parse_history(history)
    query, rw = rewrite_query(question, turns)
    # 무엇으로 검색했는지 먼저 알린다 — 답이 나오기 전에 보여야 사용자가 맥락을 잡는다.
    yield {"type": "rewrite", **rw}

    chosen = _scoped_route(decide(query, force=route), scope)
    yield {"type": "route", **_route_info(chosen)}
    if chosen.uses_agent:
        # 에이전트는 툴 호출 루프라 토큰 단위 스트리밍이 없다. 지연이 ~10초이므로
        # 경로를 먼저 알려(위 route 이벤트) 사용자가 왜 기다리는지 알게 한 뒤 한 번에 보낸다.
        from app.agent import answer as agent_answer

        try:
            res = agent_answer(query, dev_mode)
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "message": f"에이전트 오류: {e}"}
            return
        yield {"type": "sources", "sources": res.get("sources", [])}
        yield {"type": "token", "text": res.get("answer", "")}
        yield {"type": "done", "mode": res.get("mode", "agent"),
               "retrieval": res.get("retrieval", {}),
               "trace": res.get("trace", []), "llm_calls": res.get("llm_calls")}
        return

    docs, debug = search(query, doc_types=_axis_of(scope))

    if not docs:
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": _no_hit_text(scope)}
        yield {"type": "done", "mode": "no_hit", "retrieval": debug}
        return

    yield {"type": "sources", "sources": snippets_for(docs, query)}

    provider = settings.active_llm
    if provider == "extractive":
        yield {"type": "token", "text": _extractive_text(docs)}
        yield {"type": "done", "mode": "extractive", "retrieval": debug}
        return

    try:
        for chunk in _llm().stream(_messages(question, docs, dev_mode, turns)):
            piece = _text_of(chunk.content)
            if piece:
                yield {"type": "token", "text": piece}
    except Exception as e:  # noqa: BLE001 — 사용자에게 오류를 그대로 표시
        yield {"type": "error", "message": f"LLM 호출 오류: {e}"}
        return
    yield {"type": "done", "mode": provider, "retrieval": debug}
