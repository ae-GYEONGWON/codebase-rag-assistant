"""툴콜링 에이전트 — 검색을 '한 번'이 아니라 '필요한 만큼' 하는 레이어.

기존 단발 RAG(`app/rag.py`)는 질문 → 검색 1회 → 답변이다. 이 구조는 단일 사실
질문("범위 밖 임계값은?")에는 충분하지만, **멀티홉** 질문에는 구조적으로 약하다:

    "재시도 정책이 왜 바뀌었고, 지금 코드는 어떻게 동작해?"
      → git 이력에서 '왜'를, 코드에서 '지금'을 각각 찾아야 한다.
        한 번의 검색으로는 두 축 중 하나가 표면 유사도에 밀린다.

그래서 축(문서/코드/커밋)별 검색 툴을 주고 LLM 이 스스로 호출 순서를 정하게 한다.
심볼 본문이 더 필요하면 `read_symbol` 로 되돌아가 읽는다.

**설계 원칙 — 측정 가능하게 만든다**: `answer()` 는 답변뿐 아니라 `trace`(툴 호출
순서·인자·히트 수)와 `steps`/`llm_calls` 를 함께 반환한다. 에이전트가 단발 RAG 보다
정말 나은지(정답률 ↑ vs 지연·토큰 ↑)를 eval 하네스가 숫자로 비교하기 위한 것이다.
리랭커를 측정 후 껐던 것과 같은 이유로, 이 레이어도 기본값은 OFF(`USE_AGENT=false`)다.
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Callable, Dict, List, Tuple

from langchain_core.documents import Document

from app.config import settings
from app.rag import OUT_OF_SCOPE, _text_of
from app.retriever import get_symbol, search, snippets_for

AGENT_SYSTEM_PROMPT = (
    "당신은 인덱싱된 소프트웨어 프로젝트(문서·소스코드·git 이력)의 어시스턴트입니다.\n"
    "답을 지어내지 말고, 반드시 **툴로 근거를 찾아** 답하세요.\n\n"
    "[툴 사용 전략]\n"
    "- `search_docs`: 설계 의도·정책·용어 같은 '문서에 적힌 것'\n"
    "- `search_code`: 실제 구현·계산식·분기 같은 '코드가 하는 것'\n"
    "- `search_commits`: '언제/왜 바뀌었나', 최근 변경, 폐기 여부\n"
    "- `read_symbol`: 검색으로 함수명을 알아낸 뒤 그 **본문 전체**를 읽어야 할 때\n\n"
    "질문이 여러 축에 걸치면(예: '왜 바뀌었고 지금은 어떻게 동작해?') "
    "축마다 따로 호출하세요. 한 축의 결과가 비면 다른 축을 시도하세요.\n"
    "**'지금/현재 어떻게 동작하는가'는 문서가 아니라 코드가 근거입니다** — 이 부분을 답할 때는 "
    "반드시 `search_code`(필요하면 이어서 `read_symbol`)로 확인하세요. 문서·커밋만으로 "
    "현재 구현을 단정하지 마세요. 문서는 낡을 수 있습니다.\n"
    "**질문이 여러 개의 물음을 담고 있으면(‘왜’ + ‘언제’ + ‘지금 어떻게’) 물음마다 축을 하나씩 배정해 "
    "빠짐없이 호출하세요.** '왜/배경/개념'→search_docs, '언제/도입·변경 시점'→search_commits, "
    "'지금/구현/어디에'→search_code. 답하기 전에 질문의 각 물음이 근거로 뒷받침되는지 확인하고, "
    "빠진 축이 있으면 그 축을 먼저 호출하세요.\n"
    "충분한 근거를 모았다고 판단되면 더 호출하지 말고 바로 답하세요.\n\n"
    "[답변 규칙]\n"
    "1. 툴이 돌려준 [근거] 에만 기반해 한국어로 답하세요. 어느 툴에서도 근거를 못 찾으면 "
    f"'{OUT_OF_SCOPE}' 라고 답하세요.\n"
    "2. 읽는 사람은 개발자가 아닐 수 있습니다. **핵심 결론을 평이한 한 문장으로 먼저** 쓰고, "
    "그 뒤에 필요하면 불릿/표로 풀어 쓰세요.\n"
    "3. 근거로 쓴 발췌를 문장 끝에 [1], [2] 각주로 표기하세요. 번호는 [근거 N] 의 N 과 일치시킵니다.\n"
    "4. 코드가 근거일 때는 파일·함수명을 밝히세요.\n"
    "5. **문서와 코드가 어긋나면 그 사실을 명시**하세요. 실제 동작은 코드가 기준입니다.\n"
    "6. 설정값·수치는 근거에 적힌 그대로 정확히 인용하세요."
)

_DEV_ON = (
    "\n\n[개발자 모드 ON] 구현·코드 질문이고 근거에 함수 본문이 있으면 "
    "```python 코드블록으로 인용한 뒤 핵심 라인을 짚어 설명하세요."
)
_DEV_OFF = (
    "\n\n[개발자 모드 OFF] 독자는 비개발자입니다. 코드 본문(``` 블록)은 넣지 말고 "
    "그 코드가 무엇을 하는지 일상어로 설명하세요."
)

_TOOL_K = 4  # 툴 1회당 반환 청크 수. 단발 RAG(5)보다 작게 — 여러 번 부르므로 컨텍스트가 누적된다.


# ──────────────────────────────────────────────────────────────
# 툴 정의
# ──────────────────────────────────────────────────────────────
def _tool_specs() -> List[Dict[str, Any]]:
    """provider 중립 툴 스키마(OpenAI/Gemini 양쪽 bind_tools 가 받는 형식)."""
    def q(desc: str) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": desc}},
            "required": ["query"],
        }

    return [
        {
            "name": "search_docs",
            "description": "프로젝트 문서(.md)에서 설계 의도·정책·용어를 검색한다.",
            "parameters": q("검색할 자연어 질의"),
        },
        {
            "name": "search_code",
            "description": "소스코드(함수·클래스 단위)에서 실제 구현·계산식·분기를 검색한다.",
            "parameters": q("검색할 자연어 질의 또는 심볼명"),
        },
        {
            "name": "search_commits",
            "description": "git 커밋 이력에서 '언제·왜 바뀌었는지', 최근 변경, 폐기 여부를 검색한다.",
            "parameters": q("검색할 자연어 질의"),
        },
        {
            "name": "read_symbol",
            "description": "함수/클래스 이름으로 그 코드 본문 전체를 읽는다. 검색으로 이름을 알아낸 뒤 사용.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "함수 또는 클래스 이름"}},
                "required": ["name"],
            },
        },
    ]


def _run_tool(name: str, args: Dict[str, Any]) -> List[Document]:
    """툴 이름 → 청크 목록. 실행 결과를 문서로 통일해 인용 번호를 한 곳에서 매긴다."""
    if name == "read_symbol":
        return get_symbol(str(args.get("name", "")))

    query = str(args.get("query", ""))
    types: Dict[str, Tuple[str, ...]] = {
        "search_docs": ("doc",),
        "search_code": ("code",),
        "search_commits": ("commit",),
    }
    if name not in types:
        return []
    docs, _ = search(query, k=_TOOL_K, doc_types=types[name])
    return docs


# ──────────────────────────────────────────────────────────────
# 에이전트 루프
# ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _llm_with_tools():
    from app.rag import _llm

    return _llm().bind_tools(_tool_specs())


_last_call: List[float] = [0.0]


def _throttle() -> None:
    """무료 티어(분당 15요청) 보호. 에이전트는 한 질문에 3~5회 호출한다."""
    wait = settings.agent_throttle_sec - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()


def _format_docs(docs: List[Document], start_no: int) -> str:
    """툴 결과를 [근거 N] 블록으로. N 은 대화 전체에서 이어지는 전역 번호."""
    if not docs:
        return "(이 축에서는 관련 근거를 찾지 못했습니다.)"
    blocks = []
    for offset, d in enumerate(docs):
        kind = {"code": "코드", "commit": "커밋"}.get(d.metadata.get("doc_type", ""), "문서")
        src = d.metadata.get("source", "?")
        sec = d.metadata.get("section", "")
        head = f"[근거 {start_no + offset}] ({kind}) {src}" + (f" > {sec}" if sec else "")
        blocks.append(f"{head}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def answer(question: str, dev_mode: bool = False) -> Dict[str, Any]:
    """질문 → {answer, sources, mode, trace, steps, llm_calls}.

    trace/steps/llm_calls 는 eval 하네스가 '에이전트 vs 단발 RAG' 를 비교하는 재료다.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    if settings.active_llm == "extractive":
        # 툴콜링은 LLM 기능이라 extractive 폴백에서는 성립하지 않는다 → 단발 RAG 로 위임.
        from app.rag import answer as rag_answer

        return rag_answer(question, dev_mode)

    system = AGENT_SYSTEM_PROMPT + (_DEV_ON if dev_mode else _DEV_OFF)
    messages: List[Any] = [SystemMessage(content=system), HumanMessage(content=question)]

    collected: List[Document] = []
    seen: set = set()          # (source, section) — 같은 청크가 여러 툴에서 나와도 번호는 하나
    trace: List[Dict[str, Any]] = []
    llm_calls = 0
    llm = _llm_with_tools()

    for step in range(settings.agent_max_steps):
        _throttle()
        try:
            resp = llm.invoke(messages)
        except Exception as e:  # noqa: BLE001 — 호출 실패는 사용자에게 그대로 알린다
            return {
                "answer": f"에이전트 LLM 호출 오류: {e}",
                "sources": snippets_for(collected, question),
                "mode": "agent_error",
                "trace": trace,
                "steps": step,
                "llm_calls": llm_calls,
            }
        llm_calls += 1
        messages.append(resp)

        calls = getattr(resp, "tool_calls", None) or []
        if not calls:
            return {
                "answer": _text_of(resp.content),
                "sources": snippets_for(collected, question),
                "mode": "agent",
                "trace": trace,
                "steps": step,
                "llm_calls": llm_calls,
            }

        for call in calls:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            docs = _run_tool(name, args)

            # 이미 인용된 청크는 번호를 재사용하지 않고 제외 → 각주 번호 충돌 방지
            fresh = []
            for d in docs:
                key = (d.metadata.get("source", ""), d.metadata.get("section", ""))
                if key not in seen:
                    seen.add(key)
                    fresh.append(d)

            start_no = len(collected) + 1
            collected.extend(fresh)
            trace.append({"step": step, "tool": name, "args": args, "hits": len(docs), "new": len(fresh)})

            messages.append(
                ToolMessage(content=_format_docs(fresh, start_no), tool_call_id=call.get("id", name))
            )

    # 상한까지 툴만 부르고 결론을 못 낸 경우 — 모은 근거로 강제 마무리.
    _throttle()
    messages.append(
        HumanMessage(content="더 이상 툴을 호출하지 말고, 지금까지 모은 근거만으로 최종 답변을 작성하세요.")
    )
    try:
        final = _llm_with_tools().invoke(messages)
        llm_calls += 1
        text = _text_of(final.content) or OUT_OF_SCOPE
    except Exception as e:  # noqa: BLE001
        text = f"에이전트 LLM 호출 오류: {e}"

    return {
        "answer": text,
        "sources": snippets_for(collected, question),
        "mode": "agent_max_steps",
        "trace": trace,
        "steps": settings.agent_max_steps,
        "llm_calls": llm_calls,
    }
