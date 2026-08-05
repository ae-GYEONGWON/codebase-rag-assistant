"""LangGraph 판 툴콜링 에이전트 — `app/agent.py` 의 수동 루프를 그래프로 옮긴 것.

**왜 둘을 다 두는가**: `agent.py` 는 while 루프로 직접 짠 구현이고, 이 파일은 같은
동작을 LangGraph 의 상태 그래프로 표현한 것이다. 하나를 지우고 갈아타지 않는 이유는
**둘을 같은 평가셋으로 비교하기 위해서**다. 프레임워크를 도입하면 무엇이 좋아지고
무엇이 그대로인지(정답률·지연·호출 수)를 숫자로 말할 수 있어야 한다.

구조는 표준 ReAct 루프와 같다:

    (시작) → agent ──tool_calls 있음──→ tools ─┐
              ↑                                │
              └────────────────────────────────┘
              └──tool_calls 없음──→ (끝)

수동 루프와 다른 점은 **상태를 그래프가 들고 다닌다**는 것뿐이다. 인용 번호 채번과
중복 제거는 노드가 상태를 갱신하는 방식으로 옮겼다(수동 루프에서는 지역 변수였다).
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent import (
    AGENT_SYSTEM_PROMPT,
    _DEV_OFF,
    _DEV_ON,
    _format_docs,
    _llm_with_tools,
    _run_tool,
    _throttle,
)
from app.config import settings
from app.rag import OUT_OF_SCOPE, _text_of
from app.retriever import snippets_for


class AgentState(TypedDict):
    """그래프가 들고 다니는 상태.

    messages 만 add_messages 리듀서로 누적하고, 나머지는 노드가 통째로 새 값을 돌려준다.
    """

    messages: Annotated[list, add_messages]
    collected: List[Document]  # 인용 순서대로 모인 근거 청크
    seen: List[tuple]          # (source, section) — 같은 청크 재채번 방지
    trace: List[Dict[str, Any]]
    llm_calls: int
    rounds: int                # 툴 라운드 수 — 상한을 재귀 예외가 아니라 그래프 안에서 다룬다


def _call_model(state: AgentState) -> Dict[str, Any]:
    """LLM 에게 다음 행동(툴 호출 또는 최종 답변)을 물어보는 노드."""
    _throttle()  # 무료 티어 분당 15요청 보호 — 수동 루프와 동일 조건으로 비교해야 한다
    resp = _llm_with_tools().invoke(state["messages"])
    return {"messages": [resp], "llm_calls": state["llm_calls"] + 1}


def _run_tools(state: AgentState) -> Dict[str, Any]:
    """요청된 툴을 실행하고 결과를 [근거 N] 블록으로 되돌려주는 노드.

    prebuilt ToolNode 를 쓰지 않는 이유: 이 에이전트는 툴 결과를 문자열로만 쓰는 게
    아니라 **전역 인용 번호**를 매기고 중복 청크를 걸러야 한다. 그 상태가 노드 바깥에
    있어야 해서 직접 구현했다.
    """
    from langchain_core.messages import ToolMessage

    last = state["messages"][-1]
    collected = list(state["collected"])
    seen = set(state["seen"])
    trace = list(state["trace"])
    out_msgs = []

    for call in getattr(last, "tool_calls", None) or []:
        name = call.get("name", "")
        args = call.get("args", {}) or {}
        docs = _run_tool(name, args)

        fresh = []
        for d in docs:
            key = (d.metadata.get("source", ""), d.metadata.get("section", ""))
            if key not in seen:
                seen.add(key)
                fresh.append(d)

        start_no = len(collected) + 1
        collected.extend(fresh)
        trace.append({"tool": name, "args": args, "hits": len(docs), "new": len(fresh)})
        out_msgs.append(
            ToolMessage(content=_format_docs(fresh, start_no), tool_call_id=call.get("id", name))
        )

    return {
        "messages": out_msgs,
        "collected": collected,
        "seen": list(seen),
        "trace": trace,
        "rounds": state["rounds"] + 1,
    }


def _finalize(state: AgentState) -> Dict[str, Any]:
    """툴 상한에 도달했을 때 모은 근거만으로 답을 마무리하는 노드.

    ★ 이 노드가 없으면 재귀 상한 예외로 떨어지고 **모아둔 근거가 통째로 버려진다**
    (수동 루프는 같은 자리에서 '강제 마무리'를 한다). 두 구현을 비교하려면 상한 처리
    방식까지 같아야 한다 — 이게 다르면 프레임워크 차이가 아니라 구현 실수를 재게 된다.
    """
    from langchain_core.messages import HumanMessage

    _throttle()
    msgs = state["messages"] + [
        HumanMessage(content="더 이상 툴을 호출하지 말고, 지금까지 모은 근거만으로 최종 답변을 작성하세요.")
    ]
    resp = _llm_with_tools().invoke(msgs)
    return {"messages": [resp], "llm_calls": state["llm_calls"] + 1}


def _should_continue(state: AgentState) -> str:
    """툴 호출이 없으면 종료, 상한을 넘겼으면 강제 마무리, 아니면 툴 실행."""
    if not getattr(state["messages"][-1], "tool_calls", None):
        return END
    return "finalize" if state["rounds"] >= settings.agent_max_steps else "tools"


def _build():
    g = StateGraph(AgentState)
    g.add_node("agent", _call_model)
    g.add_node("tools", _run_tools)
    g.add_node("finalize", _finalize)
    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent", _should_continue, {"tools": "tools", "finalize": "finalize", END: END}
    )
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)
    return g.compile()


_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build()
    return _GRAPH


def answer(question: str, dev_mode: bool = False) -> Dict[str, Any]:
    """`app.agent.answer()` 와 **동일한 반환 형태** — 평가 하네스가 그대로 비교할 수 있게."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.errors import GraphRecursionError

    if settings.active_llm == "extractive":
        from app.rag import answer as rag_answer

        return rag_answer(question, dev_mode)

    system = AGENT_SYSTEM_PROMPT + (_DEV_ON if dev_mode else _DEV_OFF)
    init: AgentState = {
        "messages": [SystemMessage(content=system), HumanMessage(content=question)],
        "collected": [],
        "seen": [],
        "trace": [],
        "llm_calls": 0,
        "rounds": 0,
    }
    # 라운드 상한은 finalize 노드가 처리한다. recursion_limit 은 그 위의 안전망이라
    # 넉넉히 준다(한 라운드 = agent+tools 2노드 + 마지막 agent/finalize).
    limit = settings.agent_max_steps * 2 + 5

    try:
        final = _graph().invoke(init, config={"recursion_limit": limit})
        text = _text_of(final["messages"][-1].content) or OUT_OF_SCOPE
        mode = "agent_graph"
    except GraphRecursionError:
        return {
            "answer": "재귀 상한에 도달해 답변을 마무리하지 못했습니다.",
            "sources": [],
            "mode": "agent_graph_recursion",
            "trace": [],
            "steps": settings.agent_max_steps,
            "llm_calls": 0,
        }
    except Exception as e:  # noqa: BLE001 — 호출 실패는 그대로 알린다
        return {
            "answer": f"에이전트(LangGraph) 호출 오류: {e}",
            "sources": [],
            "mode": "agent_graph_error",
            "trace": [],
            "steps": 0,
            "llm_calls": 0,
        }

    return {
        "answer": text,
        "sources": snippets_for(final["collected"], question),
        "mode": mode,
        "trace": final["trace"],
        "steps": len(final["trace"]),
        "llm_calls": final["llm_calls"],
    }
