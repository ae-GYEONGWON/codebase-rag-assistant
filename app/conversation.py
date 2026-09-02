"""멀티턴 대화 — 후속 질문을 검색 가능한 형태로 되돌린다.

## 왜 필요했나

지금까지는 질문 하나를 받아 검색하고 답하는 **단발 구조**였다. 데모에서 이건 60초 만에 드러난다:

    사용자: 리랭커를 왜 껐어?
    봇:    측정해 보니 코드 질문 정확도가 떨어져서…  ✅
    사용자: 그럼 그건 어떻게 측정했어?
    봇:    (검색어 = "그럼 그건 어떻게 측정했어?" → 아무것도 안 걸린다)  ❌

"그건"이 무엇인지는 **앞 대화에만** 있다. 그래서 검색 전에 질문을 **독립형(standalone)** 으로
되돌린다 — 이걸 query rewriting 이라 부르고, 멀티턴 RAG 의 표준 처리다.

## 설계 결정

- **재작성한 질의로 검색하고, 원문 질문으로 답한다.** 검색기는 지시대명사를 못 풀지만,
  생성 모델은 대화 맥락을 함께 주면 자연스럽게 답한다. 둘의 요구가 다르므로 입력을 나눈다.
- **재작성 결과를 UI 에 노출한다.** 무엇으로 검색했는지 보이지 않으면 사용자는 왜 그 답이
  나왔는지 알 수 없다. 이 제품의 핵심 가치가 '근거를 보여주는 것'이라 여기서도 같은 원칙을 지킨다.
- **첫 질문이면 LLM 을 부르지 않는다.** 재작성할 맥락이 없는데 호출하면 지연만 늘어난다
  (무료 티어 쿼터도 아낀다).
- **재작성이 실패하면 원문으로 검색한다.** 대화 기능 때문에 기존 단발 경로가 죽으면 안 된다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.config import settings

# 프롬프트에 실을 직전 대화 턴 수. 길게 실으면 재작성이 오래된 주제로 끌려간다.
MAX_HISTORY_TURNS = 6
# 한 턴에서 잘라 쓸 최대 길이(답변이 길면 재작성 프롬프트가 답변 요약이 되어 버린다).
MAX_TURN_CHARS = 600

REWRITE_PROMPT = """다음은 어떤 소프트웨어 프로젝트에 대한 대화입니다.
마지막 질문을 **앞 대화 없이도 이해되는 하나의 독립된 질문**으로 다시 쓰세요.

규칙:
- 지시대명사("그건", "거기서", "이 방식")를 앞 대화의 실제 대상으로 바꾸세요.
- 생략된 주어·목적어를 복원하세요.
- 새로운 정보를 추가하거나 질문의 의도를 바꾸지 마세요.
- 이미 독립적으로 이해되는 질문이면 **그대로 반환**하세요.
- 질문 한 문장만 출력하세요. 따옴표·설명·접두어 금지.

[대화]
{history}

[마지막 질문]
{question}

[독립형 질문]"""

# 앞 맥락이 없으면 이해 안 되는 신호들. 재작성 필요 여부를 값싸게 가늠한다.
_DEPENDENT_HINTS = re.compile(
    r"(그것|그거|그건|그게|그런|그렇|이것|이거|이건|이게|저것|거기|여기|"
    r"위에서|앞에서|방금|아까|다시|더 자세|자세히|왜 그|어떻게 그|그럼|그러면|"
    r"^\s*(왜|어떻게|언제|어디서|누가|뭐|무엇)[\s?]*$)"
)


# 재작성이 **지어낼 수 있는 것**을 잡아내는 패턴.
#   숫자      — 임계값·비율·버전. 지어내면 검색이 없는 값을 쫓는다.
#   식별자    — 파일·함수·설정 이름. 지어내면 없는 코드를 찾으러 간다.
# 3자 미만 식별자는 제외한다(조사·관사 수준의 잡음이 섞인다).
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./]{2,}")


@dataclass
class Turn:
    role: str      # user | assistant
    content: str


def parse_history(raw: Optional[List[dict]]) -> List[Turn]:
    """API 로 받은 히스토리를 정규화. 형식이 어긋난 항목은 조용히 버린다."""
    turns: List[Turn] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).lower()
        content = str(item.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            turns.append(Turn(role, content[:MAX_TURN_CHARS]))
    return turns[-MAX_HISTORY_TURNS:]


def looks_dependent(question: str) -> bool:
    """앞 맥락이 있어야 이해되는 질문인가(휴리스틱).

    확실히 판별하려면 LLM 을 불러야 하지만, 그건 재작성 자체와 같은 비용이다.
    그래서 여기서는 **재작성을 건너뛰어도 안전한 경우만** 걸러내는 용도로 쓴다
    (놓쳐서 재작성하는 건 손해가 없지만, 필요한데 건너뛰면 검색이 실패한다 → 관대하게 판정).
    """
    q = question.strip()
    if len(q) <= 12:                      # "왜?", "더 자세히" 같은 짧은 후속
        return True
    return bool(_DEPENDENT_HINTS.search(q))


def invented_facts(rewritten: str, question: str, turns: List[Turn]) -> List[str]:
    """재작성이 **대화에 없던 숫자·식별자**를 새로 넣었으면 그 목록을 돌려준다.

    ## 왜 프롬프트로는 부족한가

    재작성 프롬프트에는 이미 "새로운 정보를 추가하지 마세요"가 있다. 그런데도 실제 데모에서
    이런 일이 났다.

        질문   "MMR 계수를 왜 그렇게 정했고 코드에서는 어떻게 구현돼 있어?"
        재작성 "MMR 계수를 왜 **0.5로** 설정했고, …"        ← 0.5 는 아무 데도 없다(실제 값 1.0)

    재작성된 질의로 **검색을 하기 때문에** 이건 그냥 어색한 문장이 아니다. 검색기가 존재하지
    않는 값을 쫓게 되고, 그 결과로 고른 근거 위에서 답이 만들어진다. 지시를 지키라고 부탁하는
    것으로는 막히지 않았으므로 **검증**으로 막는다.

    ## 판정 기준

    재작성이 하는 일은 지시대명사를 앞 대화의 **실제 대상으로 치환**하는 것이다. 그렇다면
    결과에 나오는 숫자·식별자는 전부 원문 질문이나 대화에 이미 있어야 한다. 없다면
    치환이 아니라 창작이다.

    보수적으로 본다 — 애매하면 재작성을 버리고 원문으로 검색한다. 그건 멀티턴 도입 이전의
    동작이라 최악이라도 '예전만큼'이지 더 나빠지지 않는다.
    """
    haystack = " ".join([question] + [t.content for t in turns]).lower()

    invented: List[str] = []
    for tok in _NUMBER_RE.findall(rewritten):
        # 숫자는 표기가 흔들린다(1.0 ↔ 1). 정수부만이라도 있으면 지어낸 것으로 보지 않는다.
        if tok.lower() not in haystack and tok.split(".")[0] not in haystack:
            invented.append(tok)
    for tok in _IDENT_RE.findall(rewritten):
        if tok.lower() not in haystack:
            invented.append(tok)
    return invented


def format_history(turns: List[Turn]) -> str:
    label = {"user": "사용자", "assistant": "어시스턴트"}
    return "\n".join(f"{label[t.role]}: {t.content}" for t in turns)


def rewrite_query(question: str, turns: List[Turn]) -> Tuple[str, Dict[str, object]]:
    """(검색에 쓸 질의, 진단정보) 반환.

    진단정보는 UI 에 그대로 노출한다 — 무엇으로 검색했는지 보여야 답의 출처를 이해할 수 있다.
    """
    info: Dict[str, object] = {"original": question, "rewritten": None, "applied": False}

    if not turns:
        info["skipped"] = "first_turn"
        return question, info
    if not looks_dependent(question):
        info["skipped"] = "standalone"
        return question, info
    if settings.active_llm == "extractive":
        info["skipped"] = "no_llm"
        return question, info

    try:
        from app.rag import _llm, _text_of

        prompt = REWRITE_PROMPT.format(history=format_history(turns), question=question)
        out = _text_of(_llm().invoke(prompt).content).strip()
        out = out.strip().strip('"').strip("'").split("\n")[0].strip()
        # 모델이 빈 문자열이나 장문 설명을 뱉으면 신뢰하지 않는다.
        if out and 3 <= len(out) <= 300:
            # 지시 준수를 믿지 않고 확인한다 — 없던 숫자·식별자를 넣었으면 버린다.
            invented = invented_facts(out, question, turns)
            if invented:
                info["rejected"] = out
                info["invented"] = invented
                info["skipped"] = "invented_facts"
                return question, info
            info["rewritten"] = out
            info["applied"] = out != question
            return out, info
        info["skipped"] = "unusable_output"
    except Exception as e:  # noqa: BLE001 — 대화 기능 때문에 단발 경로가 죽으면 안 된다
        info["skipped"] = f"error:{type(e).__name__}"
    return question, info


def to_messages(turns: List[Turn]):
    """LangChain 메시지 리스트로. 생성 단계에는 **원문 대화**를 그대로 준다."""
    from langchain_core.messages import AIMessage, HumanMessage

    return [
        (HumanMessage if t.role == "user" else AIMessage)(content=t.content)
        for t in turns
    ]
