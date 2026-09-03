"""라우터 — 질문마다 **단발 RAG** 와 **에이전트** 중 하나를 고른다.

## 왜 자동 선택이 필요한가

두 경로는 서로를 대체하지 못한다. 측정된 트레이드오프가 분명하다.

| | 단발 RAG | 에이전트(툴콜링) |
|---|---|---|
| LLM 호출 | 1회 | 3~5회 |
| 첫 응답 | ~1.4s | ~10s (측정치, 20배) |
| 단일 사실 질문 | 충분 | 낭비 |
| 멀티홉 질문 | 한 축이 밀림 | 축마다 따로 검색 → 정답률 42% → 50% |

그래서 "항상 에이전트"는 대부분의 질문에 20배 지연을 물리는 선택이고, "항상 단발"은
멀티홉 질문을 구조적으로 포기하는 선택이다. **질문마다 고르는 것**이 옳다.

## 무엇으로 고르는가

에이전트가 이기는 조건은 하나로 요약된다 — **한 번의 검색으로는 한 축이 밀리는 질문.**
그건 질문이 축을 둘 이상 건드릴 때 생긴다("왜 바뀌었고 지금 코드는 어떻게 동작해?").
그 판별은 이미 `app/intent.py` 가 하고 있으므로(축 표지어 집합), 여기서는 그 결과를 쓴다.

LLM 에게 "이 질문이 멀티홉이냐"를 묻지 않는 이유는 앞뒤가 안 맞기 때문이다. 그걸 물으면
**단발 질문에도 LLM 호출이 한 번 더 붙어**, 라우터가 아끼려던 비용을 라우터가 쓴다.

## 틀렸을 때의 비대칭

두 방향의 오판은 손해가 다르다.

    멀티홉을 단발로 보냄  → 답이 한 축만 다룬다(품질 손실, 사용자가 알아채기 어렵다)
    단발을 에이전트로 보냄 → 답은 맞고 느리다(비용 손실, 사용자가 바로 알아챈다)

품질 손실이 더 나쁘지만, 무료 티어에서는 에이전트 남발이 **쿼터를 태워 서비스를 멈춘다.**
그래서 기본값은 보수적으로 두고(`USE_ROUTER=false`), 켜고 끈 값을 비교할 수 있게 남긴다 —
리랭커·에이전트를 그렇게 다뤘던 것과 같은 규칙이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from app.config import settings
from app.intent import classify, josa, labels_of

# 에이전트 한 번의 대략적인 LLM 호출 수(라운드 + 최종 답변). 비용 표시용 근사치다.
AGENT_LLM_CALLS = 4


@dataclass(frozen=True)
class Route:
    """선택 결과. UI 진단 패널과 평가 리포트가 이 값을 그대로 보여 준다."""

    mode: str            # "single" | "agent"
    reason: str
    axes: Tuple[str, ...] = ()

    @property
    def uses_agent(self) -> bool:
        return self.mode == "agent"


def decide(question: str, force: str | None = None) -> Route:
    """질문 → 경로. `force` 가 주어지면 그대로 따른다(평가·디버깅용).

    라우터가 꺼져 있으면 설정값(`USE_AGENT`)을 그대로 존중한다 — 라우터를 도입했다고
    기존 스위치의 의미가 바뀌면 안 된다.
    """
    if force in ("single", "agent"):
        return Route(force, "요청하신 방식으로 찾았습니다")

    if not settings.use_router:
        return (Route("agent", "설정에 따라 나눠서 찾습니다")
                if settings.use_agent else
                Route("single", "기본 방식 — 한 번 찾아서 답합니다"))

    intent = classify(question)
    if intent.is_multihop:
        return Route("agent",
                     f"{josa(labels_of(intent.axes), '을', '를')} 함께 물으셔서 나눠서 찾았습니다 — "
                     "한 번에 찾으면 한쪽이 밀립니다 (그만큼 더 걸립니다)",
                     intent.axes)
    if intent.axis:
        return Route("single",
                     f"{intent.display_name}만 물으셔서 한 번에 찾았습니다",
                     intent.axes)
    return Route("single",
                 "어느 쪽을 묻는지 분명하지 않아 전체에서 한 번에 찾았습니다",
                 intent.axes)
