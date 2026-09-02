"""라우터 테스트 — 무엇을 에이전트로 보내고 무엇을 아끼는가.

여기서 고정하는 것은 두 가지다.
① 축을 둘 이상 물을 때만 에이전트로 간다(그 외에는 20배 지연을 물리지 않는다).
② 라우터가 꺼져 있으면 기존 스위치(`USE_AGENT`)의 의미가 그대로 유지된다 —
   장치를 하나 넣었다고 기존 동작이 조용히 바뀌면 안 된다.
"""
import pytest

from app.config import settings
from app.router import decide


@pytest.fixture
def router_on():
    prev_r, prev_a = settings.use_router, settings.use_agent
    settings.use_router, settings.use_agent = True, False
    yield
    settings.use_router, settings.use_agent = prev_r, prev_a


# --- 라우터가 켜졌을 때 -------------------------------------------------------

@pytest.mark.parametrize("q", [
    "리랭커를 왜 껐어?",
    "RRF 융합은 코드에서 어떻게 구현돼 있어?",
    "이 값은 언제 바뀌었어?",
])
def test_단일_축은_단발_RAG로_보낸다(router_on, q):
    assert decide(q).mode == "single"


@pytest.mark.parametrize("q", [
    "리랭커를 왜 껐고 지금 코드는 어떻게 돼 있어?",
    "MMR 계수를 왜 그렇게 정했고 구현은 어디에 있어?",
])
def test_축을_둘_이상_물으면_에이전트로_보낸다(router_on, q):
    r = decide(q)
    assert r.mode == "agent" and len(r.axes) >= 2


def test_에이전트_선택_이유에_비용이_적힌다(router_on):
    # 사용자가 왜 10초를 기다리는지 화면에서 알 수 있어야 한다.
    assert "LLM" in decide("리랭커를 왜 껐고 지금 코드는 어떻게 돼?").reason


def test_표지어가_없으면_단발(router_on):
    assert decide("ERR7742").mode == "single"


# --- 라우터가 꺼졌을 때: 기존 의미를 지킨다 -----------------------------------

def test_라우터가_꺼져_있으면_USE_AGENT를_따른다():
    prev_r, prev_a = settings.use_router, settings.use_agent
    try:
        settings.use_router = False
        settings.use_agent = False
        assert decide("왜 그렇고 코드는 어디 있어?").mode == "single"
        settings.use_agent = True
        assert decide("리랭커를 왜 껐어?").mode == "agent"
    finally:
        settings.use_router, settings.use_agent = prev_r, prev_a


# --- 강제 지정 ---------------------------------------------------------------

@pytest.mark.parametrize("forced", ["single", "agent"])
def test_명시적_지정이_라우터보다_우선한다(router_on, forced):
    # 평가 하네스가 두 경로를 같은 문항으로 비교하려면 강제가 가능해야 한다.
    assert decide("리랭커를 왜 껐고 지금 코드는 어떻게 돼?", force=forced).mode == forced


def test_알_수_없는_force_값은_무시하고_판단한다(router_on):
    assert decide("리랭커를 왜 껐어?", force="이상한값").mode == "single"
