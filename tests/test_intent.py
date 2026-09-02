"""의도 라우팅 규칙 테스트.

여기서 고정하려는 것은 "잘 맞힌다"가 아니라 **틀렸을 때 손해가 나지 않는 설계**다.
판별이 애매하면 unknown 을 내고 호출부가 기존 동작을 유지해야 한다 — 규칙이
욕심을 부려 한 축으로 접으면 멀티홉 질문에서 반대쪽 근거를 통째로 잃는다.
"""
import pytest

from app.intent import classify, symbol_slots_for


# --- 축 판별 ----------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "RRF 융합은 코드에서 어떻게 구현돼 있어?",
    "임베딩 제공자를 고르는 함수 어디 있어?",
    "코퍼스 프로필을 등록하는 데코레이터는 어디에 있어?",
])
def test_구현_위치를_물으면_code(q):
    assert classify(q).axis == "code"


@pytest.mark.parametrize("q", [
    "MMR 다양성 계수를 1.0 으로 정한 근거가 뭐야?",
    "리랭커를 왜 기본으로 껐어?",
    "demo 프로필과 private 프로필은 뭐가 달라?",
])
def test_이유_설계를_물으면_doc(q):
    assert classify(q).axis == "doc"


@pytest.mark.parametrize("q", [
    "이 설정은 언제 바뀌었어?",
    "최근에 바뀐 부분이 뭐야?",
    "변경 이력을 알려줘",
])
def test_변경_시점을_물으면_commit(q):
    assert classify(q).axis == "commit"


# --- 애매할 때는 손대지 않는다 ----------------------------------------------

def test_이유와_구현이_함께면_unknown():
    # 멀티홉 질문. 한 축으로 접으면 반대쪽 근거를 잃으므로 판별을 포기해야 한다.
    i = classify("RRF 를 왜 쓰고 코드에서 어떻게 구현돼 있어?")
    assert i.axis is None
    assert "멀티홉" in i.reason


def test_표지어가_없으면_unknown():
    assert classify("ERR7742").axis is None


def test_빈_질문도_안전하게_unknown():
    assert classify("   ").axis is None


def test_커밋과_구현이_함께면_한_축으로_접지_않는다():
    # "왜 바뀌었고 지금 코드는 어떻게 돼?" — 커밋 표지어가 있어도 코드가 같이 오면 보류.
    assert classify("그 값이 왜 바뀌었고 지금 코드는 어떻게 돼 있어?").axis is None


# --- 슬롯 결정 --------------------------------------------------------------

def test_코드_질문은_심볼슬롯을_유지한다():
    n, i = symbol_slots_for("RRF 융합은 코드에서 어떻게 구현돼 있어?", 2)
    assert (n, i.axis) == (2, "code")


def test_문서_질문은_심볼슬롯을_끈다():
    # 질문에 mmr 이라는 식별자가 있어도 설계 노트를 밀어내면 안 된다(노트 #17).
    n, i = symbol_slots_for("MMR 다양성 계수를 1.0 으로 정한 근거가 뭐야?", 2)
    assert (n, i.axis) == (0, "doc")


def test_커밋_질문도_심볼슬롯을_끈다():
    n, i = symbol_slots_for("이 값은 언제 바뀌었어?", 2)
    assert (n, i.axis) == (0, "commit")


def test_판별_실패시_기존_동작을_유지한다():
    # unknown 에서 슬롯을 끄면 '판별 안 하던 때'보다 나빠질 수 있다 → 기본값 유지.
    n, i = symbol_slots_for("ERR7742", 2)
    assert (n, i.axis) == (2, None)
