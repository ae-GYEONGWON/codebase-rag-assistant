"""토크나이저 구현 비교 테스트 — 두 방식이 각각 무엇을 보장하는지 고정한다.

`tests/test_tokenizer.py` 는 기존 2-gram 동작을 고정한다. 이 파일은 **형태소 분석기를
붙였을 때 깨지면 안 되는 것**을 본다. 특히 이 시스템의 핵심 능력인
**희귀 식별자 정확매칭**(ERR7742 류)이 분석기 교체로 사라지면 안 된다.
"""
import pytest

from app.tokenizer import tokenize_kiwi, tokenize_ngram

IMPLS = [tokenize_ngram, tokenize_kiwi]


# --- 두 구현이 공통으로 지켜야 할 것 -----------------------------------------

@pytest.mark.parametrize("fn", IMPLS)
def test_희귀_식별자는_원형_그대로_남는다(fn):
    # 이걸 잃으면 하이브리드 검색의 존재 이유가 사라진다(벡터가 못 잡는 토큰).
    assert "err7742" in fn("ERR7742 는 어디서 나?")


@pytest.mark.parametrize("fn", IMPLS)
def test_snake_case는_조각으로도_색인된다(fn):
    toks = fn("apply_retry_policy 가 뭐야?")
    assert "apply_retry_policy" in toks
    assert {"apply", "retry", "policy"} <= set(toks)


@pytest.mark.parametrize("fn", IMPLS)
def test_빈_문자열은_빈_토큰(fn):
    assert fn("") == []


@pytest.mark.parametrize("fn", IMPLS)
def test_조사가_달라도_어간이_겹친다(fn):
    # "리랭커를" 질의가 "리랭커는" 문서에 걸려야 한다. 방식은 달라도 목적은 같다.
    assert set(fn("리랭커를")) & set(fn("리랭커는"))


# --- 형태소 분석기가 개선하는 것 ---------------------------------------------

def test_kiwi는_조사를_떼고_어간만_남긴다():
    toks = tokenize_kiwi("리랭커를 왜 껐어?")
    assert "리랭커" in toks
    assert "리랭커를" not in toks


def test_kiwi는_한_글자_조각을_버린다():
    # '으', '수', '것' 같은 것이 남으면 2-gram 이 겪던 잡음 문제가 그대로 재현된다.
    assert all(len(t) >= 2 or t.isascii() for t in tokenize_kiwi("1.0 으로 정한 근거가 뭐야?"))


def test_kiwi는_같은_토큰을_두_번_넣지_않는다():
    # ASCII 는 정규식 패스가 이미 넣는다. 형태소 패스가 또 넣으면 BM25 의 tf 가
    # 두 배가 되어 IDF 가 깎이고, 희귀 식별자 매칭이라는 강점이 무뎌진다.
    toks = tokenize_kiwi("MMR 계수를 정한 근거")
    assert toks.count("mmr") == 1


def test_kiwi가_ngram보다_토큰이_적다():
    # 정밀도의 대가로 재현율을 조금 잃는 교환이다. 그 방향 자체를 고정해 둔다.
    q = "MMR 다양성 계수를 1.0 으로 정한 근거가 뭐야?"
    assert len(tokenize_kiwi(q)) < len(tokenize_ngram(q))


# --- 기존 2-gram 의 알려진 결함 ----------------------------------------------

def test_ngram은_짧은_한글어절에서_토큰이_중복된다():
    # 2글자 어절은 '어절 자체'와 '2-gram' 이 같은 문자열이라 두 번 들어간다.
    # BM25 의 tf 를 부풀리는 실제 결함 — 고치면 기준선 숫자가 움직이므로
    # 여기서는 **현상을 고정**만 하고, 변경은 측정과 함께 별도로 다룬다.
    assert tokenize_ngram("으로").count("으로") == 2
