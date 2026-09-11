"""실사용형 평가셋의 **통제 장치** 테스트.

이 셋의 결론(골든셋 85% vs 실사용형 50%)은 두 가지가 맞아야만 성립한다.

1. `primary_only` 가 **양쪽 다** 첫 라벨 하나로 자를 것 — 한쪽만 잘리면 그 차이는
   말씨가 아니라 라벨 수의 차이가 된다. 결론이 통째로 뒤집히는 종류의 고장이라
   눈으로는 못 본다(숫자는 그럴듯하게 나온다).
2. 문항이 세 종류를 실제로 담고 있을 것 — 종류가 비면 표의 행이 조용히 사라진다.

인덱스·임베딩이 필요 없는 순수 로직만 본다(CI 에서 그대로 돈다).
"""
import json

import pytest

from eval import realistic_eval as re_


@pytest.fixture(scope="module")
def cases():
    return re_.load_cases()


def test_문항은_세_종류를_모두_담는다(cases):
    kinds = {c["kind"] for c in cases}
    assert kinds == set(re_.KIND_LABEL), f"빠진 종류: {set(re_.KIND_LABEL) - kinds}"


def test_모든_문항에_질문과_라벨이_있다(cases):
    for c in cases:
        assert c["q"].strip(), c
        assert c["expected"], f"정답 파일이 없는 문항: {c['q']}"


def test_라벨은_수기_표시를_단다(cases):
    """자가 라벨 편향을 수치화하려면 라벨 출처가 데이터에 적혀 있어야 한다(datasets.py 규칙)."""
    assert all(c.get("origin") == "manual" for c in cases)


def test_첫_라벨만_자르기는_라벨을_하나로_만든다(cases):
    cut = re_.primary_only(cases)
    assert all(len(c["expected"]) == 1 for c in cut)
    assert re_.label_density(cut) == 1.0


def test_첫_라벨만_자르기는_질문과_종류를_보존한다(cases):
    """자르기가 문항을 갈아치우면 '같은 문항, 라벨만 다름' 이라는 통제가 깨진다."""
    cut = re_.primary_only(cases)
    assert [c["q"] for c in cut] == [c["q"] for c in cases]
    assert [c["kind"] for c in cut] == [c["kind"] for c in cases]


def test_첫_라벨만_자르기는_원본을_건드리지_않는다(cases):
    """제자리 수정이면 같은 프로세스에서 뒤에 도는 '원본 라벨' 행이 함께 잘린다."""
    before = [len(c["expected"]) for c in cases]
    re_.primary_only(cases)
    assert [len(c["expected"]) for c in cases] == before


def test_라벨_밀도는_통제_대상이라_함께_보고된다(cases):
    """이 값을 리포트에서 빼면 recall 차이에 라벨 관대함이 섞인 걸 읽는 사람이 모른다."""
    s = {"n": 10, "recall": 0.5, "mrr": 0.29, "unknown": 0.8, "labels": 2.1}
    assert "2.1" in re_._row("실사용형", s)


def test_축_판별_실패율은_0과_1_사이다():
    assert re_.unknown_rate([]) == 0.0
    rate = re_.unknown_rate(["sl이 뭐야?", "이 함수는 코드에서 어떻게 구현돼 있어?"])
    assert 0.0 <= rate <= 1.0


def test_짧고_축을_안_밝힌_질문은_판별에_실패한다():
    """이 셋이 재려는 현상 자체. 여기가 통과로 바뀌면 문항이 더는 실사용형이 아니다."""
    assert re_.unknown_rate(["sl이 뭐야?"]) == 1.0
