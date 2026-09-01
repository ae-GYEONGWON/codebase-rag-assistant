"""평가 리포트 · 회귀 게이트 로직 테스트.

게이트가 조용히 망가지면 회귀를 놓치고도 초록불이 뜬다 — 그래서 게이트 자체를 고정한다.
인덱스·임베딩이 필요 없는 순수 로직이라 CI 에서 그대로 돈다.
"""
import json

import pytest

from eval import report as rp


def _report(recall=0.9, mrr=0.72, misses=None, rejected=6, total=6, profile="demo"):
    r = rp.new_report(
        profile=profile, collection="c", k=5,
        corpus={"doc": 45, "code": 211, "commit": 29},
        dataset={"path": "questions.demo.json", "origins": {"manual": 30}},
    )
    suite = rp.Suite(title="전체", n=20)
    suite.rows.append(rp.RetrieverRow("vector only", 0.85, 0.61, []))
    suite.rows.append(rp.RetrieverRow(rp.PRIMARY_ROW, recall, mrr, list(misses or [])))
    r.suites.append(suite)
    r.out_of_scope_total, r.out_of_scope_rejected = total, rejected
    return r


def test_같은_결과면_회귀_없음():
    assert rp.compare(_report(), _report()) == []


def test_recall_하락은_회귀로_잡는다():
    problems = rp.compare(_report(recall=0.90), _report(recall=0.80))
    assert any("recall" in p for p in problems)


def test_허용치_안의_흔들림은_통과시킨다():
    """부동소수점·동점 순서 차이로 한 문항이 흔들리는 것까지 실패로 보면 게이트가 못 쓰게 된다."""
    assert rp.compare(_report(recall=0.90), _report(recall=0.895)) == []


def test_대조군_행_하락은_게이트를_막지_않는다():
    """vector only 는 비교용이다 — 서비스 품질과 무관하므로 게이트 대상이 아니다."""
    base, cur = _report(), _report()
    cur.suites[0].rows[0].recall = 0.10  # vector only 를 폭락시킴
    assert rp.compare(base, cur) == []


def test_새로_실패한_문항을_이름까지_보고한다():
    problems = rp.compare(_report(misses=["A"]), _report(misses=["A", "B"]))
    assert any("B" in p for p in problems)


def test_기존_실패가_유지되는_것은_회귀가_아니다():
    assert rp.compare(_report(misses=["A"]), _report(misses=["A"])) == []


def test_범위밖_거절률_하락도_회귀다():
    problems = rp.compare(_report(rejected=6), _report(rejected=4))
    assert any("범위 밖" in p for p in problems)


def test_프로필이_다르면_비교하지_않고_실패시킨다():
    """private baseline 에 demo 결과를 대면 전부 회귀로 보인다 — 비교 자체가 무의미."""
    problems = rp.compare(_report(profile="private"), _report(profile="demo"))
    assert len(problems) == 1 and "프로필 불일치" in problems[0]


def test_json_왕복(tmp_path):
    r = _report(misses=["질문 하나"])
    p = r.write_json(tmp_path / "x.json")
    back = rp.load(p)
    assert back.profile == r.profile
    assert back.suite("전체").row(rp.PRIMARY_ROW).misses == ["질문 하나"]
    assert rp.compare(back, r) == []


def test_markdown_에_운영행과_코퍼스가_찍힌다():
    md = _report().to_markdown()
    assert rp.PRIMARY_ROW in md and "285청크" in md and "manual" in md
