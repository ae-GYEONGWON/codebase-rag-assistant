"""판정기 패널의 순수 통계 로직 테스트.

LLM 호출부는 제외하고, **결론을 좌우하는 계산**만 고정한다. 특히 중요한 것은
"차이가 없다"를 잘못 말하지 않는 장치들이다 — 부트스트랩 구간이 0 을 걸치는지,
한쪽이 전부 동점일 때 상관·kappa 가 숫자를 지어내지 않고 None 을 내는지.
"""
import json

import pytest

from eval.judge_panel import (
    MIN_SIGNIFICANT_N, _bootstrap_ci, _kappa, _spearman, compare, render,
)


# --- Spearman ---------------------------------------------------------------

def test_순위가_같으면_상관이_1():
    assert _spearman([0.1, 0.5, 0.9], [0.2, 0.6, 0.95]) == pytest.approx(1.0)


def test_순위가_뒤집히면_상관이_마이너스1():
    assert _spearman([0.1, 0.5, 0.9], [0.9, 0.5, 0.1]) == pytest.approx(-1.0)


def test_한쪽이_전부_동점이면_순위가_없으므로_None():
    # 판정기가 모든 문항에 1.0 을 주는 일은 실제로 흔하다. 이때 상관계수를
    # 0 으로 보고하면 "무관하다"는 잘못된 결론이 된다 → 정의되지 않음으로 남긴다.
    assert _spearman([1.0, 1.0, 1.0], [0.2, 0.6, 0.9]) is None


# --- Cohen's kappa ----------------------------------------------------------

def test_완전_일치하고_판정이_갈리면_kappa_1():
    a = [True, False, True, False]
    assert _kappa(a, list(a)) == pytest.approx(1.0)


def test_완전_불일치면_kappa_음수():
    assert _kappa([True, True, False, False], [False, False, True, True]) < 0


def test_한쪽이_전부_같은_판정이면_kappa는_None():
    # 기대 일치가 1 이 되어 분모가 0. 우연 초과 일치를 논할 수 없다.
    assert _kappa([True, True, True], [True, False, True]) is None


def test_우연_수준의_일치는_kappa가_0_근처():
    a = [True, False, True, False, True, False, True, False]
    b = [True, True, False, False, True, True, False, False]
    assert abs(_kappa(a, b)) < 1e-9


# --- 부트스트랩 신뢰구간 ----------------------------------------------------

def test_차이가_모두_0이면_구간도_0():
    lo, hi = _bootstrap_ci([0.0] * 12)
    assert lo == 0.0 and hi == 0.0


def test_섞인_차이는_구간이_0을_걸친다():
    lo, hi = _bootstrap_ci([0.3, -0.3, 0.2, -0.2, 0.1, -0.1] * 2)
    assert lo <= 0 <= hi


def test_한쪽으로_쏠린_차이는_구간이_0을_안_걸친다():
    lo, hi = _bootstrap_ci([-0.3, -0.25, -0.35, -0.28, -0.31, -0.27] * 3)
    assert hi < 0


# --- compare 통합 -----------------------------------------------------------

def _judge_file(tmp_path, tag, judge, scores, self_judge=False):
    p = tmp_path / f"panel-judge-{tag}.json"
    p.write_text(json.dumps({
        "judge": judge, "generator": "gemini:gen", "self_judge": self_judge,
        "freeze_seed": 1, "profile": "eval", "n": len(scores), "n_unparsed": 0,
        "mean_score": sum(scores.values()) / len(scores),
        "scores": [{"id": k, "q": f"질문 {k}", "bucket": "in_scope",
                    "score": v, "unsupported": []} for k, v in scores.items()],
    }, ensure_ascii=False), encoding="utf-8")
    return p


def test_공통_문항만_비교한다(tmp_path):
    # 쿼터가 끊겨 한쪽이 덜 채점된 상황. 겹치는 문항만 써야 통제 비교가 유지된다.
    a = _judge_file(tmp_path, "self", "gemini:gen",
                    {"q01": 1.0, "q02": 1.0, "q03": 1.0, "q04": 1.0, "q05": 1.0}, True)
    b = _judge_file(tmp_path, "cross", "gemini:other", {"q01": 0.8, "q02": 0.6, "q03": 0.7})
    res = compare([a, b], threshold=1.0)
    assert res["pairs"][0]["n_common"] == 3
    assert res["pairs"][0]["mean_diff"] == pytest.approx(-0.3)
    # 판정기별 평균은 자기가 채점한 것 전부로 낸다(교집합으로 깎지 않는다)
    assert [j["n"] for j in res["judges"]] == [5, 3]


def test_쌍마다_자기_교집합을_쓴다(tmp_path):
    # RAGAS 는 근거 없는 범위밖 문항(q05)을 채점하지 못한다. 전체 교집합을 쓰면
    # 그 한 문항이 self↔cross 비교에서까지 빠진다 — 쌍별 교집합이어야 하는 이유.
    ids = {f"q{i:02d}": 1.0 for i in range(1, 6)}
    a = _judge_file(tmp_path, "self", "gemini:gen", ids, True)
    b = _judge_file(tmp_path, "cross", "gemini:other", {k: 0.9 for k in ids})
    c = _judge_file(tmp_path, "ragas", "ragas:x", {k: 0.8 for k in list(ids)[:4]})
    res = compare([a, b, c], threshold=1.0)
    by = {(p["a"], p["b"]): p["n_common"] for p in res["pairs"]}
    assert by[("gemini:gen", "gemini:other")] == 5
    assert by[("gemini:gen", "ragas:x")] == 4


def test_표본이_모자란_쌍은_건너뛰고_이유를_남긴다(tmp_path):
    # 쿼터로 2건만 채점된 판정기. 조용히 빼지 않고 왜 빠졌는지 리포트에 적는다.
    a = _judge_file(tmp_path, "self", "gemini:gen", {f"q{i:02d}": 1.0 for i in range(1, 6)}, True)
    b = _judge_file(tmp_path, "cross", "gemini:other", {f"q{i:02d}": 0.9 for i in range(1, 6)})
    c = _judge_file(tmp_path, "quota", "gemini:pricey", {"q01": 0.5, "q02": 0.5})
    res = compare([a, b, c], threshold=1.0)
    assert len(res["pairs"]) == 1
    assert len(res["skipped_pairs"]) == 2
    assert "gemini:pricey" in render(res)


def test_self_judge가_후하면_음의_차이로_나온다(tmp_path):
    a = _judge_file(tmp_path, "self", "gemini:gen", {f"q{i:02d}": 1.0 for i in range(1, 9)}, True)
    b = _judge_file(tmp_path, "cross", "gemini:other", {f"q{i:02d}": 0.7 for i in range(1, 9)})
    res = compare([a, b], threshold=1.0)
    p = res["pairs"][0]
    assert p["mean_diff"] == pytest.approx(-0.3)
    assert p["significant"] is True           # 전부 같은 방향 → 구간이 0 을 안 걸친다
    assert res["judges"][0]["flag_rate"] == 0.0
    assert res["judges"][1]["flag_rate"] == 1.0
    assert p["n_common"] == 8


def test_채점_실패_문항은_비교에서_빠진다(tmp_path):
    a = _judge_file(tmp_path, "self", "gemini:gen", {"q01": 1.0, "q02": 1.0}, True)
    b = tmp_path / "panel-judge-cross.json"
    b.write_text(json.dumps({
        "judge": "gemini:other", "generator": "gemini:gen", "self_judge": False,
        "freeze_seed": 1, "profile": "eval", "n": 1, "n_unparsed": 1, "mean_score": 0.5,
        "scores": [{"id": "q01", "q": "질문 q01", "bucket": "in_scope", "score": 0.5,
                    "unsupported": []},
                   {"id": "q02", "q": "질문 q02", "bucket": "in_scope", "score": None,
                    "unsupported": []}],
    }, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit):           # 공통 1건 → 비교 불가로 막는다
        compare([a, b], threshold=1.0)


def test_표본이_작으면_구간이_0을_안_걸쳐도_유의라고_안_한다(tmp_path):
    # 큰 모델은 무료 티어에서 하루 4~5회로 끊긴다. 그 표본으로도 구간은 0 을 안 걸칠 수
    # 있지만, 그건 통계가 아니라 우연이다. 버리지는 않되 유의성은 주장하지 않는다.
    n = MIN_SIGNIFICANT_N - 1
    ids = {f"q{i:02d}": 1.0 for i in range(1, n + 1)}
    a = _judge_file(tmp_path, "self", "gemini:gen", ids, True)
    b = _judge_file(tmp_path, "cross", "gemini:other", {k: 0.5 for k in ids})
    p = compare([a, b], threshold=1.0)["pairs"][0]
    lo, hi = p["ci95"]
    assert not (lo <= 0 <= hi)        # 구간 자체는 0 을 안 걸친다
    assert p["underpowered"] is True
    assert p["significant"] is False  # 그래도 유의라고 말하지 않는다


def test_표본이_충분하면_유의로_표시한다(tmp_path):
    ids = {f"q{i:02d}": 1.0 for i in range(1, MIN_SIGNIFICANT_N + 1)}
    a = _judge_file(tmp_path, "self", "gemini:gen", ids, True)
    b = _judge_file(tmp_path, "cross", "gemini:other", {k: 0.5 for k in ids})
    p = compare([a, b], threshold=1.0)["pairs"][0]
    assert p["underpowered"] is False and p["significant"] is True


def test_리포트에_판정기와_한계가_함께_적힌다(tmp_path):
    a = _judge_file(tmp_path, "self", "gemini:gen", {f"q{i:02d}": 1.0 for i in range(1, 5)}, True)
    b = _judge_file(tmp_path, "cross", "gemini:other", {f"q{i:02d}": 0.9 for i in range(1, 5)})
    md = render(compare([a, b], threshold=1.0))
    assert "gemini:other" in md and "self" in md and "cross" in md
    assert "한계" in md          # 보정값만 인용되지 않도록 한계를 항상 붙인다
