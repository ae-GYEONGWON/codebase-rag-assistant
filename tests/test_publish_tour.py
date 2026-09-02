"""발행 스냅샷과 기능 지도 테스트.

이 둘이 깨지면 **화면이 조용히 빈다.** 서버는 뜨고 200 도 나오는데 내용만 없는 종류의
고장이라, 눈으로 보지 않으면 모른다. 그래서 테스트로 고정한다.
"""
import json
from pathlib import Path

import pytest

from app.profiles import active_profile, available_profiles, use_profile
from eval import publish

PUBLISHED = Path(publish.SUMMARY)


# --- 발행 스냅샷 -------------------------------------------------------------

def test_스냅샷이_저장소에_들어_있다():
    # eval/reports/ 는 git 제외라 clone 직후 비어 있다. 스냅샷까지 없으면
    # 보러 온 사람 화면에서 대시보드가 빈 페이지가 된다.
    assert PUBLISHED.exists(), "eval/published/summary.json 이 없다 — python -m eval.publish"


@pytest.fixture(scope="module")
def summary():
    return json.loads(PUBLISHED.read_text(encoding="utf-8"))


def test_대시보드가_그리는_절이_모두_들어_있다(summary):
    for key in ("retrieval", "judges", "hard", "ann", "decisions"):
        assert summary.get(key), f"'{key}' 가 비어 있으면 그 절이 통째로 안 그려진다"


def test_스냅샷임을_알_수_있게_생성시각이_있다(summary):
    # 지금 다시 잰 값이 아닐 수 있다는 것을 화면에 밝히려면 이 값이 필요하다.
    assert summary.get("generated_at")


def test_검색_리포트에_운영_파이프라인_행이_있다(summary):
    # 대시보드가 이 행을 강조해 "무엇이 실제 경로인지" 보여준다.
    for r in summary["retrieval"]:
        names = {row["name"] for s in r["suites"] for row in s["rows"]}
        assert "운영 파이프라인" in names, f"{r['key']} 에 운영 파이프라인 행이 없다"


def test_판정기_비교에_kappa가_들어_있다(summary):
    # 이 프로젝트의 핵심 주장(평균은 모였는데 합의는 없다)이 이 값에 달려 있다.
    pairs = summary["judges"]["pairs"]
    assert pairs and any(p.get("kappa") is not None for p in pairs)


def test_기각한_판단이_채택한_것과_함께_실린다(summary):
    verdicts = [d["verdict"] for d in summary["decisions"]]
    assert any("기각" in v for v in verdicts), "기각 사례가 없으면 목록의 의미가 없다"


def test_모든_판단이_잰_값과_행동을_갖는다(summary):
    # '해봤다'가 아니라 '재고 정했다'가 이 목록의 존재 이유다.
    for d in summary["decisions"]:
        for field in ("title", "verdict", "measured", "why", "action", "note"):
            assert d.get(field), f"{d.get('title')} 에 '{field}' 가 비어 있다"


def test_적대적_평가는_실패_사례를_숨기지_않는다(summary):
    h = summary["hard"]
    assert h["pass_rate"] < 1.0, "통과율이 100% 면 실패 사례가 없어 보여줄 것이 없다"
    assert h["failures"], "통과하지 못한 문항이 있는데 목록이 비었다"


# --- 기능 지도 ---------------------------------------------------------------

@pytest.mark.parametrize("name", [p for p in available_profiles() if p in ("demo", "eval")])
def test_기능_지도가_비어_있지_않다(name):
    use_profile(name)
    assert active_profile().tour, f"{name} 프로필에 기능 지도가 없다"


@pytest.mark.parametrize("name", [p for p in available_profiles() if p in ("demo", "eval")])
def test_모든_항목이_무엇을_보게_되는지를_말한다(name):
    # 질문만 있고 'look' 이 없으면 예전의 평평한 칩과 다를 게 없다 —
    # 무엇을 물을 수 있는지는 알아도 무엇이 구현돼 있는지는 모른다.
    use_profile(name)
    for g in active_profile().tour:
        assert g.get("title") and g.get("why")
        assert g.get("items"), f"'{g['title']}' 에 질문이 없다"
        for it in g["items"]:
            assert it.get("q") and it.get("look"), f"'{it.get('q')}' 에 look 이 없다"


def test_지도의_질문이_추천질문으로_파생된다():
    use_profile("demo")
    p = active_profile()
    qs = p.tour_questions()
    assert len(qs) == sum(len(g["items"]) for g in p.tour)
