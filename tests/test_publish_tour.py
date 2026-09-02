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


# --- 근거 링크 ---------------------------------------------------------------

def test_모든_판단이_근거_링크를_갖는다(summary):
    # '노트 #7' 은 처음 보는 사람에게 아무 뜻도 없는 기호다. 제목과 링크가 있어야
    # 근거 구실을 한다.
    for d in summary["decisions"]:
        assert d.get("notes"), f"{d['title']} 에 근거 링크가 없다"
        for n in d["notes"]:
            assert n["title"] and n["url"].startswith("https://")
            assert n["kind"] in ("note", "commit")
            assert n["label"]


def test_근거로_커밋도_걸_수_있다(summary):
    # 모든 측정이 엔지니어링 노트에 있지는 않다. 없는 것을 노트인 척 적으면
    # 근거를 보여주겠다는 화면이 거짓 인용을 하게 된다(실제로 한 번 그랬다).
    kinds = {n["kind"] for d in summary["decisions"] for n in d["notes"]}
    assert "commit" in kinds


def test_노트_앵커는_줄번호가_아니라_제목이다(summary):
    # GitHub 은 마크다운을 Preview 로 보여주는데 그 화면은 `#L78` 을 무시한다(실측).
    for d in summary["decisions"]:
        for n in d["notes"]:
            if n["kind"] == "note":
                assert "#L" not in n["url"], "줄 앵커는 Preview 에서 동작하지 않는다"
                assert "#" in n["url"]


def test_깃허브_앵커_규칙():
    from eval.publish import _gh_anchor

    assert _gh_anchor("7. 리랭커 — SOTA 라고 무지성으로 넣지 않는다") == \
        "7-리랭커--sota-라고-무지성으로-넣지-않는다"
    # 점·따옴표는 지우고, 하이픈·밑줄은 남긴다
    assert _gh_anchor("17. 'MMR' 이 아니라 심볼 슬롯") == "17-mmr-이-아니라-심볼-슬롯"


# --- 서사 -------------------------------------------------------------------
# 이 절들이 비면 **화면이 조용히 사실의 목록으로 되돌아간다.** 서버는 뜨고 200 도
# 나오는데 이야기만 사라지는 종류의 고장이라 눈으로 안 보면 모른다.

def test_서사가_스냅샷에_들어_있다(summary):
    st = summary.get("story") or {}
    assert st.get("intro"), "도입이 없으면 숫자부터 보게 된다"
    assert st.get("chapters"), "장이 없으면 사실의 목록으로 되돌아간다"
    assert st.get("closing")


def test_도입은_이게_무엇인지부터_말한다(summary):
    intro = summary["story"]["intro"]
    for field in ("what", "example", "twist", "thesis"):
        assert intro.get(field), f"도입에 '{field}' 가 없다"


def test_모든_장이_문제부터_말한다(summary):
    # 무슨 문제였는지 모르면 측정값을 읽을 수 없다. 이 순서가 이 페이지의 핵심이다.
    for ch in summary["story"]["chapters"]:
        for field in ("n", "title", "problem", "did", "measured", "decided"):
            assert ch.get(field), f"{ch.get('n')}장에 '{field}' 가 없다"


def test_장_번호가_1부터_이어진다(summary):
    ns = [ch["n"] for ch in summary["story"]["chapters"]]
    assert ns == list(range(1, len(ns) + 1)), f"장 번호가 끊긴다: {ns}"


def test_장이_참조하는_판단이_실제로_있다(summary):
    # 오타 하나로 카드가 조용히 사라지면 그 판단은 화면에서 없었던 일이 된다.
    titles = {d["title"] for d in summary["decisions"]}
    for ch in summary["story"]["chapters"]:
        unknown = [t for t in ch.get("decisions", []) if t not in titles]
        assert not unknown, f"{ch['n']}장이 없는 판단을 참조: {unknown}"


def test_모든_판단이_어느_장에든_실린다(summary):
    # 장에 안 실린 판단은 화면 뒤쪽 '나머지'로 빠진다. 의도한 것인지 확인해 둔다.
    shown = {t for ch in summary["story"]["chapters"] for t in ch.get("decisions", [])}
    left = [d["title"] for d in summary["decisions"] if d["title"] not in shown]
    assert not left, f"어느 장에도 안 실린 판단: {left}"


def test_장이_참조하는_근거가_실제로_있다(summary):
    known = {r["key"] for r in summary["retrieval"]} | {"judges", "hard", "ann"}
    for ch in summary["story"]["chapters"]:
        ev = ch.get("evidence")
        assert ev is None or ev in known, f"{ch['n']}장의 근거 '{ev}' 를 그릴 수 없다"


def test_모든_판단이_무슨_문제였는지_말한다(summary):
    # MEASURED 부터 시작하면 "이걸 왜 쟀지?" 를 읽는 사람이 스스로 채워야 한다.
    for d in summary["decisions"]:
        assert d.get("problem"), f"{d['title']} 에 problem 이 없다"
