"""화면이 약속한 질문이 **실제로 근거를 찾는지** 고정한다.

## 왜 필요한가

첫 화면은 질문을 버튼으로 내놓는다. 그건 약속이다 — 누르면 답이 나온다는. 그런데
그 질문이 실제로 되는지는 **아무도 확인하지 않고** 있었다. 실제로 두 건이 깨져 있었다:

- `MMR 값을 왜 1.0 으로 정했어?` → 근거 5건 중 4건이 테스트 파일, 답은 엉뚱했다
- `README 에 적힌 설치 순서 알려줘` → README 를 아예 못 가져와 "설치 순서가 없다"고 답했다

둘 다 **화면에 "대신 이렇게 물으면 답합니다"라고 적어 둔 질문**이었다. 지키지 못할 약속을
화면에 적는 것은 답을 못 하는 것보다 나쁘다 — 면접에서 그 버튼을 누르는 사람이 있다.

`suggestions` 주석에 이미 같은 규칙이 적혀 있었다("코퍼스에 실제로 답이 있는 것만 넣는다").
규칙이 주석으로만 있으면 지켜지지 않는다는 것이 이번에 확인됐으므로 테스트로 옮긴다.

## 무엇을 검사하고 무엇을 검사하지 않나

- **검사한다**: 근거가 0건이 아닌 것. 그리고 1순위 근거가 **테스트 파일이 아닌 것** —
  답이 맞아도 근거 카드 첫 장이 테스트 파일이면 사람은 고장으로 읽는다.
- **검사하지 않는다**: 어느 파일이 1위여야 하는지. 그건 골든셋(`run_eval`)이 할 일이고,
  여기서 정답 파일을 박으면 **화면 문구가 평가셋이 되어** 둘이 서로를 증명하게 된다.

인덱스가 있어야 도는 통합 테스트라, 없으면 건너뛴다(clone 직후 CI 는 인덱싱 후 돈다).
"""
from __future__ import annotations

import pytest

from app.profiles import build_profile, use_profile


def _no_hit(profile) -> set:
    """화면이 **일부러** 답 못 하는 것을 보여주려고 놓아 둔 질문.

    골든셋의 범위 밖 목록과 맞춰 볼까 했는데, 같은 질문이 두 곳에 **다르게 적혀** 있었다
    (`김치찌개 맛있게 끓이는 법` vs `… 법 알려줘`). 문자열로 이어 붙이면 한쪽을 고칠 때
    조용히 끊어지므로, 화면 데이터가 **직접 표시**하게 한다(`"expect": "no-hit"`).
    """
    items = [it for g in profile.tour for it in g.get("items", ())] + list(profile.limits)
    return {it["q"] for it in items if it.get("expect") == "no-hit"}


def _promised(profile) -> list:
    """화면이 내놓는 질문 중 **답이 나와야 하는** 것 — 지도 · 추천 · '대신 이렇게'."""
    qs = [it["q"] for g in profile.tour for it in g.get("items", ())]
    qs += [x["ask"] for x in profile.limits if x.get("ask")]
    qs += list(profile.suggestions)
    skip = _no_hit(profile)
    seen, uniq = set(), []
    for q in qs:
        if q not in seen and q not in skip:
            seen.add(q)
            uniq.append(q)
    return uniq


@pytest.fixture(scope="module")
def search_fn():
    use_profile("demo")
    from app.ingest import get_vectorstore
    from app.retriever import search

    try:
        if get_vectorstore()._collection.count() == 0:
            pytest.skip("demo 인덱스가 비어 있다 — python -m app.ingest --profile demo")
    except Exception as e:  # chroma 없음 등
        pytest.skip(f"검색을 돌릴 수 없다: {e}")
    return search


@pytest.fixture(scope="module")
def demo():
    return build_profile("demo")


def test_화면이_내놓는_질문은_전부_근거를_찾는다(demo, search_fn):
    broken = []
    for q in _promised(demo):
        docs, _ = search_fn(q)
        if not docs:
            broken.append(q)
    assert not broken, f"눌러도 근거가 0건인 질문: {broken}"


def test_1순위_근거가_테스트_파일이면_안_된다(demo, search_fn):
    """테스트는 '왜 그렇게 정했나'의 근거가 아니다. 그 자리에 오면 근거 카드가 이상해 보인다."""
    bad = []
    for q in _promised(demo):
        docs, _ = search_fn(q)
        top = docs[0].metadata.get("source", "") if docs else ""
        if top.startswith("tests/"):
            bad.append((q, top))
    assert not bad, f"1순위 근거가 테스트 파일: {bad}"


def test_화면_문구_파일은_근거로_나오지_않는다(demo, search_fn):
    """질문 문구가 적힌 파일이 그 질문의 근거가 되면 자기 자신을 답으로 내놓는 꼴이다."""
    hits = []
    for q in _promised(demo):
        docs, _ = search_fn(q)
        srcs = [d.metadata.get("source", "") for d in docs]
        if "app/screen_copy.py" in srcs:
            hits.append((q, srcs))
    assert not hits, f"화면 문구 파일이 근거로 올라왔다: {hits}"


def test_범위_밖_시연_질문은_근거가_0건이다(demo, search_fn):
    """화면이 '눌러 보시면 멈추는 것이 보입니다'라고 약속한 자리. 답이 나오면 그 설명이 거짓이 된다."""
    shown = [it["q"] for g in demo.tour for it in g.get("items", ())]
    shown += [x["q"] for x in demo.limits]
    demo_q = [q for q in shown if q in _no_hit(demo)]
    assert demo_q, "범위 밖을 보여주는 항목이 없다"
    for q in demo_q:
        docs, _ = search_fn(q)
        assert not docs, f"{q!r} 이 근거를 {len(docs)}건 찾았다 — 멈춘다는 설명과 어긋난다"
