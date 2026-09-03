"""대화 공유 저장소 · 검색 범위 배선 테스트.

공유는 **남이 쓴 내용을 내 도메인에서 다시 재생하는** 통로라, 저장되는 모양을 서버가
못 박는지 여기서 고정한다. 검색 범위는 UI 스위치 → 검색기까지 값이 실제로 도달하는지를
본다 — 화면에만 있고 서버에 닿지 않는 스위치는 있는 것보다 나쁘다(있다고 믿게 만든다).
"""
import pytest

from app import share


@pytest.fixture(autouse=True)
def temp_store(tmp_path, monkeypatch):
    """저장 위치를 임시 디렉터리로. 테스트가 실제 공유본을 남기면 안 된다."""
    monkeypatch.setattr(share, "STORE_DIR", tmp_path / "share_store")


# --- 스키마 강제 ------------------------------------------------------------

def test_모르는_필드는_저장되지_않는다():
    sid = share.save({"title": "t", "turns": [
        {"role": "user", "content": "질문"},
        {"role": "assistant", "content": "답", "onerror": "<img>", "cookies": "x"},
    ]})
    got = share.load(sid)
    assert set(got["turns"][1]) == {
        "role", "content", "sources", "retrieval", "rewrite", "route", "mode", "scope"}


def test_근거와_검색진단은_남긴다():
    """이 둘이 빠지면 공유 링크가 여느 챗봇 캡처와 다를 게 없어진다."""
    sid = share.save({"title": "t", "turns": [{
        "role": "assistant", "content": "답[1]",
        "sources": [{"source": "app/rag.py", "section": "answer",
                     "doc_type": "code", "snippet": "발췌"}],
        "retrieval": {"reason": "hit", "best_similarity": 0.71,
                      "picked": [{"source": "app/rag.py", "similarity": 0.71, "bm25": 3.2}]},
    }]})
    turn = share.load(sid)["turns"][0]
    assert turn["sources"][0]["source"] == "app/rag.py"
    assert turn["retrieval"]["best_similarity"] == 0.71
    assert turn["retrieval"]["picked"][0]["bm25"] == 3.2


def test_알_수_없는_역할은_거부한다():
    with pytest.raises(share.ShareError):
        share.save({"turns": [{"role": "system", "content": "x"}]})


def test_빈_대화와_너무_긴_대화는_거부한다():
    with pytest.raises(share.ShareError):
        share.save({"turns": []})
    with pytest.raises(share.ShareError):
        share.save({"turns": [{"role": "user", "content": "x"}] * (share.MAX_TURNS + 1)})


def test_너무_큰_대화는_거부한다():
    with pytest.raises(share.ShareError):
        share.save({"turns": [{"role": "user", "content": "가" * 40_000}] * 20})


# --- id ---------------------------------------------------------------------

def test_id_는_내용에서_계산되지_않는다():
    """내용 해시면 내용을 아는 사람이 링크를 계산할 수 있다."""
    payload = {"title": "같은 대화", "turns": [{"role": "user", "content": "동일"}]}
    assert share.save(payload) != share.save(payload)


@pytest.mark.parametrize("bad", ["../../etc/passwd", "..", "short", "a" * 40, "", "abc/def"])
def test_형식이_틀린_id_는_파일을_건드리지_않는다(bad):
    with pytest.raises(share.ShareNotFound):
        share.load(bad)


def test_없는_링크():
    with pytest.raises(share.ShareNotFound):
        share.load("A" * 16)


# --- 검색 범위 --------------------------------------------------------------

from app import rag  # noqa: E402  (프로필 픽스처보다 뒤에 와도 되는 순수 로직)


@pytest.mark.parametrize("scope,expected", [
    ("doc", ("doc",)), ("code", ("code",)), ("commit", ("commit",)),
    ("auto", None), (None, None), ("이상한값", None),
])
def test_범위값이_검색축으로_바뀐다(scope, expected):
    assert rag._axis_of(scope) == expected


def test_범위를_고정하면_에이전트로_가지_않는다():
    """축이 하나뿐인데 에이전트를 태우면 LLM 3~5회를 쓰고 단발과 같은 답을 낸다."""
    from app.router import Route

    agent = Route("agent", "멀티홉으로 판정됨")
    assert rag._scoped_route(agent, "code").mode == "single"
    assert rag._scoped_route(agent, "auto").mode == "agent"      # 자동일 때는 손대지 않는다


def test_범위를_좁혀_못_찾은_것과_코퍼스에_없는_것을_구분한다():
    narrowed = rag._no_hit_text("commit")
    assert "커밋 이력" in narrowed and "자동" in narrowed
    assert rag._no_hit_text("auto") == rag.OUT_OF_SCOPE


def test_범위가_검색기까지_도달한다(monkeypatch):
    """화면에만 있고 서버에 닿지 않는 스위치를 막는다."""
    seen = {}

    def fake_search(query, doc_types=None):
        seen["doc_types"] = doc_types
        return [], {"reason": "out_of_scope"}

    monkeypatch.setattr(rag, "search", fake_search)
    out = rag.answer("아무 질문", scope="commit")
    assert seen["doc_types"] == ("commit",)
    assert out["mode"] == "no_hit"
    assert "커밋 이력" in out["answer"]
