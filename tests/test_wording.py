"""화면에 나가는 말에 **만든 사람만 아는 단어**가 섞이지 않는지 지킨다.

## 왜 테스트로 두는가

한 번 고쳐 놓아도 다음 기능에서 그대로 돌아온다. 실제로 그랬다 — 경로 배지에 `단일
축(unknown)` 이라고 찍히고 있었고, 그건 내부 값이 화면으로 새어 나간 것이었다.
사람이 매번 눈으로 훑어 잡을 수 있는 종류가 아니라서 규칙으로 만든다.

## 무엇을 금지어로 두는가 — 선을 어디에 긋나

**묻는 대상**은 금지어가 아니다(문서·코드·변경 이력·함수·파일). 사용자가 실제로 그것에
대해 묻고 있으므로 그 말이 없으면 오히려 설명이 안 된다.

금지어는 **구현 방식의 이름**이다(축·멀티홉·에이전트·임베딩·코사인·BM25·리랭커·청크…).
이건 시스템이 안에서 무엇을 하는지의 이름이라, 사용자가 알 이유가 없다.

## 예외 — '찾는 과정' 패널

거기서는 원래 용어를 **지우지 않고 쉬운 말 뒤에 작게** 붙인다(`.term`). 숫자를 지우면
이 제품의 차별점이 같이 사라지기 때문이다. 그래서 아래 HTML 검사는 `.term` 안에 들어간
용어가 아니라, **그 자리에 있으면 안 되는 문구**를 이름으로 콕 집어 막는다.
"""
from pathlib import Path

import pytest

from app.intent import classify, label_of
from app.profiles import REPO_ROOT, build_profile

# 구현 방식의 이름 — 사용자에게 보이는 문장에 있으면 안 된다.
JARGON = [
    "축", "멀티홉", "에이전트", "단발", "라우터", "표지어", "unknown",
    "임베딩", "코사인", "BM25", "RRF", "리랭커", "심볼 슬롯", "청크",
    "인덱싱", "인덱스", "LLM", "RAG", "토큰", "프롬프트",
]


def _offenders(text: str) -> list:
    return [w for w in JARGON if w in text]


# --- 경로 배지 · 판별 근거 ---------------------------------------------------

QUESTIONS = [
    "리랭커를 왜 기본으로 껐어?",
    "_symbol_hits 함수는 뭘 해?",
    "이 설정은 언제 왜 바뀌었어?",
    "리랭커를 왜 껐고 지금 코드는 어떻게 돼 있어?",
    "ERR7742",
    "RRF fusion is implemented how?",
    "",
]


@pytest.mark.parametrize("q", QUESTIONS)
def test_경로_설명에_만든_사람의_말이_없다(q, monkeypatch):
    from app.config import settings
    from app.router import decide

    monkeypatch.setattr(settings, "use_router", True)
    reason = decide(q).reason
    assert not _offenders(reason), f"{q!r} → {reason!r}"


@pytest.mark.parametrize("q", QUESTIONS)
def test_판별_근거에_만든_사람의_말이_없다(q):
    reason = classify(q).reason
    assert not _offenders(reason), f"{q!r} → {reason!r}"


def test_내부_이름은_화면용_이름과_분리돼_있다():
    """`unknown` 이 배지에 그대로 찍히던 사고를 막는다."""
    i = classify("ERR7742")
    assert i.name == "unknown"          # 내부 이름은 그대로 둔다(로그·평가가 쓴다)
    assert i.display_name == "전체"     # 화면용은 사람이 읽는 말


def test_축_이름은_한_곳에서만_정한다():
    """같은 것을 두 이름으로 부르는 것이 모르는 단어 하나보다 해롭다."""
    from app.rag import _scope_label

    for axis in ("doc", "code", "commit"):
        assert _scope_label(axis) == label_of(axis)
    assert label_of(None) == "전체"


# --- 답하지 못했을 때 --------------------------------------------------------

def test_거절_문구에_만든_사람의_말이_없다():
    from app.rag import OUT_OF_SCOPE, _no_hit_text

    assert not _offenders(OUT_OF_SCOPE)
    for scope in ("doc", "code", "commit", "auto"):
        assert not _offenders(_no_hit_text(scope))


def test_거절_문구는_한_곳에서만_정한다():
    """프롬프트가 같은 문장을 따로 적어 두면 화면 문구를 고칠 때 조용히 갈린다."""
    from app.rag import OUT_OF_SCOPE, SYSTEM_PROMPT

    assert OUT_OF_SCOPE in SYSTEM_PROMPT


def test_좁혀서_못_찾은_것은_다르게_말한다():
    """넓히면 답이 있다는 것을 모른 채 질문을 포기하지 않도록."""
    from app.rag import OUT_OF_SCOPE, _no_hit_text

    assert _no_hit_text("code") != OUT_OF_SCOPE
    assert "자동" in _no_hit_text("code")


# --- 첫 화면(기능 지도) ------------------------------------------------------

def test_첫_화면_설명에_만든_사람의_말이_없다():
    """추천 질문 자체는 검사하지 않는다 — 그건 사용자가 **묻는 내용**이지 화면 설명이 아니다."""
    tour = build_profile("demo").tour
    assert tour, "기능 지도가 비어 있다"
    for group in tour:
        for text in [group["title"], group["why"]]:
            assert not _offenders(text), text
        for item in group.get("items", ()):
            assert not _offenders(item["look"]), item["look"]
        follow = group.get("followup")
        if follow:
            assert not _offenders(follow["look"]), follow["look"]
            assert not _offenders(follow["hint"]), follow["hint"]


# --- 화면 파일 ---------------------------------------------------------------

# 화면에서 **지운 문구**를 이름으로 막는다. 금지어 목록으로 HTML 전체를 훑으면
# `.term` 으로 일부러 남긴 용어(코사인·BM25)까지 걸려서 쓸 수 없다.
REMOVED_FROM_UI = [
    "단발 검색", "에이전트 검색", "인덱싱된 지식원", "검색 진단 보기",
    "심볼 슬롯", "표지어", "재작성을 버리고", "커밋 이력",
    "'DOC'", "'CODE'", "'COMMIT'",
]


def _screen_text(name: str = "index.html") -> str:
    """화면 파일에서 **주석을 걷어낸** 것. 주석은 화면에 나가지 않으므로,
    거기 적힌 '예전에는 DOC 라고 썼다' 같은 설명까지 금지하면 기록을 못 남긴다."""
    import re

    html = (REPO_ROOT / "web" / name).read_text(encoding="utf-8")
    html = re.sub(r"/\*.*?\*/", " ", html, flags=re.S)      # /* ... */  (JS·CSS 공통)
    html = re.sub(r"^\s*//.*$", " ", html, flags=re.M)       # 줄 주석
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.S)      # HTML 주석
    return html


@pytest.mark.parametrize("phrase", REMOVED_FROM_UI)
def test_화면에서_지운_문구가_돌아오지_않는다(phrase):
    assert phrase not in _screen_text(), f"화면에 {phrase!r} 가 다시 들어왔다"


def test_화면과_서버가_같은_이름을_쓴다():
    """근거 표시가 화면과 서버에서 다른 말이면 같은 것을 두 이름으로 부르게 된다."""
    html = _screen_text()
    for axis in ("doc", "code", "commit"):
        assert f"'{label_of(axis)}'" in html or f"‘{label_of(axis)}’" in html \
            or f">{label_of(axis)}<" in html, label_of(axis)


def test_hidden_속성이_클래스에_밀리지_않는다():
    """`.pill`/`.btn` 이 display 를 갖고 있어 `hidden` 이 안 먹던 버그의 회귀 방지."""
    html = (REPO_ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert "[hidden] { display: none !important; }" in html


def test_화면_밝기는_사람이_고른_값만_따른다():
    """'기기 설정'(auto) 을 없앴다 — 무엇의 설정인지 화면에서 알 수 없어 되묻게 만들었다.

    되돌아오면 어두운 팔레트가 다시 두 벌이 되고, 한쪽만 고쳐 **어떤 사용자에게만**
    색이 틀어지는 고장이 함께 돌아온다.
    """
    for name in ("index.html", "eval.html"):
        html = _screen_text(name)          # 주석에 남긴 기록까지 막지는 않는다
        assert 'data-theme="auto"' not in html, f"{name} 에 auto 테마가 돌아왔다"
        assert "prefers-color-scheme" not in html, f"{name} 이 기기 설정을 다시 따라간다"


def test_어두운_팔레트는_화면마다_한_벌뿐이다():
    import re

    for name in ("index.html", "eval.html"):
        html = (REPO_ROOT / "web" / name).read_text(encoding="utf-8")
        blocks = [m for m in re.finditer(r':root\[data-theme="dark"\] \{(.*?)\}', html, re.S)
                  if "--c-bg" in m.group(1)]
        assert len(blocks) == 1, f"{name} 의 어두운 팔레트가 {len(blocks)}벌이다"


def test_두_화면이_같은_저장값을_쓴다():
    """챗봇에서 '어둡게' 를 골랐는데 측정 결과가 밝게 뜨면 같은 제품으로 읽히지 않는다."""
    chat = (REPO_ROOT / "web" / "index.html").read_text(encoding="utf-8")
    dash = (REPO_ROOT / "web" / "eval.html").read_text(encoding="utf-8")
    assert "localStorage.setItem('theme'" in chat
    assert "localStorage.getItem('theme')" in dash
    # 대시보드의 적용 조각은 <head> 안에 있어야 한다 — 아래에 두면 화면이 번쩍인다.
    assert dash.index("localStorage.getItem('theme')") < dash.index("<body>")


def test_대화_목록_글자가_감싼_요소의_색을_따른다():
    """버튼은 색을 상속하지 않는다 — 리셋을 빠뜨리면 브라우저 기본색이 그대로 나와
    라이트 모드에서 글자가 안 보인다. 실제로 그 버그가 있었다."""
    css = (REPO_ROOT / "web" / "index.html").read_text(encoding="utf-8")
    import re

    block = re.search(r"\.chat-title \{(.*?)\}", css, re.S)
    assert block, ".chat-title 규칙이 없다"
    body = block.group(1)
    for prop in ("color: inherit", "background: transparent", "border: 0", "font: inherit"):
        assert prop in body, f".chat-title 에 {prop} 가 없다"
