"""근거 원문 조회 테스트.

두 가지를 고정한다.

1. **허용목록을 뚫을 수 없다.** 이 엔드포인트는 사용자가 준 문자열로 파일을 여는 자리라,
   여기가 뚫리면 서버가 읽을 수 있는 모든 파일이 공개된다. 경로 탈출의 흔한 모양들을
   회귀 테스트로 박아 둔다 — 나중에 "편의를 위해" 검사 방식을 바꾸는 순간 여기가 붉어진다.
2. **강조 범위가 실제 줄을 가리킨다.** 라인 번호는 청크에 없고 조회 시점에 다시 찾는다
   (app/source_view.py 참고). 그 계산이 틀리면 화면은 멀쩡한데 엉뚱한 줄이 강조되고,
   그건 근거가 없는 것보다 나쁘다.
"""
import pytest

from app import source_view as sv


@pytest.fixture(autouse=True)
def demo_profile():
    """이 저장소 자기 자신을 코퍼스로 쓰는 demo 프로필로 고정.

    `.env` 가 private 을 가리키는 개발 PC 에서도 같은 결과가 나와야 한다.
    끝나면 원래 프로필로 되돌린다 — 여기서 바꾼 값이 남으면 뒤에 도는 테스트가
    엉뚱한 코퍼스를 본다.
    """
    from app.profiles import active_profile, use_profile

    before = active_profile().name
    use_profile("demo")
    yield
    use_profile(before)


# --- 허용목록 --------------------------------------------------------------

ESCAPES = [
    "../.env",
    "..\\.env",
    "app/rag.py/../../.env",
    "/etc/passwd",
    "C:\\Windows\\win.ini",
    "app/../app/rag.py",        # 정규화하면 유효하지만 **표시이름이 아니다** → 거부
    "",
    "app/nonexistent.py",
]


@pytest.mark.parametrize("ref", ESCAPES)
def test_허용목록_밖은_열리지_않는다(ref):
    with pytest.raises(sv.SourceNotFound):
        sv.read_source(ref)


def test_인덱싱된_파일만_목록에_있다():
    refs = sv.indexed_refs()
    assert "app/rag.py" in refs
    assert ".env" not in refs
    assert all(not r.startswith("..") for r in refs)


# --- 강조 범위 --------------------------------------------------------------

SAMPLE = '''"""모듈 도크스트링."""


def alpha():
    return 1


class Box:
    def beta(self):
        return 2


def gamma():
    return 3
'''


def test_코드_심볼의_줄범위():
    assert sv._code_span(SAMPLE, "alpha") == (4, 5)
    assert sv._code_span(SAMPLE, "gamma") == (13, 14)


def test_클래스_안의_메서드는_그_클래스_안에서_찾는다():
    """`Box.beta` 를 전역에서 찾으면 같은 이름의 다른 메서드를 짚을 수 있다."""
    assert sv._code_span(SAMPLE, "Box.beta") == (9, 10)
    assert sv._code_span(SAMPLE, "Box") == (8, 10)


def test_없는_심볼이면_강조하지_않는다():
    assert sv._code_span(SAMPLE, "없는함수") is None
    assert sv._code_span(SAMPLE, "") is None
    assert sv._code_span("def (", "alpha") is None       # 파싱 실패해도 죽지 않는다


MD = """# 제목

본문 1

## 절 A

본문 2
본문 3

## 절 B

본문 4
"""


def test_문서_헤더의_절_범위():
    assert sv._doc_span(MD, "절 A") == (5, 9)            # 헤더 줄 ~ 다음 동급 헤더 직전
    assert sv._doc_span(MD, "절 B") == (10, 12)
    assert sv._doc_span(MD, "없는 절") is None


def test_실제_파일을_열면_강조가_그_심볼을_가리킨다():
    got = sv.read_source("app/rag.py", "answer")
    lines = got["text"].splitlines()
    start = got["highlight"]["start"]
    assert lines[start - 1].startswith("def answer(")
    assert got["doc_type"] == "code"
    assert got["language"] == "python"


def test_커밋_근거는_메시지와_변경파일을_준다():
    import subprocess

    short = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
    got = sv.read_source(f"git:{short}")
    assert got["doc_type"] == "commit"
    assert short in got["display"]
    assert "작성자" in got["text"]


def test_없는_커밋은_404_로_이어진다():
    with pytest.raises(sv.SourceNotFound):
        sv.read_source("git:0000000")


# --- 원격 링크 --------------------------------------------------------------

def test_원격링크는_브랜치가_아니라_커밋을_가리킨다():
    """브랜치를 박으면 브랜치가 움직인 뒤 링크가 다른 줄을 가리킨다."""
    got = sv.read_source("app/rag.py", "answer")
    url = got["remote_url"]
    if url is None:
        pytest.skip("원격이 없는 환경")
    assert "/blob/" in url and "#L" in url
    assert "/blob/main/" not in url
