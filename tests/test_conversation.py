"""멀티턴 대화 처리의 순수 로직 테스트(LLM 호출 없음)."""
import pytest

from app.conversation import (
    MAX_HISTORY_TURNS,
    MAX_TURN_CHARS,
    Turn,
    format_history,
    looks_dependent,
    parse_history,
    rewrite_query,
)


# --- 히스토리 정규화 --------------------------------------------------------

def test_역할과_내용이_온전한_턴만_통과시킨다():
    raw = [
        {"role": "user", "content": "안녕"},
        {"role": "system", "content": "무시돼야 함"},   # 허용 역할 아님
        {"role": "assistant", "content": ""},           # 빈 내용
        "문자열",                                        # dict 아님
        {"role": "assistant", "content": "네"},
    ]
    assert [(t.role, t.content) for t in parse_history(raw)] == [("user", "안녕"), ("assistant", "네")]


def test_오래된_턴을_잘라_최근_대화만_남긴다():
    raw = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    turns = parse_history(raw)
    assert len(turns) == MAX_HISTORY_TURNS
    assert turns[-1].content == "q19"  # 최근 것이 남아야 한다


def test_긴_답변은_잘라서_싣는다():
    """답변 전문을 실으면 재작성 프롬프트가 '답변 요약'이 되어 질문을 놓친다."""
    turns = parse_history([{"role": "assistant", "content": "가" * 5000}])
    assert len(turns[0].content) == MAX_TURN_CHARS


def test_히스토리가_없으면_빈_리스트():
    assert parse_history(None) == [] and parse_history([]) == []


# --- 의존성 휴리스틱 --------------------------------------------------------

@pytest.mark.parametrize("q", [
    "그건 왜 그래?", "그럼 어떻게 측정했어?", "더 자세히", "왜?",
    "거기서 뭘 바꿨어?", "이 방식의 단점은?", "방금 말한 거 다시",
])
def test_앞_맥락이_필요한_질문을_잡아낸다(q):
    assert looks_dependent(q)


@pytest.mark.parametrize("q", [
    "하이브리드 검색은 어떻게 구현돼 있어?",
    "코퍼스 프로필을 분리한 이유가 뭐야?",
    "평가 하네스는 무엇을 측정하나요?",
])
def test_독립적인_질문은_재작성을_건너뛴다(q):
    assert not looks_dependent(q)


def test_짧은_질문은_보수적으로_의존으로_본다():
    """놓쳐서 재작성하면 손해가 없지만, 필요한데 건너뛰면 검색이 실패한다 → 관대하게."""
    assert looks_dependent("왜?")
    assert looks_dependent("언제부터야")


# --- 재작성 분기 (LLM 미호출 경로) ------------------------------------------

def test_첫_질문이면_LLM을_부르지_않는다():
    q = "그건 왜 그래?"  # 의존적이지만 맥락이 없다
    out, info = rewrite_query(q, [])
    assert out == q and info["skipped"] == "first_turn" and info["applied"] is False


def test_독립적인_질문이면_히스토리가_있어도_건너뛴다():
    q = "하이브리드 검색은 어떻게 구현돼 있어?"
    out, info = rewrite_query(q, [Turn("user", "앞 질문"), Turn("assistant", "앞 답변")])
    assert out == q and info["skipped"] == "standalone"


def test_LLM이_없으면_원문으로_검색한다(monkeypatch):
    """대화 기능 때문에 기존 단발 경로가 죽으면 안 된다."""
    import app.conversation as C

    monkeypatch.setattr(C.settings, "llm_provider", "extractive")
    monkeypatch.setattr(C.settings, "google_api_key", "")
    q = "그건 왜?"
    out, info = rewrite_query(q, [Turn("user", "a"), Turn("assistant", "b")])
    assert out == q and info["skipped"] == "no_llm"


def test_재작성_실패해도_예외를_밖으로_던지지_않는다(monkeypatch):
    import app.conversation as C

    monkeypatch.setattr(C.settings, "llm_provider", "gemini")
    monkeypatch.setattr(C.settings, "google_api_key", "dummy-key")

    def boom():
        raise RuntimeError("모델 호출 실패")

    import app.rag
    monkeypatch.setattr(app.rag, "_llm", boom)

    q = "그건 왜?"
    out, info = rewrite_query(q, [Turn("user", "a"), Turn("assistant", "b")])
    assert out == q and str(info["skipped"]).startswith("error:")


# --- 포맷 -------------------------------------------------------------------

def test_히스토리_포맷은_역할을_한국어로_표기한다():
    text = format_history([Turn("user", "질문"), Turn("assistant", "답변")])
    assert text == "사용자: 질문\n어시스턴트: 답변"
