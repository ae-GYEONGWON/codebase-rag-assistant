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


# --- 재작성이 사실을 지어내는 것 막기 (2026-09-02, 데모에서 실제로 발생) ------

from app.conversation import invented_facts


def _t(role, content):
    return Turn(role, content)


def test_없던_숫자를_넣으면_지어낸_것으로_본다():
    # 실제 사례: "MMR 계수를 왜 그렇게 정했고" → "MMR 계수를 왜 0.5로 설정했고"
    # 재작성된 질의로 검색하므로, 이건 문장이 어색한 정도가 아니라 검색을 망친다.
    turns = [_t("user", "리랭커를 왜 껐어?"), _t("assistant", "코드 질문 정확도가 떨어져서요.")]
    got = invented_facts("MMR 계수를 왜 0.5로 설정했어?", "MMR 계수를 왜 그렇게 정했어?", turns)
    assert "0.5" in got


def test_대화에_있던_숫자는_지어낸_것이_아니다():
    turns = [_t("assistant", "mmr_lambda 는 1.0 으로 두었습니다.")]
    assert invented_facts("mmr_lambda 를 왜 1.0 으로 뒀어?", "그걸 왜 그렇게 뒀어?", turns) == []


def test_표기가_흔들려도_정수부가_있으면_통과():
    # "5건" → "5.0" 처럼 모델이 표기를 바꾸는 것까지 창작으로 보면 과잉 차단이 된다.
    turns = [_t("assistant", "상위 5건을 봅니다.")]
    assert invented_facts("상위 5.0건이 뭐야?", "그게 뭐야?", turns) == []


def test_없던_식별자를_넣으면_지어낸_것으로_본다():
    turns = [_t("assistant", "리랭커는 껐습니다.")]
    assert "app/nonexistent.py" in invented_facts(
        "app/nonexistent.py 에서 왜 껐어?", "그걸 왜 껐어?", turns)


def test_대화에_있던_식별자로_치환하는_것이_재작성의_본래_일이다():
    turns = [_t("assistant", "app/retriever.py 의 search 함수에서 처리합니다.")]
    assert invented_facts("app/retriever.py 의 search 함수는 뭘 해?", "그건 뭘 해?", turns) == []


class _FakeLLM:
    """재작성 모델을 대신한다 — 정해진 문자열만 돌려준다(네트워크·쿼터 없이 경로를 고정)."""

    def __init__(self, text):
        self._text = text

    def invoke(self, prompt):
        class _Resp:
            content = self._text
        return _Resp()


def _run_rewrite(monkeypatch, llm_output, question, turns):
    import app.conversation as C
    import app.rag as R

    # active_llm 은 읽기 전용 property 라 provider 쪽을 바꾼다.
    monkeypatch.setattr(C.settings, "llm_provider", "gemini")
    # ★ provider 만 바꾸면 부족하다 — active_llm 은 **키가 실제로 있는지**까지 보고
    #   없으면 extractive 로 폴백한다. 그래서 이 테스트는 `.env` 에 키가 있는 PC 에서만
    #   통과하고 CI 에서는 rewrite 가 시작도 못 한 채 no_llm 으로 빠졌다.
    #   테스트가 환경에 기대고 있었던 것이지 코드가 틀린 게 아니었다.
    monkeypatch.setattr(C.settings, "google_api_key", "test-key")
    monkeypatch.setattr(R, "_llm", lambda: _FakeLLM(llm_output))
    # 재작성 경로에 실제로 들어갔는지 먼저 못박는다. 이게 없으면 다음에 같은 종류로
    # 새면 "왜 값이 다르지" 로만 보이고 원인이 환경이라는 게 드러나지 않는다.
    assert C.settings.active_llm == "gemini"
    return C.rewrite_query(question, turns)


def test_지어낸_재작성은_버리고_원문으로_검색한다(monkeypatch):
    # 재작성이 실패해도 멀티턴 이전 동작(원문 검색)으로 안전하게 떨어져야 한다.
    turns = [_t("user", "리랭커 얘기"), _t("assistant", "네.")]
    q, info = _run_rewrite(monkeypatch, "MMR 계수를 왜 0.5로 정했어?",
                           "그건 왜 그렇게 정했어?", turns)
    assert q == "그건 왜 그렇게 정했어?"          # 원문으로 되돌아간다
    assert info["skipped"] == "invented_facts"
    assert "0.5" in info["invented"]


def test_정상적인_재작성은_그대로_쓴다(monkeypatch):
    turns = [_t("user", "리랭커를 왜 껐어?"),
             _t("assistant", "app/retriever.py 의 search 에서 껐습니다.")]
    q, info = _run_rewrite(monkeypatch, "리랭커를 왜 껐어?", "그건 왜 그랬어?", turns)
    assert q == "리랭커를 왜 껐어?"
    assert info["applied"] is True and info.get("invented") is None
