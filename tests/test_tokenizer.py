"""BM25 토크나이저 — 하이브리드 검색의 어휘 축이 의도대로 자르는지."""
from app.retriever import _tokenize


def test_identifier_preserved():
    """희귀 식별자는 통째로 보존돼야 정확 매칭이 된다(ERR7742 등)."""
    assert "err7742" in _tokenize("ERR7742 오류는 왜 나?")


def test_snake_case_split():
    """snake_case 식별자는 전체 + 조각으로 색인 → 부분 질의도 매칭."""
    toks = _tokenize("apply_retry_policy")
    assert "apply_retry_policy" in toks   # 전체
    assert "retry" in toks                # 조각


def test_korean_bigram():
    """한글은 2-gram 으로 펼쳐 조사·어미 변형에도 부분 일치."""
    toks = _tokenize("배치작업")
    assert "배치" in toks
    assert "치작" in toks


def test_lowercased():
    assert _tokenize("BATCH_DEADLINE") == _tokenize("batch_deadline")
