"""BM25 토크나이저 — 하이브리드 검색의 어휘 축이 의도대로 자르는지."""
from app.retriever import _tokenize


def test_identifier_preserved():
    """희귀 식별자는 통째로 보존돼야 정확 매칭이 된다(RC4025 등)."""
    assert "rc4025" in _tokenize("RC4025 오류는 왜 나?")


def test_snake_case_split():
    """snake_case 식별자는 전체 + 조각으로 색인 → 부분 질의도 매칭."""
    toks = _tokenize("apply_vix_sl")
    assert "apply_vix_sl" in toks   # 전체
    assert "vix" in toks            # 조각


def test_korean_bigram():
    """한글은 2-gram 으로 펼쳐 조사·어미 변형에도 부분 일치."""
    toks = _tokenize("운용모드")
    assert "운용" in toks
    assert "용모" in toks


def test_lowercased():
    assert _tokenize("HARD_END") == _tokenize("hard_end")
