"""Gemini 3.x 멀티파트 응답 정규화 — content 가 문자열이 아닐 때."""
from app.rag import _text_of


def test_plain_string():
    assert _text_of("안녕하세요") == "안녕하세요"


def test_part_array():
    """Gemini 3.x: [{'type':'text','text':...}, {'extras':{...}}] → text 만 결합."""
    content = [
        {"type": "text", "text": "가"},
        {"extras": {"signature": "xxx"}},   # 비텍스트 파트는 버림
        {"type": "text", "text": "나"},
    ]
    assert _text_of(content) == "가나"


def test_string_parts_in_list():
    assert _text_of(["a", "b"]) == "ab"


def test_empty_and_none():
    assert _text_of("") == ""
    assert _text_of(None) == ""
