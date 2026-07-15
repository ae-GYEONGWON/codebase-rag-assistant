"""코드 AST 청킹 — 함수/클래스 단위로 잘리고 컨텍스트 헤더가 붙는지."""
import ast

from app.code_loader import _segments


SRC = '''import os

X = 1

def foo(a):
    return a + 1

class Bar:
    def m(self):
        return 2
'''


def _seg_map(src):
    return dict(_segments(ast.parse(src), src.splitlines()))


def test_module_head_captured():
    """첫 def 이전(import·상수)은 (module) 청크로 보존 — 상수 검색 가치."""
    segs = _seg_map(SRC)
    assert "X = 1" in segs["(module)"]


def test_function_is_own_chunk():
    segs = _seg_map(SRC)
    assert "def foo" in segs["foo"]
    assert "return a + 1" in segs["foo"]


def test_class_captured():
    segs = _seg_map(SRC)
    assert "Bar" in segs
    assert "def m" in segs["Bar"]


def test_syntax_error_is_skipped():
    """문법 오류 파일은 예외 없이 건너뛰도록 load_code 가 방어(여기선 parse 만 확인)."""
    import pytest

    with pytest.raises(SyntaxError):
        ast.parse("def broken(:\n")
