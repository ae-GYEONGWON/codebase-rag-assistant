"""합성 평가셋 생성·검수의 순수 로직 테스트.

LLM 호출이 필요한 부분은 제외하고, **평가의 신뢰도를 좌우하는 계산**만 고정한다:
어휘 중복률(합성 문항이 베껴 쓴 것인지), 워크시트 파싱, 불일치율 집계.
"""
import json

import pytest

from eval import verify
from eval.generate import lexical_overlap
from eval.llm import ModelSpec, parse_json


# --- 어휘 중복률 ------------------------------------------------------------

def test_조각을_그대로_베낀_질문은_중복률이_높다():
    chunk = "MMR 다양성 계수는 재스윕 결과 1.0 으로 채택했다"
    assert lexical_overlap("MMR 다양성 계수는 1.0 으로 채택했다", chunk) > 0.8


def test_다른_말로_물으면_중복률이_낮다():
    chunk = "MMR 다양성 계수는 재스윕 결과 1.0 으로 채택했다"
    assert lexical_overlap("검색 결과를 고를 때 겹치는 걸 얼마나 허용해?", chunk) < 0.4


def test_빈_질문은_0():
    assert lexical_overlap("", "아무 내용") == 0.0


# --- LLM 응답 파싱 ----------------------------------------------------------

def test_코드펜스를_두른_json도_파싱한다():
    assert parse_json('```json\n["a", "b"]\n```') == ["a", "b"]


def test_설명이_앞뒤로_붙어도_본문_json을_건진다():
    assert parse_json('네, 만들었습니다:\n["질문"]\n도움이 되었길!') == ["질문"]


def test_파싱_불가면_None을_돌려_호출부가_세게_한다():
    assert parse_json("죄송합니다, 만들 수 없습니다.") is None


def test_modelspec_은_라벨을_자동으로_만든다():
    assert ModelSpec("gemini", "x-1").label == "gemini:x-1"
    assert ModelSpec("gemini", "x-1", "심사관").label == "심사관"


# --- 검수 워크시트 ----------------------------------------------------------

@pytest.fixture
def worksheet(tmp_path, monkeypatch):
    items = [
        {"q": "질문A", "expected": ["a.md"], "axis": "doc", "chunk_index": 0, "lex_overlap": 0.1},
        {"q": "질문B", "expected": ["b.py"], "axis": "code", "chunk_index": 1, "lex_overlap": 0.2},
        {"q": "질문C", "expected": ["c.py"], "axis": "code", "chunk_index": 2, "lex_overlap": 0.3},
        {"q": "질문D", "expected": ["d.py"], "axis": "code", "chunk_index": 3, "lex_overlap": 0.4},
    ]
    (tmp_path / "worksheet.json").write_text(
        json.dumps({"seed": 1, "n": 4, "items": items}, ensure_ascii=False), encoding="utf-8"
    )
    md = ["# 워크시트", ""]
    for i, (it, verdict, also) in enumerate(
        zip(items, ["ok", "elsewhere", "wrong", ""], ["", "x.py y.py", "", ""]), 1
    ):
        md += [f"## {i}. {it['q']}", "", f"verdict: {verdict}", f"also: {also}", "", "---", ""]
    (tmp_path / "worksheet.md").write_text("\n".join(md), encoding="utf-8")

    monkeypatch.setattr(verify, "WORKSHEET_JSON", tmp_path / "worksheet.json")
    monkeypatch.setattr(verify, "WORKSHEET_MD", tmp_path / "worksheet.md")
    return tmp_path


def test_워크시트에서_판정과_추가정답을_읽는다(worksheet):
    rows = verify._parse_worksheet()
    assert [r["verdict"] for r in rows] == ["ok", "elsewhere", "wrong", ""]
    assert rows[1]["also"] == ["x.py", "y.py"]


def test_미기입_문항은_집계에서_빠진다(worksheet):
    rows = verify._parse_worksheet()
    filled = [r for r in rows if r["verdict"] in verify.VALID_VERDICTS]
    assert len(filled) == 3  # 빈 판정 1건 제외


def test_폐기_대상은_wrong_과_unclear_뿐이다():
    """elsewhere 는 폐기가 아니다 — 라벨이 좁았을 뿐 문항은 쓸 수 있다."""
    assert verify.DROP == {"wrong", "unclear"}
    assert "elsewhere" not in verify.DROP


def test_판정값_집합이_워크시트_안내와_일치한다():
    assert verify.VALID_VERDICTS == {"ok", "elsewhere", "wrong", "unclear"}
