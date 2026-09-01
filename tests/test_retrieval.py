"""검색 파이프라인 통합 테스트.

인덱싱된 chroma_db + 로컬 임베딩 모델이 있어야 의미가 있으므로, 없으면 skip.
(CI 에서는 인덱스를 만든 뒤 실행하거나, 이 파일만 마커로 제외)

※ 질의를 코퍼스에서 동적으로 뽑는다 — 특정 프로젝트의 심볼·용어를 하드코딩하지 않으므로
   어떤 코드베이스를 인덱싱했든 그대로 통과한다.
"""
from pathlib import Path

import pytest

from app.profiles import active_profile

_HAS_INDEX = Path(active_profile().chroma_dir).exists()
pytestmark = pytest.mark.skipif(not _HAS_INDEX, reason="chroma_db 인덱스 없음")


def _a_doc_section() -> str:
    """인덱싱된 문서 청크에서 섹션명 하나를 골라 '범위 안' 질의로 쓴다."""
    from app.retriever import _corpus

    docs, _, _ = _corpus()
    for d in docs:
        if d.metadata.get("doc_type") == "doc":
            section = (d.metadata.get("section") or "").strip()
            if len(section) >= 4:
                return section
    pytest.skip("문서 청크 없음")


def _a_code_symbol() -> str:
    """인덱싱된 코드 청크에서 식별하기 좋은 심볼명 하나를 고른다."""
    from app.retriever import _symbol_index

    for sym, _ in _symbol_index():
        if "_" in sym and len(sym) >= 8:   # snake_case 이고 충분히 희귀한 것
            return sym
    pytest.skip("코드 심볼 없음")


def test_out_of_scope_rejected():
    """지식원과 무관한 질문은 검색 단계에서 거절(빈 결과)."""
    from app.retriever import search

    docs, debug = search("고양이 키우는 법 알려줘")
    assert docs == []
    assert debug["reason"] == "out_of_scope"


def test_in_scope_hits():
    from app.retriever import search

    docs, debug = search(_a_doc_section())
    assert docs
    assert debug["reason"] == "hit"


def test_symbol_exact_match_boosts_code():
    """질문에 코드 심볼명이 있으면 그 함수 본문(코드 청크)이 top-k 에 포함."""
    from app.retriever import search

    sym = _a_code_symbol()
    docs, _ = search(f"{sym} 함수 코드 보여줘")
    types = {d.metadata.get("doc_type") for d in docs}
    assert "code" in types
    assert any(sym in (d.metadata.get("section") or "").lower() for d in docs)


def test_sources_align_with_answer_footnotes():
    """출처 개수 = 검색 청크 개수(답변 [n] 각주와 1:1 정렬 보장)."""
    from app.retriever import search, snippets_for

    question = _a_doc_section()
    docs, _ = search(question)
    assert len(snippets_for(docs, question)) == len(docs)
