"""검색 파이프라인 통합 테스트.

인덱싱된 chroma_db + 로컬 임베딩 모델이 있어야 의미가 있으므로, 없으면 skip.
(CI 에서는 인덱스를 만든 뒤 실행하거나, 이 파일만 마커로 제외)
"""
from pathlib import Path

import pytest

from app.config import settings

_HAS_INDEX = Path(settings.chroma_dir).exists()
pytestmark = pytest.mark.skipif(not _HAS_INDEX, reason="chroma_db 인덱스 없음")


def test_out_of_scope_rejected():
    """지식원과 무관한 질문은 검색 단계에서 거절(빈 결과)."""
    from app.retriever import search

    docs, debug = search("고양이 키우는 법 알려줘")
    assert docs == []
    assert debug["reason"] == "out_of_scope"


def test_in_scope_hits():
    from app.retriever import search

    docs, debug = search("지금 운용 모드가 뭐야?")
    assert docs
    assert debug["reason"] == "hit"


def test_symbol_exact_match_boosts_code():
    """질문에 코드 심볼명이 있으면 그 함수 본문(코드 청크)이 top-k 에 포함."""
    from app.retriever import search

    docs, _ = search("apply_vix_sl 함수 코드 보여줘")
    types = {d.metadata.get("doc_type") for d in docs}
    assert "code" in types
    assert any("apply_vix_sl" in d.metadata.get("section", "") for d in docs)


def test_sources_align_with_answer_footnotes():
    """출처 개수 = 검색 청크 개수(답변 [n] 각주와 1:1 정렬 보장)."""
    from app.retriever import search, snippets_for

    docs, _ = search("zombie recovery 는 코드에서 어떻게 구현돼 있어?")
    assert len(snippets_for(docs, "zombie recovery")) == len(docs)
