"""Cross-encoder 리랭커.

임베딩(bi-encoder)은 질문과 문서를 **따로** 벡터로 만들어 비교한다. 빠르지만
"이 문서가 이 질문에 답이 되는가"를 직접 보지는 못한다.
cross-encoder 는 (질문, 문서)를 **한 입력으로 함께** 넣어 관련도를 점수화한다.
느려서 전체 코퍼스에는 못 쓰고, 하이브리드 검색이 좁혀준 후보 20개에만 적용한다.

  전체 3342청크 ──BM25+벡터 RRF──▶ 후보 20 ──cross-encoder──▶ 최종 5
      (싸고 넓게)                        (비싸고 정확하게)

코드 인덱싱으로 코퍼스가 3.5배가 되면서 top-k 노이즈가 늘었고("SL 은 코드에서 어떻게
계산돼?" 가 실제 구현 파일을 못 집었다), 그 지점을 메우는 것이 이 단계의 목적이다.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

from langchain_core.documents import Document

from app.config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    # max_length: 코드 청크는 토큰이 길다. 512 를 넘으면 뒤가 잘린다.
    return CrossEncoder(settings.reranker_model, max_length=512)


def score(question: str, texts: List[str]) -> List[float]:
    """(질문, 청크) 쌍마다 관련도 점수. 점수는 로짓이라 음수도 나온다."""
    if not texts:
        return []
    return [float(s) for s in _model().predict([(question, t) for t in texts])]


def rerank(question: str, docs: List[Document], k: int) -> List[Document]:
    """후보 문서를 (질문, 문서) 쌍 점수로 재정렬해 상위 k 개 반환."""
    if not settings.use_reranker or not docs:
        return docs[:k]

    ranked: List[Tuple[float, Document]] = sorted(
        zip(score(question, [d.page_content for d in docs]), docs), key=lambda x: -x[0]
    )
    for s, doc in ranked[:k]:
        doc.metadata["rerank_score"] = round(s, 3)
    return [d for _, d in ranked[:k]]


def warmup() -> None:
    """모델 가중치를 미리 적재(첫 질문 지연 방지)."""
    if settings.use_reranker:
        _model().predict([("warmup", "warmup")])
