"""하이브리드 검색 — BM25(어휘) + 벡터(의미) → RRF 융합 → MMR 다양화 → 범위 밖 판정.

순수 벡터 검색만 쓰면 `RC4025`, `HARD_END`, `SL_CAP_PERCENT` 같은
**희귀 식별자**를 놓친다(임베딩은 의미는 잡아도 정확한 토큰 일치에 약함).
반대로 BM25 만 쓰면 "지금 어떤 모드로 돌려?" 처럼 표현이 다른 질문을 놓친다.
두 순위를 RRF(Reciprocal Rank Fusion)로 합쳐 서로의 약점을 메운다.

이어서 MMR 로 같은 내용의 청크가 top-k 를 잠식하는 것을 막고,
최고 유사도가 임계 미만이면 '아는 범위 밖'으로 판정해 환각을 원천 차단한다.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import settings
from app.embeddings import get_embeddings
from app.ingest import get_vectorstore

# 영문/숫자/밑줄 토큰(RC4025, HARD_END …) 과 한글 덩어리를 분리
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[가-힣]+")
_RRF_K = 60  # RRF 상수(관례값). 상위 순위 간 점수 차를 완만하게 만든다.


def _tokenize(text: str) -> List[str]:
    """형태소 분석기 없이 쓰는 경량 토크나이저.

    한글은 조사·어미가 붙어 그대로 매칭하면 '모드는' != '모드' 로 어긋난다.
    → 한글 덩어리는 **문자 2-gram** 으로도 펼쳐 부분 일치를 잡는다.
    """
    tokens: List[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tokens.append(tok)
        if not tok.isascii() and len(tok) > 1:
            tokens.extend(tok[i : i + 2] for i in range(len(tok) - 1))
    return tokens


@lru_cache(maxsize=1)
def _corpus() -> Tuple[List[Document], np.ndarray, BM25Okapi]:
    """Chroma 에 인덱싱된 청크 전체 + 정규화 임베딩 행렬 + BM25 인덱스.

    임베딩을 Chroma 에서 그대로 꺼내 쓰므로 재계산 비용이 없다(956청크 × 768dim ≈ 3MB).
    """
    raw = get_vectorstore().get(include=["documents", "metadatas", "embeddings"])
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]

    embs = np.asarray(raw["embeddings"], dtype=np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9  # 코사인용 정규화

    # 파일명·섹션명도 BM25 본문에 포함 → "zombie_recovery" 같은 파일명 질의가 걸린다.
    bm25 = BM25Okapi(
        [
            _tokenize(
                f"{d.page_content} {d.metadata.get('source', '')} {d.metadata.get('section', '')}"
            )
            for d in docs
        ]
    )
    return docs, embs, bm25


def _mmr(cands: List[int], embs: np.ndarray, rel: Dict[int, float], k: int, lam: float) -> List[int]:
    """Maximal Marginal Relevance — 적합도와 '이미 고른 것과의 차별성'을 lam 으로 절충.

    적합도 항은 **RRF 융합 점수**를 쓴다. 여기서 코사인을 쓰면 BM25 가 끌어올린
    희귀 식별자 문서가 최종 선택에서 다시 탈락해 하이브리드가 무의미해진다.
    임베딩은 중복 penalty(청크 간 유사도) 계산에만 쓴다.
    """
    top = max(rel.values()) or 1.0
    selected: List[int] = []
    pool = list(cands)
    while pool and len(selected) < k:
        if not selected:
            best = max(pool, key=lambda i: rel[i])
        else:
            chosen = embs[selected]  # (n_sel, dim)
            best = max(
                pool,
                key=lambda i: lam * (rel[i] / top)
                - (1 - lam) * float(np.max(chosen @ embs[i])),
            )
        selected.append(best)
        pool.remove(best)
    return selected


def _snippet(text: str, query: str, width: int = 220) -> str:
    """질의어가 처음 등장하는 지점 주변을 잘라 인용 스니펫으로."""
    body = " ".join(text.split())
    terms = [t for t in _tokenize(query) if len(t) > 1]
    pos = min(
        (body.lower().find(t) for t in terms if body.lower().find(t) >= 0),
        default=0,
    )
    start = max(0, pos - width // 3)
    out = body[start : start + width]
    return ("…" if start > 0 else "") + out + ("…" if start + width < len(body) else "")


def search(question: str, k: int | None = None) -> Tuple[List[Document], Dict[str, Any]]:
    """질문 → (선택된 청크, 진단정보). 범위 밖이면 빈 리스트.

    진단정보(debug)는 /chat 응답과 평가 하네스에서 검색 품질을 들여다보는 용도.
    """
    k = k or settings.retrieval_k
    docs, embs, bm25 = _corpus()
    if not docs:
        return [], {"reason": "empty_index"}

    qv = np.asarray(get_embeddings().embed_query(question), dtype=np.float32)
    qv /= np.linalg.norm(qv) + 1e-9

    sims = embs @ qv  # 코사인 유사도 (−1~1, 실사용 0~1)
    bm_scores = np.asarray(bm25.get_scores(_tokenize(question)), dtype=np.float32)

    fetch = min(settings.fetch_k, len(docs))
    vec_rank = np.argsort(-sims)[:fetch]
    bm_rank = np.argsort(-bm_scores)[:fetch]

    # --- RRF 융합: 점수 스케일이 다른 두 랭킹을 '순위'만으로 합친다(정규화 불필요) ---
    fused: Dict[int, float] = {}
    for rank, idx in enumerate(vec_rank):
        fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (_RRF_K + rank + 1)
    for rank, idx in enumerate(bm_rank):
        fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (_RRF_K + rank + 1)

    cands = sorted(fused, key=lambda i: -fused[i])[: max(fetch, k)]

    # --- 범위 밖 판정: 코사인 단독 게이트 ---
    # BM25 는 게이트로 쓰지 않는다. 한글 2-gram 이 흔한 음절에 걸려 "고양이 키우는 법"
    # 같은 질문도 16+ 를 받기 때문(반면 코사인은 0.25). 식별자 단독 질의(RC4025 등)는
    # 코사인이 오히려 0.5+ 라 이 게이트만으로 충분하다. — 실측 보정, config 주석 참고.
    best_sim = float(sims.max())
    best_bm = float(bm_scores.max())
    if best_sim < settings.min_similarity:
        return [], {
            "reason": "out_of_scope",
            "best_similarity": round(best_sim, 3),
            "best_bm25": round(best_bm, 2),
        }

    picked = _mmr(cands, embs, fused, k, settings.mmr_lambda)
    chosen = [docs[i] for i in picked]

    debug = {
        "reason": "hit",
        "best_similarity": round(best_sim, 3),
        "best_bm25": round(best_bm, 2),
        "picked": [
            {
                "source": docs[i].metadata.get("source", "?"),
                "section": docs[i].metadata.get("section", ""),
                "similarity": round(float(sims[i]), 3),
                "bm25": round(float(bm_scores[i]), 2),
            }
            for i in picked
        ],
    }
    return chosen, debug


def snippets_for(docs: List[Document], question: str) -> List[Dict[str, str]]:
    """출처 목록 + 인용 스니펫(웹UI 각주 펼치기에 사용). 문서 순서 = [n] 번호."""
    return [
        {
            "source": d.metadata.get("source", "?"),
            "section": d.metadata.get("section", ""),
            "snippet": _snippet(d.page_content, question),
        }
        for d in docs
    ]
