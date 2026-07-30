"""하이브리드 검색 — BM25(어휘) + 벡터(의미) → RRF 융합 → MMR 다양화 → 범위 밖 판정.

순수 벡터 검색만 쓰면 `ERR7742`, `BATCH_DEADLINE`, `RATE_CAP_PERCENT` 같은
**희귀 식별자**를 놓친다(임베딩은 의미는 잡아도 정확한 토큰 일치에 약함).
반대로 BM25 만 쓰면 "지금 어떤 설정으로 동작해?" 처럼 표현이 다른 질문을 놓친다.
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

# 영문/숫자/밑줄 토큰(ERR7742, BATCH_DEADLINE …) 과 한글 덩어리를 분리
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[가-힣]+")
_RRF_K = 60  # RRF 상수(관례값). 상위 순위 간 점수 차를 완만하게 만든다.
_SYMBOL_SLOTS = 2  # 심볼 정확매칭 코드를 top-k 에 보장할 최대 개수(나머지는 일반 검색).

# 질문에서 ASCII 식별자 토큰을 추출한다. [a-zA-Z0-9_] 만 매칭하므로
# 한글이 붙어있어도 "ERR7742가" → "ERR7742" 로 자동 분리된다(\b 불필요).
_ID_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*")


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
        elif "_" in tok:
            # snake_case 식별자는 조각으로도 색인. 파일명 composite_m9s5 를 통째 토큰으로만
            # 두면 "composite m9s5" 질의가 어디에도 걸리지 않는다(본문 표기는 'm=9 / s=5').
            tokens.extend(p for p in tok.split("_") if p)
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

    # 파일명·섹션명도 BM25 본문에 포함 → "orphan_recovery" 같은 파일명 질의가 걸린다.
    bm25 = BM25Okapi(
        [
            _tokenize(
                f"{d.page_content} {d.metadata.get('source', '')} {d.metadata.get('section', '')}"
            )
            for d in docs
        ]
    )
    return docs, embs, bm25


@lru_cache(maxsize=1)
def _symbol_index() -> List[tuple]:
    """(소문자 심볼명, 청크 인덱스) 목록 — 코드 청크의 함수/메서드명 정확 매칭용.

    코드 본문은 영어 식별자뿐이라 한국어 질문과 임베딩 유사도가 낮아 검색에서 밀린다.
    질문에 심볼명이 그대로 등장하면("apply_retry_policy 함수?") 그 청크를 강제로 끌어올린다.
    """
    docs, _, _ = _corpus()
    out = []
    for i, d in enumerate(docs):
        if d.metadata.get("doc_type") != "code":
            continue
        sym = d.metadata.get("section", "")
        # 전체 심볼(apply_retry_policy)과 메서드명(WorkerMain.foo → foo) 둘 다 후보
        for cand in {sym, sym.split(".")[-1]}:
            c = cand.lower().lstrip("_")
            if c and c != "module":
                out.append((c, i))
    return out


def _symbol_hits(question: str) -> List[int]:
    """질문에 코드 심볼명이 등장하는 코드 청크 인덱스 목록(중복 제거).

    _ID_TOKEN_RE 로 ASCII 식별자 토큰을 먼저 추출해 한글 조사 오염을 막은 뒤,
    아래 두 전략으로 후보를 선별한다.
    ① 어휘 매칭: 토큰이 등록 심볼 vocab 에 정확히 있는 것 (4자 심볼도 포착)
    ② 패턴 매칭: snake_case(apply_retry_policy) / ALL_CAPS(ERR7742) 형태
       → 길이 휴리스틱 없이 식별자 형태 자체로 판별
    """
    index = _symbol_index()
    vocab = {sym for sym, _ in index}

    candidates: set = set()
    for tok in _ID_TOKEN_RE.findall(question):
        tl = tok.lower()
        if tl in vocab:
            candidates.add(tl)
        elif re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", tl):  # snake_case
            candidates.add(tl)
        elif re.fullmatch(r"[A-Z][A-Z0-9_]{3,}", tok):  # ALL_CAPS (원본 케이스로 검사)
            candidates.add(tl)

    seen: set = set()
    hits: List[int] = []
    for sym, i in index:
        if sym in candidates and i not in seen:
            seen.add(i)
            hits.append(i)
    return hits


def _mmr(cands: List[int], embs: np.ndarray, rel: Dict[int, float], k: int, lam: float) -> List[int]:
    """Maximal Marginal Relevance — 적합도와 '이미 고른 것과의 차별성'을 lam 으로 절충.

    적합도 항은 **RRF 융합 점수**를 쓴다. 여기서 코사인을 쓰면 BM25 가 끌어올린
    희귀 식별자 문서가 최종 선택에서 다시 탈락해 하이브리드가 무의미해진다.
    임베딩은 중복 penalty(청크 간 유사도) 계산에만 쓴다.
    """
    # rel 에 없는 후보(심볼 매칭으로 뒤늦게 합류한 청크)는 적합도 0 으로 본다.
    top = (max(rel.values()) if rel else 0.0) or 1.0
    selected: List[int] = []
    pool = list(cands)
    while pool and len(selected) < k:
        if not selected:
            best = max(pool, key=lambda i: rel.get(i, 0.0))
        else:
            chosen = embs[selected]  # (n_sel, dim)
            best = max(
                pool,
                key=lambda i: lam * (rel.get(i, 0.0) / top)
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

    # 질문에 코드 심볼명이 그대로 있으면(예 "apply_retry_policy 함수?") 그 청크를 후보에 넣는다.
    sym_hits = _symbol_hits(question)

    n_cands = max(settings.rerank_candidates if settings.use_reranker else fetch, k)
    cands = sorted(fused, key=lambda i: -fused[i])[:n_cands]
    for i in sym_hits:  # 후보에 없으면 합류(MMR 대상이 되도록)
        if i not in cands:
            cands.append(i)

    # --- 범위 밖 판정: 코사인 단독 게이트 ---
    # BM25 는 게이트로 쓰지 않는다. 한글 2-gram 이 흔한 음절에 걸려 "고양이 키우는 법"
    # 같은 질문도 16+ 를 받기 때문(반면 코사인은 0.25). 식별자 단독 질의(ERR7742 등)는
    # 코사인이 오히려 0.5+ 라 이 게이트만으로 충분하다. — 실측 보정, config 주석 참고.
    best_sim = float(sims.max())
    best_bm = float(bm_scores.max())
    if best_sim < settings.min_similarity and not sym_hits:  # 심볼 정확매칭이면 범위 안으로 인정
        return [], {
            "reason": "out_of_scope",
            "best_similarity": round(best_sim, 3),
            "best_bm25": round(best_bm, 2),
        }

    # --- 리랭킹: 후보만 cross-encoder 로 재채점 → MMR 의 적합도 항으로 사용 ---
    # 리랭커 점수는 로짓이라 음수가 나온다. MMR 이 rel[i]/max 로 정규화하므로 0~1 로 맞춰준다.
    rel: Dict[int, float] = fused
    rr_scores: Dict[int, float] = {}
    if settings.use_reranker:
        from app.reranker import score as rr_score

        raw = rr_score(question, [docs[i].page_content for i in cands])
        rr_scores = {i: s for i, s in zip(cands, raw)}
        lo, hi = min(raw), max(raw)
        span = (hi - lo) or 1.0
        rel = {i: (s - lo) / span for i, s in rr_scores.items()}

    picked = _mmr(cands, embs, rel, k, settings.mmr_lambda)

    # 심볼 정확매칭 청크를 top-k 에 최대 _SYMBOL_SLOTS 개 보장(융합점수 순). 나머지 슬롯은
    # 일반 검색 결과 유지 → "batch_deadline 시각?"(정답=문서)이 코드에 독점당하지 않게.
    if sym_hits:
        extra = [i for i in sorted(sym_hits, key=lambda i: -fused.get(i, 0.0)) if i not in picked][:_SYMBOL_SLOTS]
        if extra:
            picked = extra + [p for p in picked if p not in extra]
            picked = picked[:k]

    chosen = [docs[i] for i in picked]

    debug = {
        "reason": "hit",
        "reranked": bool(rr_scores),
        "best_similarity": round(best_sim, 3),
        "best_bm25": round(best_bm, 2),
        "picked": [
            {
                "source": docs[i].metadata.get("source", "?"),
                "section": docs[i].metadata.get("section", ""),
                "similarity": round(float(sims[i]), 3),
                "bm25": round(float(bm_scores[i]), 2),
                "rerank": round(rr_scores[i], 3) if i in rr_scores else None,
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
            "doc_type": d.metadata.get("doc_type", "doc"),
            "snippet": _snippet(d.page_content, question),
        }
        for d in docs
    ]
