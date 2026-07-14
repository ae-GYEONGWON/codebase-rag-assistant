"""검색 평가 하네스 — 벡터 단독 / BM25 단독 / 하이브리드(RRF+MMR) 비교.

측정:
  recall@k  — 정답 출처 파일이 top-k 안에 있으면 hit (질문셋의 expected 중 하나라도)
  MRR       — 정답 출처가 처음 등장한 순위의 역수 평균(상위에 꽂을수록 높음)
  거절률     — 범위 밖 질문을 검색 단계에서 컷한 비율(높을수록 환각 방지)

실행:
    python -m eval.run_eval            # k=5
    python -m eval.run_eval --k 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from app.config import settings
from app.embeddings import get_embeddings
from app.retriever import _RRF_K, _corpus, _mmr, _tokenize, search

QUESTIONS = Path(__file__).parent / "questions.json"


def _query_vec(question: str) -> np.ndarray:
    qv = np.asarray(get_embeddings().embed_query(question), dtype=np.float32)
    return qv / (np.linalg.norm(qv) + 1e-9)


def retrieve_vector(question: str, k: int) -> List[str]:
    docs, embs, _ = _corpus()
    sims = embs @ _query_vec(question)
    return [docs[i].metadata.get("source", "?") for i in np.argsort(-sims)[:k]]


def retrieve_bm25(question: str, k: int) -> List[str]:
    docs, _, bm25 = _corpus()
    scores = np.asarray(bm25.get_scores(_tokenize(question)), dtype=np.float32)
    return [docs[i].metadata.get("source", "?") for i in np.argsort(-scores)[:k]]


def retrieve_hybrid(question: str, k: int) -> List[str]:
    docs, _ = search(question, k=k)
    return [d.metadata.get("source", "?") for d in docs]


def retrieve_rrf_no_mmr(question: str, k: int) -> List[str]:
    """MMR 을 뺀 순수 RRF — MMR 이 실제로 기여하는지 분리 측정용."""
    docs, embs, bm25 = _corpus()
    qv = _query_vec(question)
    sims = embs @ qv
    bm = np.asarray(bm25.get_scores(_tokenize(question)), dtype=np.float32)
    fetch = min(settings.fetch_k, len(docs))
    fused: Dict[int, float] = {}
    for rank, i in enumerate(np.argsort(-sims)[:fetch]):
        fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (_RRF_K + rank + 1)
    for rank, i in enumerate(np.argsort(-bm)[:fetch]):
        fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (_RRF_K + rank + 1)
    top = sorted(fused, key=lambda i: -fused[i])[:k]
    return [docs[i].metadata.get("source", "?") for i in top]


def score(retriever: Callable[[str, int], List[str]], cases: List[dict], k: int) -> Tuple[float, float, List[str]]:
    hits, rr, misses = 0, 0.0, []
    for c in cases:
        got = retriever(c["q"], k)
        rank = next((i for i, s in enumerate(got, 1) if s in c["expected"]), None)
        if rank:
            hits += 1
            rr += 1.0 / rank
        else:
            misses.append(c["q"])
    n = len(cases)
    return hits / n, rr / n, misses


def main() -> None:
    ap = argparse.ArgumentParser(description="검색 평가")
    ap.add_argument("--k", type=int, default=settings.retrieval_k)
    args = ap.parse_args()
    k = args.k

    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    in_scope: List[dict] = data["in_scope"]
    out_scope: List[str] = data["out_of_scope"]

    retrievers = {
        "vector only": retrieve_vector,
        "bm25 only": retrieve_bm25,
        "hybrid (RRF)": retrieve_rrf_no_mmr,
        "hybrid + MMR": retrieve_hybrid,
    }

    print(f"\n■ 검색 품질 — 질문 {len(in_scope)}개, k={k}\n")
    print(f"{'retriever':<16}{'recall@k':>10}{'MRR':>8}   miss")
    print("-" * 64)
    results = {}
    for name, fn in retrievers.items():
        recall, mrr, misses = score(fn, in_scope, k)
        results[name] = (recall, mrr)
        print(f"{name:<16}{recall:>9.0%}{mrr:>8.2f}   {len(misses)}건")
        for m in misses:
            print(f"{'':<16}{'':>18}   ✗ {m}")

    # 범위 밖 거절: 프로덕션 게이트(hybrid)만 해당. 나머지는 게이트가 없어 항상 무언가를 반환.
    rejected = sum(1 for q in out_scope if not search(q)[0])
    print(f"\n■ 범위 밖 거절 — {rejected}/{len(out_scope)} ({rejected/len(out_scope):.0%})  "
          f"[임계 cos ≥ {settings.min_similarity}]")
    for q in out_scope:
        docs, dbg = search(q)
        mark = "컷" if not docs else "통과(!)"
        print(f"   {mark:<6} cos={dbg.get('best_similarity', 0):.3f}  {q}")
    print()


if __name__ == "__main__":
    main()
