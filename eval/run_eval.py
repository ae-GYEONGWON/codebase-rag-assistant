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
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from app.agent import _TOOL_K
from app.config import settings
from app.profiles import active_profile, available_profiles, use_profile
from eval.datasets import load_questions
from eval import report as rp
from app.embeddings import get_embeddings
from app.retriever import _RRF_K, _corpus, _mmr, _tokenize, search

# 에이전트가 한 질문에서 보는 최대 청크 수(툴 3축 × 툴당 k). 단발 RAG 의 '예산 맞춤' 비교 기준.
_TOOL_BUDGET = _TOOL_K * 3




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
    """설정 그대로(리랭커 on/off 는 settings.use_reranker)."""
    docs, _ = search(question, k=k)
    return [d.metadata.get("source", "?") for d in docs]


def _without_reranker(fn: Callable[[str, int], List[str]]) -> Callable[[str, int], List[str]]:
    def wrapped(question: str, k: int) -> List[str]:
        prev = settings.use_reranker
        settings.use_reranker = False
        try:
            return fn(question, k)
        finally:
            settings.use_reranker = prev

    return wrapped


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


def score_multihop(sources_of: Callable[[str], List[str]], cases: List[dict]) -> Dict[str, float]:
    """멀티홉 채점 — hops 는 **AND**. 모든 홉을 채워야 정답.

    단일 홉 질문의 recall 은 'expected 중 하나라도 top-k 에 있으면 hit' 이지만,
    멀티홉에서 그 기준을 쓰면 한 축만 채운 답도 hit 가 되어 에이전트의 이점이 지워진다.
    그래서 두 가지를 나눠 본다:
      hop_coverage — 채운 홉 비율(부분 점수). 왜 틀렸는지 보이게 하는 진단용
      full        — 모든 홉을 채운 비율. 실제 '답할 수 있었나' 에 해당
    """
    total_hops, covered_hops, full = 0, 0, 0
    for c in cases:
        got = set(sources_of(c["q"]))
        hits = [bool(got & set(h["expected"])) for h in c["hops"]]
        total_hops += len(hits)
        covered_hops += sum(hits)
        full += all(hits)
    n = len(cases) or 1
    return {
        "hop_coverage": covered_hops / (total_hops or 1),
        "full": full / n,
    }


def _missing_axes(sources_of: Callable[[str], List[str]], case: dict) -> List[str]:
    got = set(sources_of(case["q"]))
    return [h["axis"] for h in case["hops"] if not (got & set(h["expected"]))]


def run_multihop(cases: List[dict], k: int, use_agent: bool) -> None:
    """멀티홉 비교: 단발 RAG(운영 k) / 단발 RAG(예산 맞춤 k) / 에이전트.

    **예산 맞춤 행이 핵심 통제**다. 에이전트는 툴 3회 × 4청크 = 최대 12청크를 보므로,
    단발 RAG 를 k=5 로만 비교하면 "에이전트가 그냥 더 많이 봐서 이긴 것" 이 된다.
    같은 청크 예산을 준 단발 RAG 와 비교해야 '축을 나눈 것' 의 기여가 분리된다.
    """
    budget_k = _TOOL_BUDGET
    print(f"\n■ 멀티홉 — {len(cases)}문항 (홉 AND 조건)\n")
    print(f"{'retriever':<26}{'홉 커버리지':>12}{'전체정답':>10}{'지연':>9}{'LLM호출':>9}")
    print("-" * 70)

    rows = [
        (f"단발 RAG (k={k})", lambda q: [d.metadata.get("source", "?") for d in search(q, k=k)[0]]),
        (
            f"단발 RAG (k={budget_k}, 예산맞춤)",
            lambda q: [d.metadata.get("source", "?") for d in search(q, k=budget_k)[0]],
        ),
    ]

    stats: Dict[str, Dict[str, float]] = {}
    for name, fn in rows:
        t0 = time.perf_counter()
        s = score_multihop(fn, cases)
        elapsed = (time.perf_counter() - t0) / (len(cases) or 1)
        stats[name] = s
        print(f"{name:<26}{s['hop_coverage']:>11.0%}{s['full']:>10.0%}{elapsed:>8.1f}s{'-':>9}")

    if use_agent:
        from app.agent import answer as loop_answer
        from app.agent_graph import answer as graph_answer

        # 수동 루프 판과 LangGraph 판을 같은 문항·같은 스로틀로 비교한다.
        # 프레임워크 도입이 정답률을 바꾸는지, 아니면 구조만 바뀌고 결과는 같은지를
        # 말로 하지 않고 숫자로 확인하기 위한 행.
        agents = [("에이전트(수동 루프)", loop_answer), ("에이전트(LangGraph)", graph_answer)]
        last_sources_fn = None

        for label, fn in agents:
            cache: Dict[str, Dict] = {}

            def sources_of(q: str, _fn=fn, _cache=cache) -> List[str]:
                if q not in _cache:
                    t0 = time.perf_counter()
                    r = _fn(q)
                    r["_elapsed"] = time.perf_counter() - t0
                    _cache[q] = r
                return [s["source"] for s in _cache[q]["sources"]]

            s = score_multihop(sources_of, cases)
            stats[label] = s
            lat = sum(r["_elapsed"] for r in cache.values()) / (len(cache) or 1)
            calls = sum(r.get("llm_calls", 0) for r in cache.values()) / (len(cache) or 1)
            pad = 26 - (len(label) - len(label.encode("ascii", "ignore").decode()))
            print(f"{label:<{pad}}{s['hop_coverage']:>11.0%}{s['full']:>10.0%}{lat:>8.1f}s{calls:>9.1f}")
            last_sources_fn = sources_of

        print("\n  못 채운 축(에이전트 LangGraph):")
        for c in cases:
            miss = _missing_axes(last_sources_fn, c)
            if miss:
                print(f"    ✗ {'/'.join(miss):<12} {c['q'][:44]}")

    print("\n  못 채운 축(단발 RAG, 운영 k):")
    base = lambda q: [d.metadata.get("source", "?") for d in search(q, k=k)[0]]  # noqa: E731
    for c in cases:
        miss = _missing_axes(base, c)
        if miss:
            print(f"    ✗ {'/'.join(miss):<12} {c['q'][:44]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="검색 평가")
    ap.add_argument("--k", type=int, default=settings.retrieval_k)
    ap.add_argument("--rerank", action="store_true", help="리랭커 비교 행 포함(느림, CPU)")
    ap.add_argument("--multihop", action="store_true", help="멀티홉 스위트만 실행")
    ap.add_argument("--agent", action="store_true", help="멀티홉에 에이전트 행 추가(LLM 호출·느림)")
    ap.add_argument(
        "--profile",
        choices=available_profiles(),
        help="코퍼스 프로필(기본: .env 의 CORPUS_PROFILE). 인덱스·평가셋이 함께 바뀐다.",
    )
    ap.add_argument("--report", action="store_true",
                    help="결과를 eval/reports/<profile>.{json,md} 로 저장")
    ap.add_argument("--gate", action="store_true",
                    help="eval/baselines/<profile>.json 과 비교해 회귀면 exit 1 (CI 용)")
    ap.add_argument("--save-baseline", action="store_true",
                    help="이번 결과를 baseline 으로 저장(의도한 변화일 때만)")
    args = ap.parse_args()
    if args.profile:
        use_profile(args.profile)
    k = args.k

    qs = load_questions()
    print(f"[eval] {qs.summary()}")
    doc_q: List[dict] = qs.in_scope
    code_q: List[dict] = qs.in_scope_code
    multi_q: List[dict] = qs.multihop
    out_scope: List[str] = qs.out_of_scope

    if args.multihop or args.agent:
        if not multi_q:
            print("멀티홉 문항이 없습니다(questions.json 의 multihop).")
            return
        run_multihop(multi_q, k, use_agent=args.agent)
        return

    # ★행 이름 주의(2026-09-01 정정): 마지막 행은 'MMR 의 기여'가 아니라 **운영 파이프라인 전체**다.
    #   mmr_lambda=1.0 이면 MMR 은 순위를 바꾸지 않는다(no-op) — 적합도 최대를 순서대로 고르므로
    #   순수 RRF 순위와 같다. 따라서 'hybrid RRF' 와의 차이는 MMR 이 아니라
    #   **심볼 슬롯**(질문 속 ASCII 식별자와 정확매칭된 코드 청크를 top-k 에 강제 삽입)과 범위밖 게이트다.
    #   예전 이름('hybrid+MMR')은 그 차이를 MMR 의 공으로 읽히게 해 잘못된 결론을 부른다.
    retrievers = {
        "vector only": _without_reranker(retrieve_vector),
        "bm25 only": _without_reranker(retrieve_bm25),
        "hybrid RRF": _without_reranker(retrieve_rrf_no_mmr),
        "운영 파이프라인": _without_reranker(retrieve_hybrid),
    }
    if args.rerank:
        # 리랭커를 강제로 켠 변형(설정과 무관하게 비교용)
        def _with_reranker(question: str, kk: int) -> List[str]:
            prev = settings.use_reranker
            settings.use_reranker = True
            try:
                return retrieve_hybrid(question, kk)
            finally:
                settings.use_reranker = prev

        retrievers["hybrid+rerank"] = _with_reranker

    suites = [("문서 질문", doc_q), ("코드 질문", code_q), ("전체", doc_q + code_q)]

    docs_all, _, _ = _corpus()
    corpus_stats: Dict[str, int] = {}
    for d in docs_all:
        t = d.metadata.get("doc_type", "?")
        corpus_stats[t] = corpus_stats.get(t, 0) + 1

    report = rp.new_report(
        profile=active_profile().name,
        collection=active_profile().collection_name,
        k=k,
        corpus=corpus_stats,
        dataset={"path": qs.path.name, "counts": {
            "in_scope": len(qs.in_scope), "in_scope_code": len(qs.in_scope_code),
            "multihop": len(qs.multihop), "out_of_scope": len(qs.out_of_scope)},
            "origins": qs.origin_counts()},
    )

    for title, cases in suites:
        if not cases:
            continue
        suite = rp.Suite(title=title, n=len(cases))
        report.suites.append(suite)
        print(f"\n■ {title} — {len(cases)}문항, k={k}\n")
        print("   (운영 파이프라인 = RRF + 심볼슬롯 + MMR(lambda=1.0 -> no-op) + 범위밖 게이트)")
        print(f"{'retriever':<16}{'recall@k':>10}{'MRR':>8}   miss")
        print("-" * 64)
        for name, fn in retrievers.items():
            recall, mrr, misses = score(fn, cases, k)
            suite.rows.append(rp.RetrieverRow(name=name, recall=recall, mrr=mrr, misses=misses))
            print(f"{name:<16}{recall:>9.0%}{mrr:>8.2f}   {len(misses)}건")
            if title == "전체":
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

    report.out_of_scope_total = len(out_scope)
    report.out_of_scope_rejected = rejected

    if args.report or args.gate or args.save_baseline:
        jp = report.write_json(rp.REPORT_DIR / f"{report.profile}.json")
        mp = report.write_markdown(rp.REPORT_DIR / f"{report.profile}.md")
        print(f"[report] 저장: {jp}  ·  {mp}")

    if args.save_baseline:
        bp = report.write_json(rp.BASELINE_DIR / f"{report.profile}.json")
        print(f"[baseline] 갱신: {bp}")

    if args.gate:
        bp = rp.BASELINE_DIR / f"{report.profile}.json"
        if not bp.exists():
            print(f"[gate] baseline 없음({bp}) — --save-baseline 으로 먼저 만드세요.")
            raise SystemExit(1)
        problems = rp.compare(rp.load(bp), report)
        if problems:
            print("[gate] ✗ 회귀 감지")
            for p in problems:
                print(f"   - {p}")
            raise SystemExit(1)
        print("[gate] ✓ baseline 대비 회귀 없음")


if __name__ == "__main__":
    main()
