"""ANN 전환 임계 실증 — 브루트포스는 언제까지 버티는가(노트 #15 의 주장을 검증한다).

## 무엇을 확인하려는 것인가

이 시스템은 벡터 검색을 **전수 매칭**으로 한다(넘파이 행렬곱 한 번). 노트 #15 는 그것이
게으름이 아니라 선택이며, 전환 임계는 **N ≈ 10만이고 먼저 무너지는 것은 지연이 아니라
메모리**라고 적었다. 그 주장은 3,619 청크에서 잰 값으로부터의 **외삽**이었다.

외삽은 근거가 아니다. 그래서 N 을 실제로 키워 가며 잰다.

## 어떻게 재는가

실제 문서를 100만 개 모을 수는 없으므로 **임베딩 행렬만** 합성한다. 여기서 재려는 것은
검색 품질이 아니라 **비용의 스케일링**이고, 그건 행렬 크기와 차원만으로 결정된다.
합성 벡터는 정규분포에서 뽑아 정규화한다 — 실제 임베딩과 분포는 다르지만,
행렬곱의 비용은 값이 아니라 모양이 정한다.

품질 쪽은 별도로 본다. HNSW 는 근사이므로 **정답을 놓칠 수 있고**, 그 손실을
`recall@k`(전수 매칭 결과 대비 일치율)로 함께 잰다. 지연만 보고 ANN 으로 갈아타면
"빨라졌는데 답이 나빠졌다"를 놓친다 — 이 저장소가 리랭커에서 이미 겪은 실패다.

## 합성으로 재도 되는 것과 안 되는 것 — 이걸 안 나누면 틀린 결론이 나온다

    지연·메모리   합성으로 충분하다. 행렬곱 비용은 값이 아니라 **모양**이 정한다.
    recall 손실   합성으로 재면 안 된다. ← 이 실험에서 실제로 함정에 빠졌다

768차원 정규분포 벡터는 서로 거의 직교해서 **최근접 이웃이라는 구조 자체가 없다.**
그런 데이터에 HNSW 를 걸면 recall 이 0~19% 로 나오는데, 그건 HNSW 가 나쁘다는 뜻이 아니라
**근사 탐색이 붙잡을 이웃 그래프가 없다**는 뜻이다. 그 표를 그대로 실으면 "ANN 은 못 쓴다"는
틀린 결론을 준다.

그래서 recall 은 **실제 코퍼스 임베딩**으로 잰다(`--real`). 규모는 작지만 진짜 분포다.
지연·메모리는 합성으로, 품질 손실은 실물로 — 각각 그것이 유효한 자리에서 잰다.

## 정직한 한계

- **메모리 상한 때문에 큰 N 은 실제로 만들지 못할 수 있다.** 그때는 만들지 못했다는
  사실 자체가 결과다(그게 노트 #15 가 말한 '메모리가 먼저 무너진다'의 실물이다).
- 실측 recall 은 **현재 코퍼스 크기에서의** 값이다. 코퍼스가 커지면 다시 재야 한다.

실행:
    python -m eval.ann_threshold                          # 지연·메모리 스케일링(합성)
    python -m eval.ann_threshold --real --profile eval    # recall 손실(실제 임베딩)
    python -m eval.ann_threshold --no-ann                 # 전수 매칭만(faiss 없이)
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from eval import report as rp

DEFAULT_SIZES = (1_000, 10_000, 50_000, 100_000)
DEFAULT_DIM = 768        # jhgan/ko-sroberta-multitask 의 차원
DEFAULT_K = 5
N_QUERIES = 20


def _synth(n: int, dim: int, seed: int) -> np.ndarray:
    """정규화된 합성 임베딩 행렬. float32 = 실제 저장 형식과 같게."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, dim), dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    return m


def _time_brute(embs: np.ndarray, queries: np.ndarray, k: int) -> Dict[str, float]:
    """전수 매칭 지연. 첫 회는 캐시 워밍이라 버리고 중앙값을 쓴다."""
    lat: List[float] = []
    idx = None
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        sims = embs @ q
        idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        idx = idx[np.argsort(-sims[idx])]
        lat.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.median(lat[1:] or lat)),
            "p95_ms": float(np.percentile(lat[1:] or lat, 95))}


def _brute_truth(embs: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    out = np.empty((len(queries), k), dtype=np.int64)
    for i, q in enumerate(queries):
        sims = embs @ q
        idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        out[i] = idx[np.argsort(-sims[idx])]
    return out


def _run_hnsw(embs: np.ndarray, queries: np.ndarray, k: int,
              truth: np.ndarray, ef_values: List[int]) -> Optional[dict]:
    """HNSW 색인 + ef_search 스윕. faiss 가 없으면 None."""
    try:
        import faiss
    except ImportError:
        return None

    dim = embs.shape[1]
    t0 = time.perf_counter()
    # efConstruction 은 색인 구축 비용을 지배한다. 200 으로 두면 30만 벡터에서
    # 구축만 수십 분이 걸려 **재려던 것(질의 지연)보다 측정 자체가 더 비싸진다.**
    # 40 이면 recall 이 조금 떨어지지만 스케일링 경향을 보는 데는 충분하고,
    # 그 손실은 recall 표에 그대로 드러나므로 숨겨지지 않는다.
    index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 40
    index.add(embs)
    build_s = time.perf_counter() - t0

    rows = []
    for ef in ef_values:
        index.hnsw.efSearch = ef
        lat: List[float] = []
        got = np.empty((len(queries), k), dtype=np.int64)
        for i, q in enumerate(queries):
            t = time.perf_counter()
            _, ids = index.search(q.reshape(1, -1), k)
            lat.append((time.perf_counter() - t) * 1000)
            got[i] = ids[0]
        # recall@k = 전수 매칭이 고른 것 중 몇 개를 되찾았나
        hit = sum(len(set(got[i]) & set(truth[i])) for i in range(len(queries)))
        rows.append({"ef_search": ef,
                     "p50_ms": float(np.median(lat[1:] or lat)),
                     "recall_at_k": hit / (len(queries) * k)})
    del index
    gc.collect()
    return {"build_sec": build_s, "sweep": rows}


def measure(sizes: List[int], dim: int, k: int, seed: int, with_ann: bool,
            ef_values: List[int], ann_max_n: int = 100_000,
            synthetic_recall: bool = False) -> dict:
    """지연·메모리 스케일링을 잰다.

    `synthetic_recall` 은 기본 False 다 — 합성 벡터로 잰 recall 은 HNSW 가 아니라
    합성 데이터의 성질을 재는 것이라, 리포트에 실으면 오독을 부른다(모듈 docstring 참고).
    """
    queries = _synth(N_QUERIES, dim, seed + 1)
    results: List[dict] = []
    for n in sizes:
        need_mb = n * dim * 4 / 1e6
        print(f"\n■ N={n:,}  (행렬 {need_mb:,.0f} MB · float32 {dim}차원)")
        try:
            embs = _synth(n, dim, seed)
        except (MemoryError, ValueError) as e:
            # 만들지 못했다는 사실 자체가 결과다 — '메모리가 먼저 무너진다'의 실물.
            print(f"  ✗ 행렬 생성 실패: {type(e).__name__} — 여기가 메모리 한계다")
            results.append({"n": n, "matrix_mb": need_mb, "failed": "memory"})
            break

        brute = _time_brute(embs, queries, k)
        print(f"  전수 매칭   p50 {brute['p50_ms']:.2f} ms · p95 {brute['p95_ms']:.2f} ms")
        row = {"n": n, "matrix_mb": need_mb, "brute": brute}

        if with_ann and synthetic_recall and n <= ann_max_n:
            truth = _brute_truth(embs, queries, k)
            ann = _run_hnsw(embs, queries, k, truth, ef_values)
            if ann is None:
                print("  (faiss 미설치 — HNSW 비교 생략)")
            else:
                print(f"  HNSW 색인 구축 {ann['build_sec']:.1f}s")
                for r in ann["sweep"]:
                    print(f"    ef={r['ef_search']:<4} p50 {r['p50_ms']:.2f} ms"
                          f"   recall@{k} {r['recall_at_k']:.0%}")
                row["hnsw"] = ann
        results.append(row)
        del embs
        gc.collect()
    return {"dim": dim, "k": k, "n_queries": N_QUERIES, "results": results}


def measure_real(k: int, ef_values: List[int]) -> Optional[dict]:
    """실제 코퍼스 임베딩으로 HNSW 의 **품질 손실**을 잰다.

    질의도 실제 평가셋에서 가져온다 — 질의 분포가 다르면 recall 도 달라지기 때문이다.
    """
    from app.embeddings import get_embeddings
    from app.profiles import active_profile
    from app.retriever import _corpus
    from eval.datasets import load_questions

    docs, embs, _ = _corpus()
    if len(docs) < 50:
        print("[real] 코퍼스가 너무 작아 건너뜁니다.")
        return None

    qs = load_questions()
    cases = (qs.in_scope + qs.in_scope_code + qs.in_scope_commit)[:N_QUERIES]
    if not cases:
        print("[real] 평가 문항이 없어 건너뜁니다.")
        return None

    emb = get_embeddings()
    queries = np.asarray([emb.embed_query(c["q"]) for c in cases], dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9

    print(f"\n= 실제 코퍼스 - {len(docs):,}청크 / 실제 질의 {len(queries)}개")
    truth = _brute_truth(embs, queries, k)
    brute = _time_brute(embs, queries, k)
    print(f"  전수 매칭   p50 {brute['p50_ms']:.2f} ms")
    ann = _run_hnsw(np.ascontiguousarray(embs), queries, k, truth, ef_values)
    if ann is None:
        print("  (faiss 미설치)")
        return None
    for r in ann["sweep"]:
        print(f"    ef={r['ef_search']:<4} p50 {r['p50_ms']:.2f} ms"
              f"   recall@{k} {r['recall_at_k']:.0%}")
    return {"profile": active_profile().name, "n": len(docs), "n_queries": len(queries),
            "brute": brute, "hnsw": ann}


def render(res: dict) -> str:
    k = res["k"]
    out = ["# ANN 전환 임계 실증 — 브루트포스는 언제까지 버티는가", "",
           f"합성 임베딩 {res['dim']}차원 · float32 · 질의 {res['n_queries']}회 · k={k}", "",
           "노트 #15 의 **N ≈ 10만**은 3,619 청크까지만 재고 계산으로 밀어 추정한 값이었다"
           "(외삽). 잰 범위 밖에서는 성질이 달라질 수 있어 실제로 키워 가며 다시 쟀다.", "",
           "| N | 행렬 크기 | 전수매칭 p50 | 전수매칭 p95 |", "|---:|---:|---:|---:|"]
    for r in res["results"]:
        if r.get("failed"):
            out.append(f"| {r['n']:,} | {r['matrix_mb']:,.0f} MB | **생성 실패(메모리)** | — |")
            continue
        out.append(f"| {r['n']:,} | {r['matrix_mb']:,.0f} MB | "
                   f"{r['brute']['p50_ms']:.2f} ms | {r['brute']['p95_ms']:.2f} ms |")

    real = res.get("real")
    if real:
        out += ["", "## HNSW 와의 교환 - 빨라지는 대신 무엇을 잃나", "",
                f"**실제 코퍼스 임베딩** {real['n']:,}청크 · 실제 평가 질의 {real['n_queries']}개. "
                f"`recall@{k}` 는 전수 매칭이 고른 것 대비 일치율이고, 1.00 이 아니면 "
                "그만큼 정답을 놓친 것이다.", "",
                f"| ef_search | HNSW p50 | 전수매칭 p50 | recall@{k} |", "|---:|---:|---:|---:|"]
        for r in real["hnsw"]["sweep"]:
            out.append(f"| {r['ef_search']} | {r['p50_ms']:.2f} ms | "
                       f"{real['brute']['p50_ms']:.2f} ms | {r['recall_at_k']:.0%} |")
        out += ["", f"색인 구축 {real['hnsw']['build_sec']:.1f}s", "",
                "★ **합성 벡터로는 이 값을 재면 안 된다.** 768차원 정규분포 벡터는 서로 거의 "
                "직교해 최근접 이웃 구조가 없고, 그러면 근사 탐색이 붙잡을 그래프가 없어 "
                "recall 이 바닥으로 나온다. 그건 HNSW 의 성질이 아니라 데이터의 성질이다 - "
                "이 실험에서 먼저 그 함정에 빠졌다.", ""]

    has_ann = any("hnsw" in r for r in res["results"])
    if has_ann:
        out += ["", "## HNSW 와의 교환 — 빨라지는 대신 무엇을 잃나", "",
                f"`recall@{k}` 는 **전수 매칭이 고른 것 대비** 일치율이다. 1.00 이 아니면 "
                "그만큼 정답을 놓친 것이고, 그 손실은 검색 품질에 그대로 반영된다.", "",
                f"| N | ef_search | HNSW p50 | 전수매칭 p50 | recall@{k} |",
                "|---:|---:|---:|---:|---:|"]
        for r in res["results"]:
            if "hnsw" not in r:
                continue
            for s in r["hnsw"]["sweep"]:
                out.append(f"| {r['n']:,} | {s['ef_search']} | {s['p50_ms']:.2f} ms | "
                           f"{r['brute']['p50_ms']:.2f} ms | {s['recall_at_k']:.0%} |")
        out += ["", "색인 구축 비용: "
                + " · ".join(f"N={r['n']:,} → {r['hnsw']['build_sec']:.1f}s"
                             for r in res["results"] if "hnsw" in r), ""]

    out += ["", "## 한계", "",
            "- 지연·메모리는 합성으로 쟀다(행렬곱 비용은 값이 아니라 모양이 정한다). "
            "recall 은 합성으로 재면 안 되므로 **실제 임베딩**으로 따로 쟀다.",
            "- 실측 recall 은 **현재 코퍼스 크기에서의** 값이다. 코퍼스가 커지면 다시 재야 한다.",
            "- 메모리는 프로세스가 실제로 쓸 수 있는 양에 달렸다 — 같은 N 이라도 기계가 다르면 "
            "다르게 무너진다.", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="ANN 전환 임계 실증")
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--ef", default="16,64,256", help="HNSW ef_search 스윕 값")
    ap.add_argument("--no-ann", action="store_true", help="전수 매칭만 측정")
    ap.add_argument("--real", action="store_true",
                    help="실제 코퍼스 임베딩으로 HNSW recall 측정(품질 손실은 여기서만 유효)")
    ap.add_argument("--profile", default=None, help="--real 에서 쓸 코퍼스 프로필")
    ap.add_argument("--synthetic-recall", action="store_true",
                    help="합성 벡터로도 recall 을 재 본다(오독 위험 - 진단용)")
    ap.add_argument("--ann-max-n", type=int, default=100_000,
                    help="이보다 큰 N 에서는 HNSW 색인 구축을 건너뛴다(구축이 질의보다 비싸진다)")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    ef = [int(e) for e in args.ef.split(",") if e.strip()]
    if args.profile:
        from app.profiles import use_profile

        use_profile(args.profile)

    res = measure(sizes, args.dim, args.k, args.seed, not args.no_ann, ef, args.ann_max_n,
                  args.synthetic_recall)
    if args.real and not args.no_ann:
        res["real"] = measure_real(args.k, ef)

    md = render(res)
    print("\n" + md)
    rp.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (rp.REPORT_DIR / "ann-threshold.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    (rp.REPORT_DIR / "ann-threshold.md").write_text(md, encoding="utf-8")
    print(f"[report] 저장: {rp.REPORT_DIR / 'ann-threshold.md'}")


if __name__ == "__main__":
    main()
