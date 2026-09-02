"""판정기 패널 — **같은 답변**을 여러 판정기에 걸어 self vs cross 편차를 수치화한다(Phase 0-3).

## 왜 기존 방식으로는 안 되나

`faithfulness.py --judge-model X` 를 두 번 돌리면 판정기는 바뀌지만 **답변도 매번 새로
생성된다.** 그러면 점수 차이가 판정기 탓인지 답변 탓인지 나눌 수 없다. 통제가 아니다.

그래서 절차를 둘로 자른다.

    freeze  — 답변을 한 번 만들어 (질문, 근거, 답변) 을 파일에 고정한다.  [생성기 호출 N회]
    judge   — 그 고정된 입력을 판정기에 건다.                              [판정기 호출 N회]

이러면 두 판정 결과의 차이는 **오직 판정기 차이**다. 같은 freeze 파일을 쓰는 한
몇 달 뒤 다른 모델을 붙여도 그대로 비교된다.

## 무료 티어 제약이 표본을 정한다

`gemini-3.5-flash` 무료 티어는 **하루 20요청**이다(노트 #20). 그래서 기본 표본을 16문항으로
두고, 쿼터가 끊기면 거기까지를 부분 결과로 보존한다. n=16 은 작다 — 그래서 평균 차이에
**부트스트랩 신뢰구간**을 붙인다. 구간이 0 을 걸치면 "차이가 있다"고 말하지 않는다.

## 무엇을 세는가

평균 차이만 보면 안 된다. 평균이 같아도 문항마다 엇갈리면 두 판정기는 **다른 것을 재고 있다.**

    mean_diff        판정기의 관대함 차이(계통 편향)
    MAD              문항별 평균 절대차 — 이게 크면 평균 일치는 우연이다
    Spearman rho     순위 일치. 절대값은 달라도 순서가 같으면 '상대 비교'에는 쓸 수 있다
    Cohen's kappa    "환각 있음/없음" 이분 판정의 우연 초과 일치도

## 정직한 한계

키가 Gemini 하나뿐이라 판정기 둘 다 같은 계열이다. **모델 독립이 아니라 버전 독립**이고,
여기서 나온 편차는 진짜 판정기 독립성의 **하한**이다. 다른 벤더 키가 생기면 같은 freeze
파일에 `--provider openai` 로 한 줄 더 얹으면 된다.

실행:
    python -m eval.judge_panel freeze  --profile eval --n 16
    python -m eval.judge_panel judge   --profile eval --model gemini-3.1-flash-lite --tag self
    python -m eval.judge_panel judge   --profile eval --model gemini-3.5-flash      --tag cross
    python -m eval.judge_panel compare
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from app.profiles import available_profiles, use_profile
from eval import llm as L
from eval import report as rp
from eval.datasets import load_questions

# 판정 프롬프트는 faithfulness 와 **같은 것**을 쓴다. 프롬프트가 다르면 판정기 비교가 아니라
# 프롬프트 비교가 된다.
from eval.faithfulness import JUDGE_PROMPT

FREEZE_PATH = rp.REPORT_DIR / "panel-answers.json"
PANEL_PATH = rp.REPORT_DIR / "judge-panel.json"
PANEL_MD = rp.REPORT_DIR / "judge-panel.md"

DEFAULT_N = 16       # gemini-3.5-flash 무료 티어 20/day 아래로 여유를 둔 값
DEFAULT_SEED = 20260902


# ---------------------------------------------------------------- freeze


def freeze(qs, n: int, seed: int) -> dict:
    """답변을 한 번 생성해 고정한다. 이후 모든 판정은 이 파일만 본다."""
    from app.rag import _format_context, answer
    from app.retriever import search

    # 적대적 문항이 있으면 **그것만** 쓴다. 평범한 문항과 섞으면 다시 천장에 붙어
    # 판정기 편차가 묻힌다 — 노트 #21 이 진단한 것이 정확히 그 희석이었다.
    if qs.hard:
        cases: List[dict] = [dict(c, bucket=c.get("kind", "hard")) for c in qs.hard]
        print(f"[freeze] 적대적 평가셋 {len(cases)}문항을 대상으로 한다(평범한 문항과 섞지 않음)")
    else:
        cases = [dict(c, bucket="in_scope") for c in qs.in_scope]
        cases += [dict(c, bucket="in_scope_code") for c in qs.in_scope_code]
    rng = random.Random(seed)
    rng.shuffle(cases)
    picked = cases[: max(n - 1, 1)]
    # 범위 밖 질문 하나를 반드시 섞는다 — "거절도 충실한 답"을 두 판정기가 같게 보는지가
    # 가장 갈리기 쉬운 지점이다.
    if qs.out_of_scope:
        picked.append({"q": qs.out_of_scope[0], "expected": [], "bucket": "out_of_scope"})

    gen = L.generator_spec()
    print(f"[freeze] 생성기={gen.provider}:{gen.model} · {len(picked)}문항 (seed={seed})")
    items: List[dict] = []
    for i, c in enumerate(picked, 1):
        q = c["q"]
        try:
            res = L.call(lambda: answer(q))
        except Exception as e:  # noqa: BLE001 — 쿼터 소진 등. 여기까지를 보존한다.
            print(f"[freeze] {i}/{len(picked)} 중단: {type(e).__name__} {str(e)[:90]}")
            break
        docs, _ = search(q)
        items.append({
            "id": f"q{i:02d}",
            "q": q,
            "bucket": c["bucket"],
            "expected": c.get("expected", []),
            "answer": res["answer"],
            # 자체 판정기는 생성기가 본 것과 같은 형식(파일명·헤더 포함)의 문자열을 받고,
            # RAGAS 는 청크 리스트를 받는다. 같은 근거를 두 형태로 함께 고정해 둔다 —
            # 판정기마다 근거를 새로 만들면 그것도 통제가 깨진다.
            "context": _format_context(docs) if docs else "",
            "contexts": [d.page_content for d in docs],
            "sources": [d.metadata.get("source") for d in docs],
        })
        print(f"[freeze] {i}/{len(picked)} {c['bucket']:14} {q[:46]}")

    return {
        "generator": f"{gen.provider}:{gen.model}",
        "profile": qs.profile,
        "questions": qs.path.name,
        "seed": seed,
        "n": len(items),
        "items": items,
    }


# ---------------------------------------------------------------- judge


def judge_all(frozen: dict, spec: L.ModelSpec, max_calls: Optional[int]) -> dict:
    items = frozen["items"]
    if max_calls:
        items = items[:max_calls]
    print(f"[judge] {spec.provider}:{spec.model} · {len(items)}문항")
    scores: List[dict] = []
    for i, it in enumerate(items, 1):
        try:
            raw = L.ask(spec, JUDGE_PROMPT.format(context=it["context"][:12000], answer=it["answer"]))
        except Exception as e:  # noqa: BLE001 — 하루 20요청 소진이 여기로 온다
            print(f"[judge] {i}/{len(items)} 중단: {type(e).__name__} {str(e)[:90]}")
            print(f"[judge] 여기까지 {len(scores)}건을 보존한다 — compare 는 공통 문항만 쓴다.")
            break
        obj = L.parse_json(raw) or {}
        try:
            s = float(obj["score"])
        except (KeyError, TypeError, ValueError):
            s = None
        unsupported = obj.get("unsupported", []) if isinstance(obj, dict) else []
        scores.append({
            "id": it["id"], "q": it["q"], "bucket": it["bucket"],
            "score": s, "unsupported": unsupported,
        })
        shown = f"{s:.2f}" if s is not None else " -- "
        print(f"[judge] {i}/{len(items)} {shown:>5} 환각{len(unsupported):>2}  {it['q'][:44]}")

    ok = [s["score"] for s in scores if s["score"] is not None]
    return {
        "judge": f"{spec.provider}:{spec.model}",
        "generator": frozen["generator"],
        "self_judge": f"{spec.provider}:{spec.model}" == frozen["generator"],
        "freeze_seed": frozen["seed"],
        "profile": frozen["profile"],
        "n": len(ok),
        "n_unparsed": len(scores) - len(ok),
        "mean_score": (sum(ok) / len(ok)) if ok else None,
        "scores": scores,
    }


# ---------------------------------------------------------------- 통계


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """순위 상관. 동점은 평균 순위로 처리하고, 한쪽이 전부 동점이면 정의되지 않는다(None)."""

    def rank(v: Sequence[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None       # 한쪽이 전부 같은 점수 — 순위 자체가 없다
    return num / (dx * dy)


def _kappa(a: Sequence[bool], b: Sequence[bool]) -> Optional[float]:
    """Cohen's kappa — 이분 판정('환각 있음')의 우연 초과 일치도.

    두 판정기가 우연히 맞을 확률을 빼고 남은 일치를 센다.

    **한쪽 판정기가 모든 문항에 같은 판정을 내리면 None 을 돌려준다.** 이때 kappa 는
    상대가 무엇을 하든 항상 0 이 나오는데(kappa 역설), 그 0 을 "우연 수준의 일치"로
    읽으면 틀린다. 잴 수 없는 것이지 잰 결과가 0 인 게 아니다. 실제로 판정기가 모든
    답변을 1.0 으로 주는 일은 흔하므로 이 구분이 필요하다.
    """
    n = len(a)
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if abs(1 - pe) < 1e-12:
        return None
    return (po - pe) / (1 - pe)


def _bootstrap_ci(diffs: Sequence[float], iters: int = 5000, seed: int = 7) -> Tuple[float, float]:
    """평균 차이의 95% 부트스트랩 신뢰구간.

    n=16 에서 '차이가 있다'를 함부로 말하지 않기 위한 장치. 구간이 0 을 걸치면 보류한다.
    """
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


MIN_PAIR_N = 3        # 이보다 적으면 비교 자체를 하지 않는다
# 무료 티어에서 큰 모델은 하루 4~5회로 끊긴다. 그 표본도 버리지는 않되,
# **유의성은 주장하지 않는다.** 부트스트랩 구간이 0 을 안 걸쳐도 n 이 작으면 그건 우연이다.
MIN_SIGNIFICANT_N = 8


def compare(files: List[Path], threshold: float) -> dict:
    """판정 결과들을 **쌍별 교집합**으로 비교한다.

    전체 교집합을 쓰면 한 판정기가 못 채점한 문항(RAGAS 는 근거 없는 범위밖 질문을
    채점할 수 없고, 쿼터가 끊긴 판정기는 뒷부분이 통째로 없다) 때문에 **무관한 쌍의
    표본까지 깎인다.** 쌍마다 자기 교집합을 쓰고 n 을 함께 적는다.
    """
    loaded = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    maps = [{s["id"]: s["score"] for s in d["scores"] if s["score"] is not None} for d in loaded]

    per_judge = [{
        "file": p.name, "judge": d["judge"], "self_judge": d["self_judge"],
        "n": len(m),
        "mean": (sum(m.values()) / len(m)) if m else None,
        "flag_rate": (sum(1 for v in m.values() if v < threshold) / len(m)) if m else None,
    } for p, d, m in zip(files, loaded, maps)]

    qtext: dict = {}
    for d in loaded:
        for s in d["scores"]:
            qtext.setdefault(s["id"], s["q"])

    pairs, skipped = [], []
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            common = sorted(set(maps[i]) & set(maps[j]))
            label = f"{loaded[i]['judge']} vs {loaded[j]['judge']}"
            if len(common) < MIN_PAIR_N:
                skipped.append({"pair": label, "n_common": len(common)})
                continue
            xs = [maps[i][c] for c in common]
            ys = [maps[j][c] for c in common]
            diffs = [y - x for x, y in zip(xs, ys)]
            lo, hi = _bootstrap_ci(diffs)
            biggest = sorted(zip(common, xs, ys), key=lambda t: -abs(t[2] - t[1]))[:5]
            pairs.append({
                "a": loaded[i]["judge"], "b": loaded[j]["judge"],
                "n_common": len(common),
                "mean_a": sum(xs) / len(xs), "mean_b": sum(ys) / len(ys),
                "mean_diff": sum(diffs) / len(diffs),
                "ci95": [lo, hi],
                "underpowered": len(common) < MIN_SIGNIFICANT_N,
                "significant": not (lo <= 0 <= hi) and len(common) >= MIN_SIGNIFICANT_N,
                "mad": sum(abs(d) for d in diffs) / len(diffs),
                "max_abs_diff": max(abs(d) for d in diffs),
                "within_0_1": sum(1 for d in diffs if abs(d) <= 0.1) / len(diffs),
                "exact_agree": sum(1 for d in diffs if d == 0) / len(diffs),
                "spearman": _spearman(xs, ys),
                "kappa": _kappa([x < threshold for x in xs], [y < threshold for y in ys]),
                "biggest_gaps": [{"id": c, "a": x, "b": y} for c, x, y in biggest],
            })

    if not pairs:
        raise SystemExit(
            f"비교 가능한 쌍이 없습니다(공통 문항 {MIN_PAIR_N}건 미만) — "
            "같은 freeze 파일로 판정했는지, 쿼터로 중단되지 않았는지 확인하세요.")

    return {
        "threshold": threshold,
        "generator": loaded[0]["generator"],
        "judges": per_judge,
        "pairs": pairs,
        "skipped_pairs": skipped,
        "questions": qtext,
        "caveat": "판정기가 모두 Gemini 계열이다 — 모델 독립이 아니라 버전 독립. 여기 편차는 하한.",
    }


def _fmt(v: Optional[float], spec: str = ".3f") -> str:
    return "n/a" if v is None else format(v, spec)


def render(res: dict) -> str:
    t = res["threshold"]
    out = ["# 판정기 패널 — 같은 답변, 다른 판정기 (통제 비교)", "",
           f"생성기 `{res['generator']}` 가 만든 **동일한 답변**을 판정기만 바꿔 채점했다.",
           "답변·근거가 고정돼 있으므로 아래 차이는 전부 판정기 차이다.", "",
           "| 판정기 | 종류 | 채점 | 평균 groundedness | 환각 지적 비율 |",
           "|---|---|---:|---:|---:|"]
    for j in res["judges"]:
        kind = "self" if j["self_judge"] else "cross"
        flag = "n/a" if j["flag_rate"] is None else format(j["flag_rate"], ".0%")
        out.append(f"| `{j['judge']}` | {kind} | {j['n']} | {_fmt(j['mean'])} | {flag} |")
    out += ["", f"환각 지적 = score < {t}", "", "## 쌍별 편차", ""]
    for p in res["pairs"]:
        lo, hi = p["ci95"]
        if p["underpowered"]:
            verdict = f"**표본 부족** (n<{MIN_SIGNIFICANT_N}) — 참고용, 유의성 주장 안 함"
        elif p["significant"]:
            verdict = "**유의** (구간이 0 을 안 걸침)"
        else:
            verdict = "판단 보류 (구간이 0 을 걸침)"
        tail = "  ⚠️ 쿼터로 표본 부족" if p["underpowered"] else ""
        out += [f"### `{p['a']}` → `{p['b']}`  (공통 {p['n_common']}문항){tail}", "",
                "| 지표 | 값 | 읽는 법 |", "|---|---:|---|",
                f"| 평균 | {p['mean_a']:.3f} → {p['mean_b']:.3f} | |",
                f"| 평균 차이 | {p['mean_diff']:+.3f} | 계통 편향(관대함 차이) |",
                f"| 95% 부트스트랩 CI | [{lo:+.3f}, {hi:+.3f}] | {verdict} |",
                f"| 문항별 평균 절대차 | {p['mad']:.3f} | 평균이 같아도 이게 크면 서로 다른 걸 잰다 |",
                f"| 최대 절대차 | {p['max_abs_diff']:.2f} | 한 문항에서 벌어질 수 있는 폭 |",
                f"| 0.1 이내 일치 | {p['within_0_1']:.0%} | |",
                f"| 완전 동점 | {p['exact_agree']:.0%} | |",
                f"| Spearman rho | {_fmt(p['spearman'])} | 순위 일치. n/a = 한쪽이 전부 동점 |",
                f"| Cohen's kappa | {_fmt(p['kappa'])} | 이분 판정의 우연 초과 일치. n/a = 한쪽이 전부 같은 판정 |",
                "", "가장 크게 갈린 문항:", ""]
        for g in p["biggest_gaps"]:
            q = res["questions"].get(g["id"], "")[:60]
            out.append(f"- `{g['id']}` {g['a']:.2f} → {g['b']:.2f} ({g['b'] - g['a']:+.2f}) — {q}")
        out.append("")
    if res["skipped_pairs"]:
        out += ["### 비교하지 못한 쌍", ""]
        for s in res["skipped_pairs"]:
            out.append(f"- {s['pair']} — 공통 {s['n_common']}문항 (쿼터 중단 등으로 표본 부족)")
        out.append("")
    out += ["## 한계", "", res["caveat"], "",
            "표본 크기는 무료 티어 일일 쿼터가 정한 상한이다 — 그래서 평균 차이에 "
            "부트스트랩 신뢰구간을 붙였고, 구간이 0 을 걸치면 차이를 주장하지 않는다.", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(description="판정기 패널(동일 표본 통제 비교)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("freeze", help="답변을 생성해 고정(이후 판정은 이 파일만 본다)")
    f.add_argument("--profile", choices=available_profiles())
    f.add_argument("--questions", type=Path, default=None)
    f.add_argument("--n", type=int, default=DEFAULT_N, help=f"표본 크기(기본 {DEFAULT_N}, 범위밖 1 포함)")
    f.add_argument("--seed", type=int, default=DEFAULT_SEED)
    f.add_argument("--out", type=Path, default=FREEZE_PATH)

    j = sub.add_parser("judge", help="고정된 답변을 판정기 하나로 채점")
    j.add_argument("--profile", choices=available_profiles())
    j.add_argument("--provider", default="gemini", choices=["gemini", "openai"])
    j.add_argument("--model", required=True)
    j.add_argument("--tag", required=True, help="리포트 파일명 꼬리표(self / cross / ...)")
    j.add_argument("--freeze", type=Path, default=FREEZE_PATH)
    j.add_argument("--max-calls", type=int, default=None, help="쿼터 보호용 호출 상한")

    c = sub.add_parser("compare", help="판정 결과 2개 이상을 비교")
    c.add_argument("files", nargs="*", type=Path, help="기본: reports/panel-judge-*.json 전부")
    c.add_argument("--threshold", type=float, default=1.0,
                   help="'환각 있음' 이분 기준(기본 1.0 = 하나라도 지적되면 환각)")
    c.add_argument("--out", default="judge-panel",
                   help="리포트 파일명(확장자 제외). 평범한 셋과 적대적 셋의 비교를 "
                        "나란히 남기려면 서로 다른 이름을 줘야 한다")

    args = ap.parse_args()
    rp.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if getattr(args, "profile", None):
        use_profile(args.profile)

    if args.cmd == "freeze":
        qs = load_questions(args.questions)
        print(f"[freeze] {qs.summary()}")
        res = freeze(qs, args.n, args.seed)
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[freeze] 저장: {args.out}  ({res['n']}문항)")
        print("  다음: python -m eval.judge_panel judge --model <모델> --tag <꼬리표>")
        return

    if args.cmd == "judge":
        if not args.freeze.exists():
            raise SystemExit(f"freeze 파일이 없습니다: {args.freeze}  (먼저 freeze 를 실행)")
        frozen = json.loads(args.freeze.read_text(encoding="utf-8"))
        spec = L.ModelSpec(args.provider, args.model, "판정기")
        if not spec.available:
            raise SystemExit(f"{args.provider} 키가 없습니다(.env 확인).")
        res = judge_all(frozen, spec, args.max_calls)
        out = rp.REPORT_DIR / f"panel-judge-{args.tag}.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        m = res["mean_score"]
        print(f"\n[judge] 평균 {_fmt(m)} (n={res['n']}, 파싱실패 {res['n_unparsed']})")
        print(f"[judge] 저장: {out}")
        return

    files = args.files or sorted(rp.REPORT_DIR.glob("panel-judge-*.json"))
    if len(files) < 2:
        raise SystemExit("판정 결과가 2개 미만입니다 — judge 를 다른 --tag 로 한 번 더 돌리세요.")
    res = compare(list(files), args.threshold)
    out_json = rp.REPORT_DIR / f"{args.out}.json"
    out_md = rp.REPORT_DIR / f"{args.out}.md"
    out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render(res)
    out_md.write_text(md, encoding="utf-8")
    print(md)
    print(f"[report] 저장: {out_json} · {out_md}")


if __name__ == "__main__":
    main()
