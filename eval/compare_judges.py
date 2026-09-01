"""판정자 비교 — 자체 판정기 vs RAGAS, 그리고 self-judge vs cross-judge.

## 이 파일이 답하려는 질문

groundedness 0.962 라는 숫자의 문제는 값이 아니라 **출처**였다. 생성한 모델이 자기 답을
채점했으므로(self-enhancement bias), 그 값이 얼마나 후한지 모르면 쓸 수가 없다.

두 방향으로 잰다.

1. **다른 도구로 재기** — 같은 답변을 RAGAS 의 faithfulness 로 다시 채점해 대조.
   `eval/faithfulness.py` 가 RAGAS faithfulness 의 수기 재구현이므로, 값이 갈리면
   **무엇을 다르게 보는지**가 자체 구현의 한계를 드러낸다.
2. **다른 모델로 재기** — 판정 모델만 바꿔(cross-judge) 같은 답변을 채점해 편차를 낸다.

## 읽는 법

평균 차이보다 **문항별 차이의 분포**가 중요하다. 평균이 같아도 문항마다 크게 엇갈리면
두 판정자는 서로 다른 것을 재고 있는 것이고, 그러면 어느 한쪽 값을 단독으로 인용할 수 없다.

    python -m eval.compare_judges                          # 자체 vs RAGAS
    python -m eval.compare_judges --a self --b cross       # self vs cross judge
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPORT_DIR = Path(__file__).resolve().parent / "reports"


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"파일이 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _scores_from_faithfulness(d: dict) -> Dict[str, Optional[float]]:
    return {r["q"]: r.get("score") for r in d.get("records", [])}


def _scores_from_ragas(d: dict) -> Dict[str, Optional[float]]:
    return {r["q"]: r.get("ragas_faithfulness") for r in d.get("scores", [])}


def _load_scores(path: Path) -> Tuple[str, Dict[str, Optional[float]]]:
    d = _load(path)
    if "records" in d:
        return f"{d.get('judge', '?')}{' (self)' if d.get('self_judge') else ''}", _scores_from_faithfulness(d)
    return f"RAGAS/{d.get('judge_model', '?')}", _scores_from_ragas(d)


def _stats(pairs: List[Tuple[str, float, float]]) -> dict:
    n = len(pairs)
    diffs = [b - a for _, a, b in pairs]
    mean_a = sum(a for _, a, _ in pairs) / n
    mean_b = sum(b for _, _, b in pairs) / n
    mad = sum(abs(d) for d in diffs) / n
    within = sum(1 for d in diffs if abs(d) <= 0.1) / n
    big = sorted(pairs, key=lambda p: -abs(p[2] - p[1]))[:5]
    return {"n": n, "mean_a": mean_a, "mean_b": mean_b, "mean_diff": mean_b - mean_a,
            "mad": mad, "within_0.1": within, "biggest": big}


def main() -> None:
    ap = argparse.ArgumentParser(description="판정자 비교")
    ap.add_argument("--a", type=Path, default=REPORT_DIR / "faithfulness-self.json",
                    help="기준 판정 결과(JSON)")
    ap.add_argument("--b", type=Path, default=REPORT_DIR / "ragas-faithfulness.json",
                    help="비교 대상 판정 결과(JSON)")
    args = ap.parse_args()

    label_a, sa = _load_scores(args.a)
    label_b, sb = _load_scores(args.b)

    pairs = [(q, sa[q], sb[q]) for q in sa
             if q in sb and sa[q] is not None and sb[q] is not None]
    if not pairs:
        raise SystemExit("겹치는 채점 문항이 없습니다(같은 평가셋·같은 답변으로 만든 결과인지 확인).")

    st = _stats(pairs)
    print(f"\n■ 판정자 비교 — 공통 {st['n']}문항\n")
    print(f"   A: {label_a}   평균 {st['mean_a']:.3f}")
    print(f"   B: {label_b}   평균 {st['mean_b']:.3f}")
    print(f"\n   ▸ 평균 차이(B-A)      {st['mean_diff']:+.3f}")
    print(f"   ▸ 문항별 평균 절대차   {st['mad']:.3f}   ← 평균이 같아도 이 값이 크면 서로 다른 걸 재는 것")
    print(f"   ▸ 0.1 이내 일치 비율   {st['within_0.1']:.0%}")
    print("\n   가장 크게 갈린 문항:")
    for q, a, b in st["biggest"]:
        print(f"     {a:.2f} → {b:.2f}  ({b - a:+.2f})  {q[:52]}")

    if st["mean_diff"] < -0.05:
        print(f"\n   해석: B 가 더 박하다. A 의 값은 최소 {abs(st['mean_diff']):.3f} 만큼 후하게 잡혔을 수 있다.")
    elif st["mean_diff"] > 0.05:
        print("\n   해석: B 가 더 후하다. 두 판정자의 기준이 다르므로 절대값 인용은 피할 것.")
    else:
        print("\n   해석: 평균은 근접. 단, 위 '평균 절대차'가 크면 문항 단위로는 일치하지 않는다.")

    out = REPORT_DIR / "judge-comparison.json"
    out.write_text(json.dumps({
        "a": {"file": args.a.name, "label": label_a, "mean": st["mean_a"]},
        "b": {"file": args.b.name, "label": label_b, "mean": st["mean_b"]},
        "n": st["n"],
        "mean_diff": st["mean_diff"],
        "mean_abs_diff": st["mad"],
        "within_0_1": st["within_0.1"],
        "biggest_gaps": [{"q": q, "a": a, "b": b} for q, a, b in st["biggest"]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[report] 저장: {out}\n")


if __name__ == "__main__":
    main()
