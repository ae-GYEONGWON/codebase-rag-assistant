"""실사용형 평가셋 채점 — "평가셋이 실제 질문보다 쉬웠나"를 통제 비교로 잰다.

## 왜 별도 러너인가

`run_eval.py` 는 골든셋(`questions.demo.json`)으로 회귀 게이트의 기준선을 만든다.
거기에 문항을 더하면 기준선이 함께 움직여, **재려던 것(질문 말씨의 영향)과
게이트가 재는 것(코드 변경의 영향)이 한 숫자에 섞인다.** 그래서 파일도 러너도
따로 둔다 — `hard_eval.py` 가 적대적 문항을 따로 둔 것과 같은 이유다.

## 무엇을 통제하는가

    같음:  코퍼스(eval-corpus-v1 고정 스냅샷) · 검색기(운영 파이프라인) · k · 채점 방식
    다름:  질문의 말씨뿐

골든셋 문항은 전부 **이미 시스템을 아는 사람**이 썼다.

    골든셋   "SL 은 **코드에서** 어떻게 계산돼?"      ← 볼 축을 질문이 알려준다
    실사용   "sl이 뭐야?"                              ← 아무것도 안 알려준다

그래서 **축 판별 unknown 비율**도 함께 센다. 라우팅이 좋아서 개입이 적었던 건지,
문항이 축을 미리 말해 줘서 라우팅이 할 일이 없었던 건지가 이 숫자에서 갈린다.

★ 그런데 말씨 말고 한 가지가 더 다르다. 실사용형은 정답 파일을 문항당 평균 **2.1개**,
골든셋은 **1.2개** 준다(`expected` 는 OR 채점이라 많을수록 맞히기 쉽다). 말씨를 재려다
**라벨 관대함**을 같이 재면 두 효과가 한 숫자에 섞인다. 그래서 두 벌을 함께 낸다.

    원본 라벨   — 각 문항이 실제로 가진 정답 파일 전부(OR)
    첫 라벨만   — 양쪽 모두 `expected[0]` 하나로 잘라 **라벨 수를 맞춘** 보수적 하한

실행:
    python -m eval.realistic_eval --profile eval
    python -m eval.realistic_eval --profile eval --k 3
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

from app.intent import classify
from app.profiles import available_profiles, use_profile
from eval import report as rp
from eval.datasets import load_questions
from eval.run_eval import retrieve_hybrid, score

REALISTIC_PATH = Path(__file__).resolve().parent / "questions.realistic.json"

# 리포트에 쓰는 이름. 내부 kind 값을 그대로 찍으면 처음 보는 사람에게 기호일 뿐이다.
KIND_LABEL = OrderedDict([
    ("definition", "개념을 묻는 질문"),
    ("outsider-word", "코퍼스에 없는 말로 묻는 질문"),
    ("no-axis", "어디를 볼지 안 밝힌 질문"),
])

HEAD = ("| 질문셋 | 문항 | 라벨/문항 | recall@%d | MRR | 축 판별 실패 |\n"
        "|---|---:|---:|---:|---:|---:|")


def load_cases(path: Path | None = None) -> List[dict]:
    data = json.loads((path or REALISTIC_PATH).read_text(encoding="utf-8"))
    return data["cases"]


def golden_cases() -> List[dict]:
    """대조군 — 기존 골든셋의 단일 홉 문항(문서 + 코드)."""
    qs = load_questions()
    return [*qs.in_scope, *qs.in_scope_code]


def primary_only(cases: List[dict]) -> List[dict]:
    """`expected` 를 첫 라벨 하나로 자른다 — 두 질문셋의 라벨 수를 맞추기 위한 것.

    관대한 라벨은 recall 을 올린다. 실사용형이 골든셋보다 라벨이 많으므로, 이 통제 없이는
    '말씨 때문에 어려웠나'를 '라벨이 많아서 쉬웠나'와 구분할 수 없다.
    """
    return [{**c, "expected": c["expected"][:1]} for c in cases]


def label_density(cases: List[dict]) -> float:
    """문항당 평균 정답 파일 수. 두 셋의 난도를 비교할 때 같이 봐야 하는 값."""
    return sum(len(c["expected"]) for c in cases) / (len(cases) or 1)


def unknown_rate(questions: List[str]) -> float:
    """축 판별이 '모르겠음'으로 떨어진 비율."""
    if not questions:
        return 0.0
    return sum(1 for q in questions if classify(q).axis is None) / len(questions)


def stats(cases: List[dict], k: int) -> dict:
    recall, mrr, misses = score(retrieve_hybrid, cases, k)
    return {
        "n": len(cases),
        "recall": recall,
        "mrr": mrr,
        "misses": misses,
        "unknown": unknown_rate([c["q"] for c in cases]),
        "labels": label_density(cases),
    }


def by_kind(cases: List[dict], k: int) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for kind in KIND_LABEL:
        sub = [c for c in cases if c["kind"] == kind]
        if sub:
            out[kind] = stats(sub, k)
    return out


def _row(name: str, s: dict) -> str:
    return (f"| {name} | {s['n']} | {s['labels']:.1f} | {s['recall']:.0%} | "
            f"{s['mrr']:.2f} | {s['unknown']:.0%} |")


def render(rows: Dict[str, dict], kinds: Dict[str, dict], k: int) -> str:
    lines = [
        "# 실사용형 평가셋 — 말씨만 바꾼 통제 비교",
        "",
        f"코퍼스(eval-corpus-v1) · 검색기(운영 파이프라인) · k(={k}) · 채점 방식이 모두 같다.",
        "**다른 것은 질문의 말씨뿐이다.**",
        "",
        "## 원본 라벨 (문항이 실제로 가진 정답 파일 전부)",
        "",
        HEAD % k,
        _row("골든셋(대조군)", rows["golden"]),
        _row("**실사용형**", rows["realistic"]),
        "",
        "## 첫 라벨만 — 라벨 수를 맞춘 보수적 하한",
        "",
        "`expected` 를 양쪽 모두 하나로 잘랐다. 위 표의 recall 차이에 **라벨 관대함**이",
        "섞여 있어서, 그걸 제거하고 다시 본 값이다.",
        "",
        HEAD % k,
        _row("골든셋(대조군)", rows["golden_p"]),
        _row("**실사용형**", rows["realistic_p"]),
        "",
        "## 실사용형 — 종류별 (원본 라벨)",
        "",
        HEAD % k,
    ]
    lines += [_row(KIND_LABEL[kind], s) for kind, s in kinds.items()]
    if rows["realistic"]["misses"]:
        lines += ["", "## 못 찾은 문항", ""]
        lines += [f"- {q}" for q in rows["realistic"]["misses"]]
    lines += ["", f"생성: {rp._git_sha()}", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=available_profiles(), default=None)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    if args.profile:
        use_profile(args.profile)

    k = args.k
    cases, golden = load_cases(), golden_cases()
    rows = {
        "golden": stats(golden, k),
        "realistic": stats(cases, k),
        "golden_p": stats(primary_only(golden), k),
        "realistic_p": stats(primary_only(cases), k),
    }
    kinds = by_kind(cases, k)

    def line(name: str, s: dict) -> None:
        pad = 30 - (len(name) - len(name.encode("ascii", "ignore").decode()))
        print(f"{name:<{pad}}{s['n']:>5}{s['labels']:>7.1f}{s['recall']:>9.0%}"
              f"{s['mrr']:>7.2f}{s['unknown']:>10.0%}")

    print(f"\n■ 말씨만 바꾼 통제 비교 (k={k})\n")
    print(f"{'질문셋':<30}{'문항':>5}{'라벨':>7}{'recall':>9}{'MRR':>7}{'축판별실패':>11}")
    print("-" * 72)
    print("[원본 라벨]")
    line("골든셋(대조군)", rows["golden"])
    line("실사용형", rows["realistic"])
    for kind, s in kinds.items():
        line("  └ " + KIND_LABEL[kind], s)
    print("\n[첫 라벨만 — 라벨 수를 맞춘 하한]")
    line("골든셋(대조군)", rows["golden_p"])
    line("실사용형", rows["realistic_p"])

    if rows["realistic"]["misses"]:
        print("\n못 찾은 문항:")
        for q in rows["realistic"]["misses"]:
            print("  -", q)

    rp.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = rp.REPORT_DIR / "realistic-eval.md"
    out.write_text(render(rows, kinds, k), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
