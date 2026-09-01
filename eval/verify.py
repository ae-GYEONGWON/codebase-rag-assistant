"""합성 평가셋 수동 검수 — 자동 라벨을 사람이 표본 검사해 **불일치율**을 잰다.

## 왜 필요했나

합성 평가셋은 문항 수 문제를 풀지만 **라벨 품질 문제를 새로 만든다.** LLM 이 만든 질문에
"그 청크의 파일"을 정답으로 달았을 뿐이라, 두 종류의 오류가 섞여 있다:

- **답이 그 파일에 없다** — 질문이 청크를 벗어났다(생성 규칙 위반).
- **답이 다른 파일에도 있다** — 검색기가 옳은 파일을 찾아도 miss 로 채점된다(거짓 오답).

두 번째가 특히 위험하다. recall 을 **실제보다 낮게** 보이게 하고, 그 낮은 값을 보고
검색을 고치려 들면 없는 병을 치료하게 된다.

그래서 표본 40~50문항을 사람이 직접 보고 **합성 라벨 vs 수동 라벨 불일치율**을 낸다.
이 숫자가 있어야 합성셋으로 잰 recall 에 오차 막대를 붙일 수 있다.

## 검수 프로토콜 — 워크시트에 검색기 출력을 넣지 않는다

검수자에게 "검색기가 이 파일들을 찾았다"를 보여주면, 사람이 그걸 보고 라벨을 맞추게 되어
**순환이 다시 들어온다**(그러면 검수는 검색기를 정당화하는 절차가 된다).
워크시트에는 **질문과 라벨된 조각만** 싣고, 검수자는 오직 그 둘만 보고 판정한다.

## 사용법

    python -m eval.verify sample --n 50          # 워크시트 생성
    #  → eval/verification/worksheet.md 를 열어 verdict 를 채운다
    python -m eval.verify score                  # 불일치율 계산 + 검수본 저장

판정값(`verdict`):
    ok        조각이 질문에 답한다. 라벨 유효.
    elsewhere 조각도 답하지만 **다른 파일에도 답이 있다** → `also:` 에 파일 적기(거짓 오답 후보)
    wrong     조각이 질문에 답하지 못한다 → 문항 폐기
    unclear   질문 자체가 모호하다 → 문항 폐기
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from app.profiles import available_profiles, use_profile
from eval.datasets import ORIGIN_SYNTHETIC, ORIGIN_SYNTHETIC_VERIFIED

EVAL_DIR = Path(__file__).resolve().parent
SYNTHETIC = EVAL_DIR / "questions.synthetic.json"
VERIFY_DIR = EVAL_DIR / "verification"
WORKSHEET_MD = VERIFY_DIR / "worksheet.md"
WORKSHEET_JSON = VERIFY_DIR / "worksheet.json"
VERIFIED_OUT = EVAL_DIR / "questions.synthetic.verified.json"

BUCKETS = ("in_scope", "in_scope_code", "in_scope_commit")
VALID_VERDICTS = {"ok", "elsewhere", "wrong", "unclear"}

# 폐기 대상 — 이 판정이 붙은 문항은 검수본에서 제외한다.
DROP = {"wrong", "unclear"}


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"파일이 없습니다: {path}\n먼저 `python -m eval.generate` 를 실행하세요.")
    return json.loads(path.read_text(encoding="utf-8"))


def _chunk_text(index: int) -> str:
    from app.retriever import _corpus

    docs, _, _ = _corpus()
    return docs[index].page_content if 0 <= index < len(docs) else "(청크를 찾을 수 없음)"


def cmd_sample(n: int, seed: int) -> None:
    data = _load(SYNTHETIC)
    cases: List[dict] = []
    for b in BUCKETS:
        for c in data.get(b, []):
            cases.append({**c, "_bucket": b})
    if not cases:
        raise SystemExit("합성 문항이 없습니다.")

    # 축별 비율을 유지해 뽑는다 — 코드 문항만 검수하면 문서 축의 라벨 품질을 모른다.
    by_axis: Dict[str, List[dict]] = defaultdict(list)
    for c in cases:
        by_axis[c.get("axis", "doc")].append(c)
    rng = random.Random(seed)
    picked: List[dict] = []
    for axis, items in sorted(by_axis.items()):
        rng.shuffle(items)
        take = max(1, round(n * len(items) / len(cases)))
        picked.extend(items[:take])
    rng.shuffle(picked)
    picked = picked[:n]

    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    WORKSHEET_JSON.write_text(
        json.dumps({"seed": seed, "n": len(picked), "items": picked}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    L: List[str] = []
    L.append("# 합성 평가셋 수동 검수 워크시트")
    L.append("")
    L.append(f"표본 {len(picked)}문항 (전체 {len(cases)}문항에서 축 비율 유지 추출, seed={seed})")
    L.append("")
    L.append("**판정에 필요한 건 화면에 보이는 조각뿐이다.** 저장소를 뒤질 필요 없다.")
    L.append("각 항목의 `verdict:` 줄에 아래 하나를 적고, 다 채운 뒤 `python -m eval.verify score`.")
    L.append("")
    L.append("| 판정 | 언제 |")
    L.append("|---|---|")
    L.append("| `ok` | 아래 조각을 읽고 질문에 답할 수 있다 ← **기본값. 헷갈리면 이걸로** |")
    L.append("| `wrong` | 조각에 질문의 답이 없다(엉뚱한 조각이 붙었다) |")
    L.append("| `unclear` | 질문 자체가 무슨 말인지 모르겠다 |")
    L.append("")
    L.append("한 문항 30초. 오래 고민되면 `ok` 로 두고 넘어갈 것. 빈칸은 집계에서 자동 제외된다.")
    L.append("")
    L.append("> **'다른 파일에도 답이 있나'는 판정하지 않아도 된다.** 그건 저장소 전체를 알아야 하는")
    L.append("> 일이라 사람보다 기계가 낫다 — `eval/audit_misses.py` 가 독립 판정기로 전수 확인한다.")
    L.append("> (아는 경우에만 `elsewhere` + `also:` 로 적으면 그 값도 함께 반영된다.)")
    L.append(">")
    L.append("> 이 워크시트에는 **검색기가 무엇을 찾았는지 넣지 않았다.** 그걸 보고 판정하면")
    L.append("> 검수가 검색기를 정당화하는 절차가 되어 버린다.")
    L.append("")
    L.append("---")
    L.append("")
    for i, c in enumerate(picked, 1):
        L.append(f"## {i}. {c['q']}")
        L.append("")
        L.append(f"- 라벨: `{c['expected'][0]}` · 축 `{c.get('axis')}` · 어휘중복 {c.get('lex_overlap')}")
        L.append("")
        L.append("```")
        L.append(_chunk_text(c.get("chunk_index", -1))[:1500])
        L.append("```")
        L.append("")
        L.append("verdict: ")
        L.append("also: ")  # 선택 — 아는 경우에만
        L.append("")
        L.append("---")
        L.append("")
    WORKSHEET_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[verify] 워크시트 {len(picked)}문항 → {WORKSHEET_MD}")
    print("[verify] verdict 를 채운 뒤: python -m eval.verify score")


def _parse_worksheet() -> List[dict]:
    """워크시트 md 에서 (문항 번호 → verdict, also) 를 읽는다."""
    if not WORKSHEET_MD.exists():
        raise SystemExit(f"워크시트가 없습니다: {WORKSHEET_MD}\n먼저 `sample` 을 실행하세요.")
    items = json.loads(WORKSHEET_JSON.read_text(encoding="utf-8"))["items"]
    text = WORKSHEET_MD.read_text(encoding="utf-8")

    blocks = re.split(r"^## (\d+)\. ", text, flags=re.M)[1:]
    out: List[dict] = []
    for num, body in zip(blocks[0::2], blocks[1::2]):
        idx = int(num) - 1
        if idx >= len(items):
            continue
        # \s* 를 쓰면 줄바꿈을 넘어가 다음 줄(also:)을 판정으로 읽는다 → 같은 줄로 제한.
        v = re.search(r"^verdict:[ 	]*(\S*)", body, re.M)
        a = re.search(r"^also:[ 	]*(.*)$", body, re.M)
        verdict = (v.group(1).strip().lower() if v else "")
        also = [x.strip() for x in re.split(r"[,\s]+", a.group(1).strip())] if a and a.group(1).strip() else []
        out.append({**items[idx], "verdict": verdict, "also": [x for x in also if x]})
    return out


def cmd_score() -> None:
    reviewed = _parse_worksheet()
    filled = [r for r in reviewed if r["verdict"] in VALID_VERDICTS]
    blank = len(reviewed) - len(filled)
    if not filled:
        raise SystemExit("채워진 판정이 없습니다. 워크시트의 verdict 줄을 먼저 채우세요.")

    counts: Dict[str, int] = defaultdict(int)
    for r in filled:
        counts[r["verdict"]] += 1

    n = len(filled)
    disagree = n - counts["ok"]          # ok 가 아니면 자동 라벨과 사람 판단이 갈렸다는 뜻
    unusable = counts["wrong"] + counts["unclear"]
    false_neg = counts["elsewhere"]      # 라벨은 맞지만 다른 파일도 정답 → 거짓 오답 유발

    print(f"\n■ 수동 검수 결과 — 표본 {n}문항 (미기입 {blank})\n")
    for k in ("ok", "elsewhere", "wrong", "unclear"):
        print(f"   {k:<10} {counts[k]:>3}문항  ({counts[k]/n:>5.0%})")
    print(f"\n   ▸ 합성 라벨 불일치율     {disagree/n:.0%}  (사람 판단과 갈린 비율)")
    print(f"   ▸ 폐기 대상(문항 불량)    {unusable/n:.0%}")
    print(f"   ▸ 거짓 오답 유발 비중     {false_neg/n:.0%}  ← recall 을 실제보다 낮게 보이게 한다")
    print(
        f"\n   해석: 합성셋으로 잰 recall 은 최대 {false_neg/n:.0%}p 만큼 과소평가일 수 있다"
        f"(거짓 오답 상한). 이 오차 막대를 붙이지 않은 숫자는 쓰지 말 것."
    )

    # --- 검수 반영본 저장 ---
    data = _load(SYNTHETIC)
    by_q = {r["q"]: r for r in filled}
    out = {k: v for k, v in data.items() if not isinstance(v, list)}
    kept = dropped = fixed = 0
    for b in BUCKETS:
        bucket = []
        for c in data.get(b, []):
            r = by_q.get(c["q"])
            if r is None:
                bucket.append(c)  # 미검수 문항은 그대로(origin 은 synthetic 유지)
                kept += 1
                continue
            if r["verdict"] in DROP:
                dropped += 1
                continue
            c = {**c, "origin": ORIGIN_SYNTHETIC_VERIFIED}
            if r["verdict"] == "elsewhere" and r["also"]:
                # 정답 후보를 넓힌다 — expected 는 OR 채점이므로 거짓 오답이 사라진다
                c["expected"] = sorted(set(c["expected"]) | set(r["also"]))
                fixed += 1
            bucket.append(c)
            kept += 1
        out[b] = bucket
    out["out_of_scope"] = data.get("out_of_scope", [])
    out["_verification"] = {
        "sample_n": n,
        "disagreement_rate": round(disagree / n, 3),
        "unusable_rate": round(unusable / n, 3),
        "false_negative_rate": round(false_neg / n, 3),
        "counts": dict(counts),
    }
    VERIFIED_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verify] 검수 반영본 저장: {VERIFIED_OUT}")
    print(f"[verify] 유지 {kept} · 폐기 {dropped} · 정답 확장 {fixed}")


def main() -> None:
    ap = argparse.ArgumentParser(description="합성 평가셋 수동 검수")
    ap.add_argument("cmd", choices=["sample", "score"])
    ap.add_argument("--n", type=int, default=50, help="검수 표본 크기")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--profile", choices=available_profiles(),
                    help="청크 본문을 읽을 코퍼스 프로필(합성셋을 만든 것과 같아야 함)")
    args = ap.parse_args()
    if args.profile:
        use_profile(args.profile)

    if args.cmd == "sample":
        cmd_sample(args.n, args.seed)
    else:
        cmd_score()


if __name__ == "__main__":
    main()
