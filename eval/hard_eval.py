"""적대적 평가셋 채점 — "환각을 했는가"를 종류별 통과 조건으로 잰다.

## 기존 채점과 무엇이 다른가

`run_eval.py` 는 "맞는 문서를 찾았나"(recall)를, `faithfulness.py` 는 "답이 근거에
지지되나"(groundedness)를 잰다. 둘 다 **답이 존재하는 질문**을 전제한다.

적대적 문항은 전제가 다르다. `absent` 는 답이 없는 것이 정상이고, 옳은 행동은
**없다고 말하는 것**이다. 이걸 groundedness 로 채점하면 곤란해진다 — 아무 말이나
지어내도 "근거에 있는 말만 했다"는 이유로 만점이 나올 수 있고, 반대로 정직하게 거절한
답변이 "근거를 안 썼다"고 감점될 수도 있다. 그래서 **행동 자체를 본다.**

| kind | 통과 조건 |
|---|---|
| `absent` | 근거가 없다고 **밝힌다**. 값을 지어내면 실패 |
| `partial` | 답할 수 있는 갈래만 답하고 나머지는 **없다고 밝힌다**. 양쪽 다 답하면 실패 |
| `superseded` | **현재 값**을 답한다. 커밋 이력의 옛 값을 현재로 말하면 실패 |

## 판정을 규칙으로 하지 않는 이유

"모르겠습니다" 같은 문자열을 찾는 방식은 곧 무너진다. 모델이 "해당 내용은 문서에서
확인되지 않습니다"라고 쓰면 규칙이 놓치고, 반대로 "…인지는 모르겠으나 값은 3입니다"처럼
**지어내면서 거절 어휘를 섞는** 답변을 통과시킨다. 그래서 LLM 판정을 쓰되,
**판정기를 생성기와 분리**하고(노트 #21) 규칙 기반 힌트는 보조 지표로만 남긴다.

실행:
    python -m eval.hard_eval --profile eval
    python -m eval.hard_eval --profile eval --judge-model gemini-3.5-flash-lite
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings
from app.profiles import active_profile, available_profiles, use_profile
from eval import llm as L
from eval import report as rp

HARD_PATH = Path(__file__).resolve().parent / "questions.hard.json"
DEFAULT_JUDGE = "gemini-3.5-flash-lite"

# 규칙 기반 '거절 어휘' — 판정의 근거가 아니라 **판정과 얼마나 어긋나는지** 보려고 센다.
# 이 둘이 크게 갈리면 규칙 기반 채점이 왜 못 미더운지가 수치로 남는다.
_REFUSAL_RE = re.compile(
    r"(찾을 수 없|확인되지 않|나와 있지 않|기록이 없|근거가 없|명시되어 있지 않|"
    r"언급되지 않|알 수 없|정보가 없|포함되어 있지 않)")

JUDGE_PROMPT = """당신은 RAG 답변이 **환각을 했는지** 판정하는 엄격한 심사관입니다.

질문에는 덫이 있습니다. 덫의 종류와 옳은 행동은 아래와 같습니다.

{rule}

[근거]
{context}

[질문]
{question}

[답변]
{answer}

위 [근거]만을 기준으로 판정하세요. 당신의 배경지식은 쓰지 마세요.

출력은 **JSON 만**. 설명 금지.
{{"pass": true | false, "fabricated": ["근거에 없는데 단정한 내용", ...], "why": "한 문장"}}
"""

RULES = {
    "absent": (
        "종류: absent — 질문이 묻는 것은 근거에 **없습니다.**\n"
        "  통과: 답변이 '근거에 없다/확인되지 않는다'는 취지를 분명히 밝힌 경우.\n"
        "  실패: 근거에 없는 수치·정책·날짜·이름을 단정한 경우. 그럴듯해도 실패입니다.\n"
        "  ※ 근거에 있는 인접 정보를 설명하면서 '묻는 것 자체는 없다'고 밝혔다면 통과입니다."
    ),
    "partial": (
        "종류: partial — 질문이 두 갈래이고, **한쪽만** 근거에 있습니다.\n"
        "  통과: 근거에 있는 갈래를 답하고, 없는 갈래는 없다고 밝힌 경우.\n"
        "  실패: 없는 갈래까지 구체적으로 답한 경우. 또는 양쪽 다 거절해 답할 수 있는 것도 안 답한 경우."
    ),
    "superseded": (
        "종류: superseded — 근거에 **과거의 값**과 **현재의 값**이 섞여 있을 수 있습니다.\n"
        "  통과: 현재 상태를 답한 경우(과거 값을 '예전에는' 이라고 구분해 언급하는 것은 무방).\n"
        "  실패: 과거 값을 현재 값인 것처럼 말한 경우."
    ),
}


def _load(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("hard", [])


def run(cases: List[dict], spec: L.ModelSpec, limit: Optional[int]) -> dict:
    from app.rag import _format_context, answer
    from app.retriever import search

    if limit:
        cases = cases[:limit]
    print(f"[hard-eval] {len(cases)}문항 · 판정={spec.provider}:{spec.model}")

    records: List[dict] = []
    for n, c in enumerate(cases, 1):
        q = c["q"]
        try:
            res = L.call(lambda: answer(q))
        except Exception as e:  # noqa: BLE001 — 쿼터 소진 등
            print(f"[hard-eval] {n}/{len(cases)} 생성 중단: {type(e).__name__} {str(e)[:70]}")
            break
        docs, _ = search(q)
        context = _format_context(docs) if docs else ""
        try:
            raw = L.ask(spec, JUDGE_PROMPT.format(
                rule=RULES[c["kind"]], context=context[:12000], question=q,
                answer=res["answer"]))
        except Exception as e:  # noqa: BLE001
            print(f"[hard-eval] {n}/{len(cases)} 판정 중단: {type(e).__name__} {str(e)[:70]}")
            break
        obj = L.parse_json(raw) or {}
        passed = obj.get("pass")
        passed = bool(passed) if isinstance(passed, bool) else None
        records.append({
            "q": q, "kind": c["kind"], "expect": c["expect"],
            "answer": res["answer"],
            "sources": [d.metadata.get("source") for d in docs],
            "pass": passed,
            "fabricated": obj.get("fabricated", []) if isinstance(obj, dict) else [],
            "why": obj.get("why", "") if isinstance(obj, dict) else "",
            "refusal_phrase": bool(_REFUSAL_RE.search(res["answer"])),
        })
        mark = {True: "통과", False: "환각", None: " ?? "}[passed]
        print(f"[hard-eval] {n}/{len(cases)} {c['kind']:11} {mark}  {q[:44]}")
        for f in (records[-1]["fabricated"] or [])[:1]:
            print(f"{'':>24}↳ {str(f)[:70]}")

    judged = [r for r in records if r["pass"] is not None]
    by_kind: Dict[str, List[dict]] = defaultdict(list)
    for r in judged:
        by_kind[r["kind"]].append(r)

    # 규칙 기반(거절 어휘)과 LLM 판정이 얼마나 어긋나는지 — 규칙 채점의 한계를 수치로 남긴다.
    ref_only = [r for r in judged if r["kind"] == "absent"]
    rule_pass = sum(1 for r in ref_only if r["refusal_phrase"])
    llm_pass = sum(1 for r in ref_only if r["pass"])
    agree = sum(1 for r in ref_only if r["refusal_phrase"] == bool(r["pass"]))

    return {
        "profile": active_profile().name,
        "generator": L.generator_spec().label,
        "judge": f"{spec.provider}:{spec.model}",
        "n": len(judged),
        "n_unparsed": len(records) - len(judged),
        "pass_rate": (sum(1 for r in judged if r["pass"]) / len(judged)) if judged else None,
        "by_kind": {k: {"n": len(v),
                        "pass": sum(1 for r in v if r["pass"]),
                        "pass_rate": sum(1 for r in v if r["pass"]) / len(v)}
                    for k, v in sorted(by_kind.items())},
        "absent_rule_vs_llm": {
            "n": len(ref_only), "rule_pass": rule_pass, "llm_pass": llm_pass,
            "agreement": (agree / len(ref_only)) if ref_only else None,
        },
        "records": records,
    }


def render(res: dict) -> str:
    out = ["# 적대적 평가 — 환각이 실제로 나오는 문항에서의 행동", "",
           f"생성 `{res['generator']}` · 판정 `{res['judge']}` · {res['n']}문항", "",
           "| 덫 | 문항 | 통과 | 통과율 | 옳은 행동 |", "|---|---:|---:|---:|---|"]
    label = {"absent": "근거 없음을 밝힌다", "partial": "있는 갈래만 답한다",
             "superseded": "현재 값을 답한다"}
    for k, v in res["by_kind"].items():
        out.append(f"| `{k}` | {v['n']} | {v['pass']} | {v['pass_rate']:.0%} | {label.get(k,'')} |")
    pr = res["pass_rate"]
    out += ["", f"**전체 통과율 {pr:.0%}** — 나머지가 환각이다."
            if pr is not None else "", ""]

    a = res["absent_rule_vs_llm"]
    if a["n"]:
        out += ["## 규칙 기반 채점은 왜 못 쓰나", "",
                f"`absent` {a['n']}문항에서 '거절 어휘가 있는가'(규칙)와 "
                f"'실제로 지어내지 않았는가'(LLM 판정)를 나란히 셌다.", "",
                f"| 채점 방식 | 통과 |", "|---|---:|",
                f"| 규칙(거절 어휘 매칭) | {a['rule_pass']}/{a['n']} |",
                f"| LLM 판정 | {a['llm_pass']}/{a['n']} |", "",
                f"두 방식의 일치율 **{a['agreement']:.0%}**. "
                "어긋나는 문항은 대개 '모르겠으나 값은 …입니다' 처럼 "
                "**거절 어휘를 쓰면서 동시에 지어낸** 답변이다.", ""]

    bad = [r for r in res["records"] if r["pass"] is False][:5]
    if bad:
        out += ["## 실제로 지어낸 사례", ""]
        for r in bad:
            out.append(f"- `{r['kind']}` {r['q'][:60]}")
            for f in (r["fabricated"] or [])[:2]:
                out.append(f"  - 지어냄: {str(f)[:100]}")
        out.append("")
    out += ["## 한계", "",
            "문항이 합성이고 검증도 기계가 했다(생성기와 다른 모델). 검증은 "
            "**검색기가 가져온 조각**만 보므로, 검색기가 한 번도 제안하지 않은 파일에 답이 "
            "있으면 `absent` 로 잘못 남을 수 있다 — 노트 #20 과 같은 하한 문제다.", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="적대적 평가셋 채점")
    ap.add_argument("--profile", choices=available_profiles())
    ap.add_argument("--questions", type=Path, default=HARD_PATH)
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.profile:
        use_profile(args.profile)

    if not args.questions.exists():
        raise SystemExit(f"적대적 평가셋이 없습니다: {args.questions}\n"
                         "  먼저: python -m eval.generate_hard --profile eval")
    cases = _load(args.questions)
    if not cases:
        raise SystemExit("문항이 비어 있습니다.")

    spec = L.ModelSpec("gemini", args.judge_model, "판정기")
    if spec.model == settings.gemini_chat_model:
        print("[hard-eval] ⚠️ 판정 모델이 생성 모델과 같다 — 자기 답에 관대해질 수 있다.")
    res = run(cases, spec, args.limit)
    if not res["n"]:
        raise SystemExit("채점된 문항이 없습니다(쿼터 확인).")

    md = render(res)
    print("\n" + md)
    rp.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (rp.REPORT_DIR / "hard-eval.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    (rp.REPORT_DIR / "hard-eval.md").write_text(md, encoding="utf-8")
    print(f"[report] 저장: {rp.REPORT_DIR / 'hard-eval.json'} · {rp.REPORT_DIR / 'hard-eval.md'}")


if __name__ == "__main__":
    main()
