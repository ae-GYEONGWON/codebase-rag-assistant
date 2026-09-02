"""거짓 오답 감사 — miss 로 채점된 문항이 **정말로 틀린 것인지** 독립 판정기로 다시 본다.

## 왜 필요했나

합성 평가셋의 정답 라벨은 "질문을 만들어 낸 청크의 파일" 하나뿐이다. 그런데 같은 내용이
여러 파일에 있으면(테스트와 구현, 문서와 노트) **검색기가 옳은 파일을 가져오고도 miss** 가 된다.
이걸 세지 않으면 recall 이 실제보다 낮게 나오고, 그 낮은 값을 보고 검색을 고치려 들면
없는 병을 치료하게 된다.

사람이 검수하면 좋지만, 검수자가 **코퍼스를 알아야** 판정할 수 있다. 저장소 전체를 아는 일은
사람보다 기계가 낫다. 그래서 역할을 나눈다:

    사람  — "이 조각이 이 질문에 답하나?"        (화면의 조각만 읽으면 됨)
    기계  — "다른 파일에도 답이 있나?"           (전수 확인이 필요 — 이 파일)

## 순환에 대한 정직한 진술

이 감사는 **검색기가 가져온 것**을 후보로 삼는다. 즉 검색기가 한 번도 제안하지 않은 파일은
후보에 오르지 않으므로, 여기서 나오는 거짓 오답률은 **하한**이다.

그리고 이 절차는 구조상 recall 을 **올리는 방향으로만** 작동한다. 그래서 두 가지를 지킨다.

1. **판정기를 생성기와 다른 모델로 쓴다.** 같은 모델이면 자기가 만든 질문에 관대해진다.
2. **판정 기준을 엄격하게 준다.** "관련 있음"이 아니라 "질문에 대한 답이 그 조각 안에 실제로
   들어 있음"만 인정한다.

리포트에는 **원래 recall 과 보정 recall 을 함께** 남긴다. 보정값만 인용하면 절차가 점수를
올리는 장치로 오해된다.

실행:
    python -m eval.audit_misses --profile eval --questions eval/questions.synthetic.json
    python -m eval.audit_misses --profile eval --questions ... --apply   # 라벨에 반영
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.profiles import active_profile, available_profiles, use_profile
from eval import llm as L
from eval import report as rp
from eval.datasets import load_questions

# 생성기(gemini-3.1-flash-lite)와 **다른** 모델. 완전한 독립은 아니지만(같은 계열),
# 같은 모델이 자기 생성물을 채점하는 것보다는 낫다. 이 한계는 리포트에 함께 적는다.
DEFAULT_AUDIT_MODEL = "gemini-3.5-flash"

PROMPT = """당신은 검색 평가의 **정답 라벨**이 옳은지 감사하는 엄격한 심사관입니다.

아래 질문에 대해, 제시된 조각들 중 **질문의 답이 실제로 그 안에 들어 있는 것**을 고르세요.

엄격히 판단하세요:
- "주제가 관련 있다", "같은 모듈이다" 정도로는 **선택하지 마세요.**
- 질문이 묻는 내용을 그 조각만 읽고 **답할 수 있어야** 선택합니다.
- 해당하는 것이 하나도 없으면 빈 배열을 반환하세요. 그게 정상적인 결과입니다.

출력은 **JSON 만**. 설명 금지.
{{"answers": [번호, ...]}}

[질문]
{question}

[조각들]
{blocks}
"""


def _blocks(docs) -> str:
    out = []
    for i, d in enumerate(docs, 1):
        out.append(f"--- [{i}] {d.metadata.get('source', '?')}\n{d.page_content[:1200]}")
    return "\n\n".join(out)


def audit(qs, k: int, spec: L.ModelSpec, limit: Optional[int],
          sample: Optional[int] = None, seed: int = 20260904) -> dict:
    from app.retriever import search

    cases: List[dict] = []
    for bucket in ("in_scope", "in_scope_code", "in_scope_commit"):
        cases.extend(getattr(qs, bucket, []) or [])

    # 1) 먼저 miss 를 모은다(LLM 호출 없음)
    misses: List[Tuple[dict, list]] = []
    hits = 0
    for c in cases:
        docs, _ = search(c["q"], k=k)
        got = [d.metadata.get("source", "?") for d in docs]
        if set(got) & set(c["expected"]):
            hits += 1
        elif docs:
            misses.append((c, docs))
    raw_recall = hits / len(cases) if cases else 0.0
    n_miss_total = len(misses)
    print(f"[audit] 전체 {len(cases)}문항 · 원래 recall {raw_recall:.1%} · 감사 대상 miss {n_miss_total}건")
    if sample and sample < len(misses):
        import random
        random.Random(seed).shuffle(misses)
        misses = misses[:sample]
        print(f"[audit] 무작위 표본 {sample}/{n_miss_total}건 (seed={seed}) — 무료 티어 일일 쿼터 때문")
    if limit:
        misses = misses[:limit]
        print(f"[audit] --limit {limit} 적용")

    # 2) miss 만 판정기에 올린다
    results: List[dict] = []
    false_neg = 0
    for n, (c, docs) in enumerate(misses, 1):
        try:
            raw = L.ask(spec, PROMPT.format(question=c["q"], blocks=_blocks(docs)))
        except Exception as e:  # noqa: BLE001 — 무료 티어 일일 쿼터 소진 등
            print(f"[audit] {n}/{len(misses)} 중단: {type(e).__name__} {str(e)[:90]}")
            print(f"[audit] 여기까지 {len(results)}건을 표본으로 집계한다(부분 결과 보존).")
            break
        parsed = L.parse_json(raw) or {}
        picks = [i for i in parsed.get("answers", []) if isinstance(i, int) and 1 <= i <= len(docs)]
        extra = sorted({docs[i - 1].metadata.get("source", "?") for i in picks})
        if extra:
            false_neg += 1
        results.append({
            "q": c["q"],
            "labeled": c["expected"],
            "retrieved": [d.metadata.get("source", "?") for d in docs],
            "also_answers": extra,
        })
        mark = "거짓오답" if extra else "진짜miss"
        print(f"[audit] {n}/{len(misses)} {mark:8} {c['q'][:44]}"
              + (f"  → {', '.join(extra)}" if extra else ""))

    n_cases = len(cases) or 1
    n_judged = len(results)
    # 표본만 판정했으므로 그 비율을 전체 miss 에 적용한다.
    # 판정한 것만 세면 나머지 miss 를 전부 '진짜 오답'으로 치는 셈이라 과소평가가 된다.
    fn_share_of_miss = (false_neg / n_judged) if n_judged else 0.0
    fn_rate = fn_share_of_miss * n_miss_total / n_cases
    return {
        "n_miss_total": n_miss_total,
        "n_judged": n_judged,
        "fn_share_of_miss": fn_share_of_miss,
        "profile": active_profile().name,
        "questions": qs.path.name,
        "audit_model": spec.model,
        "generator_model": settings.gemini_chat_model,
        "k": k,
        "n_cases": len(cases),
        "raw_recall": raw_recall,
        "n_miss": len(misses),
        "n_false_negative": false_neg,
        "false_negative_rate": fn_rate,
        "corrected_recall": raw_recall + fn_rate,
        "results": results,
    }


def apply_to_questions(path: Path, audit_result: dict) -> Path:
    """감사에서 확인된 추가 정답을 expected 에 합쳐 새 평가셋을 만든다."""
    data = json.loads(path.read_text(encoding="utf-8"))
    extra_by_q = {r["q"]: r["also_answers"] for r in audit_result["results"] if r["also_answers"]}
    n = 0
    for bucket in ("in_scope", "in_scope_code", "in_scope_commit"):
        for c in data.get(bucket, []):
            if c["q"] in extra_by_q:
                c["expected"] = sorted(set(c["expected"]) | set(extra_by_q[c["q"]]))
                c["label_audited"] = True
                n += 1
    data["_label_audit"] = {
        "audit_model": audit_result["audit_model"],
        "expanded": n,
        "false_negative_rate": round(audit_result["false_negative_rate"], 4),
    }
    out = path.with_name(path.stem + ".audited.json")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[audit] 정답 확장 {n}문항 → {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="거짓 오답 감사(독립 판정기)")
    ap.add_argument("--profile", choices=available_profiles())
    ap.add_argument("--questions", type=Path, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_AUDIT_MODEL, help="감사용 판정 모델(생성기와 달라야 함)")
    ap.add_argument("--limit", type=int, default=None, help="감사할 miss 수 상한")
    ap.add_argument("--sample", type=int, default=None,
                    help="miss 중 무작위 표본 N건만 판정(무료 티어 일일 쿼터 대응). "
                         "표본에서 나온 비율을 전체 miss 에 적용해 계산한다")
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--apply", action="store_true", help="확인된 추가 정답을 평가셋에 반영")
    args = ap.parse_args()
    if args.profile:
        use_profile(args.profile)

    qs = load_questions(args.questions)
    k = args.k or settings.retrieval_k
    spec = L.ModelSpec("gemini", args.model, "감사관")
    if spec.model == settings.gemini_chat_model:
        print("[audit] ⚠️ 감사 모델이 생성 모델과 같다 — 자기 생성물에 관대해질 수 있다.")
    print(f"[audit] {qs.summary()}\n[audit] 감사 모델={spec.model} (생성={settings.gemini_chat_model})")

    res = audit(qs, k, spec, args.limit, args.sample, args.seed)

    print(f"\n■ 거짓 오답 감사 — {res['n_cases']}문항\n")
    print(f"   원래 recall@{k}        {res['raw_recall']:.1%}")
    print(f"   판정한 miss           {res['n_judged']}/{res['n_miss_total']}건 (표본)")
    print(f"   그중 거짓 오답         {res['n_false_negative']}건 = miss 의 {res['fn_share_of_miss']:.1%}")
    print(f"   거짓 오답률(전체 적용)  {res['false_negative_rate']:.1%}")
    print(f"   보정 recall@{k}        {res['corrected_recall']:.1%}  (추정)")
    print("\n   ※ 감사는 검색기가 가져온 것만 후보로 보므로 이 값은 **하한**이다.")
    print("   ※ 절차상 recall 을 올리는 방향으로만 작동한다 — 원래 값과 함께 인용할 것.")

    out = rp.REPORT_DIR / f"label-audit-{Path(res['questions']).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[report] 저장: {out}")

    if args.apply and args.questions:
        apply_to_questions(args.questions, res)


if __name__ == "__main__":
    main()
