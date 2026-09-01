"""답변 품질 평가 — LLM-as-judge 로 groundedness(근거 충실도) 채점.

검색 평가(run_eval.py)는 "맞는 문서를 찾았나"만 본다. 하지만 RAG 의 최종 실패는
**찾은 근거를 두고도 답변이 딴소리(환각)를 하는 것**이다. 그걸 잡는 지표가 groundedness:
답변의 각 주장이 제공된 근거에서 실제로 지지되는가.

judge 는 답변을 만든 것과 같은 Gemini(무료)를 쓴다. self-judge 편향을 줄이려 judge 에는
**답변 생성 맥락을 주지 않고**, 오직 (근거, 답변)만 주고 "근거에 없는 주장을 찾아라"라는
반대 방향 과제를 준다. 이렇게 하면 생성기가 놓친 환각을 판별기가 잡아낼 여지가 생긴다.

실행:
    python -m eval.faithfulness            # 문서+코드 질문 전체
    python -m eval.faithfulness --n 6      # 앞 6개만(빠른 점검)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from app.config import settings
from app.profiles import available_profiles, use_profile
from app.rag import _format_context, _llm, _text_of, answer
from app.retriever import search
from eval.datasets import load_questions
from eval import llm as L
from eval import report as rp



JUDGE_PROMPT = (
    "당신은 RAG 답변의 사실 충실도(groundedness)를 검사하는 엄격한 심사관입니다.\n"
    "아래 [근거] 와 [답변] 만 보고, 답변의 각 주장이 근거에서 지지되는지 판정하세요.\n"
    "근거에 없는 내용을 답변이 단정하면 '환각'입니다. 일반 상식이라도 근거에 없으면 감점하세요.\n"
    "단, '문서에서 찾을 수 없다'는 취지의 답변은 근거가 비어도 충실한 것으로 봅니다.\n"
    "프로젝트/시스템 명칭, 근거에 표기된 파일명·섹션 인용은 감점 대상이 아닙니다.\n\n"
    "다음 JSON 만 출력하세요(설명 금지):\n"
    '{{"score": 0.0~1.0, "unsupported": ["근거 없는 주장", ...]}}\n'
    "score 는 지지되는 주장의 비율입니다. unsupported 가 없으면 빈 배열.\n\n"
    "[근거]\n{context}\n\n[답변]\n{answer}"
)


# 무료 티어는 분당 15요청. 문항당 answer+judge=2요청이라 선제 throttle + 429 재시도.
_THROTTLE_SEC = 4.5


def _call(fn: Callable[[], Any]) -> Any:
    """LLM 호출을 429(RESOURCE_EXHAUSTED) 재시도로 감싼다."""
    for _ in range(5):
        time.sleep(_THROTTLE_SEC)
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                m = re.search(r"retry in ([\d.]+)", msg) or re.search(r"retryDelay'?: ?'?(\d+)", msg)
                wait = float(m.group(1)) if m else 20.0
                print(f"    (429 — {wait:.0f}s 대기 후 재시도)")
                time.sleep(wait + 1)
                continue
            raise
    return fn()


def _judge(question: str, context: str, ans: str, spec=None) -> Dict[str, Any]:
    """판정 모델은 **주입받는다** - 생성기와 다른 모델로 갈아끼울 수 있어야 하기 때문이다.

    같은 모델이 답을 만들고 채점하면 자기 답에 후해진다(self-enhancement bias).
    표준 완화책이 모델 분리이고, 그 차이를 수치화하는 것이 이 인자의 존재 이유다.
    """
    spec = spec or L.generator_spec()
    if settings.active_llm == "extractive":
        return {"score": None, "unsupported": [], "note": "LLM 미연결(extractive)"}
    raw = L.ask(spec, JUDGE_PROMPT.format(context=context[:12000], answer=ans))
    txt = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        obj = json.loads(txt)
        return {"score": float(obj.get("score", 0.0)), "unsupported": obj.get("unsupported", [])}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"score": None, "unsupported": [], "note": f"판정 파싱 실패: {raw[:80]}"}


def main() -> None:
    ap = argparse.ArgumentParser(description="답변 groundedness 평가")
    ap.add_argument("--n", type=int, default=0, help="앞 N개만(0=전체)")
    ap.add_argument("--questions", type=Path, default=None, help="평가셋 파일 경로(기본: 프로필 것)")
    ap.add_argument("--judge-provider", default=None, choices=["gemini", "openai"],
                    help="판정 모델 provider(기본: 생성기와 동일 = self-judge)")
    ap.add_argument("--judge-model", default=None, help="판정 모델 이름")
    ap.add_argument("--label", default="", help="리포트 파일명 꼬리표(예: self / cross)")
    ap.add_argument(
        "--profile",
        choices=available_profiles(),
        help="코퍼스 프로필(기본: .env 의 CORPUS_PROFILE)",
    )
    args = ap.parse_args()
    if args.profile:
        use_profile(args.profile)

    qs = load_questions(args.questions)
    print(f"[eval] {qs.summary()}")
    cases: List[dict] = qs.in_scope + qs.in_scope_code
    if args.n:
        cases = cases[: args.n]

    # 범위 밖 질문 하나를 섞어 '거절도 충실한 답'으로 채점되는지 확인
    oos = qs.out_of_scope[0]

    gen_spec = L.generator_spec()
    if args.judge_provider:
        judge_spec = L.ModelSpec(args.judge_provider,
                                 args.judge_model or settings.gemini_chat_model, "판정기")
    elif args.judge_model:
        judge_spec = L.ModelSpec(gen_spec.provider, args.judge_model, "판정기")
    else:
        judge_spec = L.ModelSpec(gen_spec.provider, gen_spec.model, "판정기(self)")
    self_judge = (judge_spec.provider, judge_spec.model) == (gen_spec.provider, gen_spec.model)

    print(f"\n= 답변 groundedness - {len(cases)}문항 + 범위밖 1")
    print(f"   생성 {gen_spec.provider}:{gen_spec.model} · 판정 {judge_spec.provider}:{judge_spec.model}"
          f"  -> {'self-judge(편향 있음)' if self_judge else 'cross-judge'}\n")
    print(f"{'score':>6}  {'환각':>4}  질문")
    print("-" * 70)

    scored: List[float] = []
    records: List[dict] = []
    for c in cases + [{"q": oos, "expected": []}]:
        q = c["q"]
        result = _call(lambda: answer(q))
        docs, _ = search(q)
        # 생성기가 본 것과 동일한 근거(파일명·섹션 헤더 포함)를 judge 에도 준다.
        context = _format_context(docs) if docs else ""
        verdict = _judge(q, context, result["answer"], judge_spec)
        s = verdict.get("score")
        records.append({
            "q": q, "expected": c.get("expected", []),
            "answer": result["answer"],
            "contexts": [d.page_content for d in docs],
            "sources": [d.metadata.get("source") for d in docs],
            "score": s, "unsupported": verdict.get("unsupported", []),
        })
        if s is not None:
            scored.append(s)
        n_unsup = len(verdict.get("unsupported", []))
        flag = verdict.get("note", "")
        print(f"{(f'{s:.2f}' if s is not None else '  -- '):>6}  {n_unsup:>4}  {q[:48]}  {flag}")
        for u in verdict.get("unsupported", [])[:2]:
            print(f"{'':>14}↳ {u[:70]}")

    avg = sum(scored) / len(scored) if scored else None
    if avg is not None:
        print(f"\n평균 groundedness = {avg:.3f}  (n={len(scored)}, 1.0=환각 0)")
        if self_judge:
            print("  [주의] self-judge - 생성·판정이 같은 모델이라 자기 답에 후할 수 있다(절대값으로 쓰지 말 것).")

    stem = f"faithfulness-{args.label}" if args.label else "faithfulness"
    out = rp.REPORT_DIR / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generator": f"{gen_spec.provider}:{gen_spec.model}",
        "judge": f"{judge_spec.provider}:{judge_spec.model}",
        "self_judge": self_judge,
        "profile": qs.profile,
        "questions": qs.path.name,
        "mean_score": avg,
        "n": len(scored),
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] 저장: {out}   (RAGAS 대조·cross-judge 비교의 입력)")
    print()


if __name__ == "__main__":
    main()
