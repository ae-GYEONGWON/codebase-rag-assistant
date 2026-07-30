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
from app.rag import _format_context, _llm, _text_of, answer
from app.retriever import search

# 실제 평가셋(questions.json)은 인덱싱한 코드베이스에 종속이라 git 에 넣지 않는다.
_QDIR = Path(__file__).parent
QUESTIONS = _QDIR / "questions.json"
if not QUESTIONS.exists():
    QUESTIONS = _QDIR / "questions.example.json"

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


def _judge(question: str, context: str, ans: str) -> Dict[str, Any]:
    if settings.active_llm == "extractive":
        return {"score": None, "unsupported": [], "note": "LLM 미연결(extractive)"}
    raw = _text_of(
        _call(lambda: _llm().invoke(JUDGE_PROMPT.format(context=context[:12000], answer=ans))).content
    )
    txt = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        obj = json.loads(txt)
        return {"score": float(obj.get("score", 0.0)), "unsupported": obj.get("unsupported", [])}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"score": None, "unsupported": [], "note": f"판정 파싱 실패: {raw[:80]}"}


def main() -> None:
    ap = argparse.ArgumentParser(description="답변 groundedness 평가")
    ap.add_argument("--n", type=int, default=0, help="앞 N개만(0=전체)")
    args = ap.parse_args()

    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    cases: List[dict] = data["in_scope"] + data.get("in_scope_code", [])
    if args.n:
        cases = cases[: args.n]

    # 범위 밖 질문 하나를 섞어 '거절도 충실한 답'으로 채점되는지 확인
    oos = data["out_of_scope"][0]

    print(f"\n■ 답변 groundedness — {len(cases)}문항 + 범위밖 1 (judge={settings.active_llm})\n")
    print(f"{'score':>6}  {'환각':>4}  질문")
    print("-" * 70)

    scored: List[float] = []
    for c in cases + [{"q": oos, "expected": []}]:
        q = c["q"]
        result = _call(lambda: answer(q))
        docs, _ = search(q)
        # 생성기가 본 것과 동일한 근거(파일명·섹션 헤더 포함)를 judge 에도 준다.
        context = _format_context(docs) if docs else ""
        verdict = _judge(q, context, result["answer"])
        s = verdict.get("score")
        if s is not None:
            scored.append(s)
        n_unsup = len(verdict.get("unsupported", []))
        flag = verdict.get("note", "")
        print(f"{(f'{s:.2f}' if s is not None else '  -- '):>6}  {n_unsup:>4}  {q[:48]}  {flag}")
        for u in verdict.get("unsupported", [])[:2]:
            print(f"{'':>14}↳ {u[:70]}")

    if scored:
        avg = sum(scored) / len(scored)
        print(f"\n평균 groundedness = {avg:.3f}  (n={len(scored)}, 1.0=환각 0)")
    print()


if __name__ == "__main__":
    main()
