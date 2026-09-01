"""RAGAS 로 같은 답변을 다시 채점 — 자체 판정기와 대조하기 위한 스크립트.

## 왜 하는가

`eval/faithfulness.py` 는 사실상 **RAGAS 의 faithfulness 지표를 손으로 재구현한 것**이다.
그러면 자연히 물어야 할 질문이 생긴다 — **표준 도구로 재도 같은 값이 나오는가?**

- 값이 비슷하면 자체 구현이 타당하다는 근거가 된다.
- 값이 갈리면 **어느 쪽이 무엇을 다르게 보는지**가 드러난다. 그게 더 값진 결과다.

## 왜 별도 환경에서 도는가

`ragas` 는 `langchain-core`·`langgraph` 버전을 바꿔 설치한다. 평가 도구가 **서비스 런타임을
오염시키면** 안 되므로 `.venv-ragas` 에 격리했다. 그래서 이 파일은 `app.*` 를 임포트하지 않고,
주 환경이 떨궈 둔 JSON(질문·근거·답변)만 읽는다.

    # 주 환경: 답변과 자체 점수를 만든다
    python -m eval.faithfulness --profile eval --label self

    # 격리 환경: 같은 답변을 RAGAS 로 다시 채점
    .venv-ragas/Scripts/python eval/ragas_score.py --in eval/reports/faithfulness-self.json

    # 주 환경: 두 점수를 대조
    python -m eval.compare_judges

⚠️ 무료 티어(분당 15요청)에서 RAGAS 는 샘플당 여러 번 호출한다 → `--n` 으로 표본을 줄일 것.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    """.env 에서 GOOGLE_API_KEY 만 읽는다(격리 환경에 dotenv 를 깔지 않으려고 직접 파싱)."""
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="RAGAS 로 faithfulness 재채점")
    ap.add_argument("--in", dest="src", type=Path,
                    default=REPO_ROOT / "eval/reports/faithfulness-self.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n", type=int, default=0, help="앞 N개만(0=전체). 무료 티어에선 줄일 것")
    ap.add_argument("--model", default=None, help="판정 모델(기본: .env 의 GEMINI_CHAT_MODEL)")
    args = ap.parse_args()

    load_env()
    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("GOOGLE_API_KEY 가 없습니다(.env 확인).")

    data = json.loads(args.src.read_text(encoding="utf-8"))
    records = data["records"]
    # 근거가 없는 항목(범위 밖 질문)은 faithfulness 정의상 채점 대상이 아니다.
    records = [r for r in records if r.get("contexts")]
    if args.n:
        records = records[: args.n]
    print(f"[ragas] 입력 {args.src.name} · 채점 대상 {len(records)}문항")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness
    from ragas.run_config import RunConfig

    model = args.model or os.environ.get("GEMINI_CHAT_MODEL", "gemini-3.1-flash-lite")
    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=model, temperature=0))
    # 무료 티어 분당 15요청 → 동시 실행 1, 재시도 넉넉히.
    metric = Faithfulness(llm=llm)
    metric.init(RunConfig(max_workers=1, max_retries=6, timeout=180))

    scores = []
    for i, r in enumerate(records, 1):
        sample = SingleTurnSample(
            user_input=r["q"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
        )
        try:
            s = metric.single_turn_score(sample)
        except Exception as e:  # noqa: BLE001
            print(f"[ragas] {i}/{len(records)} 실패: {type(e).__name__} {str(e)[:100]}")
            s = None
        scores.append({"q": r["q"], "ragas_faithfulness": s, "ours": r.get("score")})
        print(f"[ragas] {i}/{len(records)}  ragas={s if s is None else round(s, 3)}"
              f"  ours={r.get('score')}  {r['q'][:40]}")

    ok = [x["ragas_faithfulness"] for x in scores if x["ragas_faithfulness"] is not None]
    mean = sum(ok) / len(ok) if ok else None
    out = args.out or REPO_ROOT / "eval/reports/ragas-faithfulness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": args.src.name,
        "judge_model": model,
        "metric": "ragas.Faithfulness",
        "mean": mean,
        "n": len(ok),
        "failed": len(scores) - len(ok),
        "scores": scores,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ragas] 평균 {mean if mean is None else round(mean, 3)} (n={len(ok)}) → {out}")


if __name__ == "__main__":
    main()
