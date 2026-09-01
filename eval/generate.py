"""합성 평가셋 생성 — 원본 청크에서 LLM 이 질문을 만든다.

## 왜 필요했나

수기 평가셋 40문항이 모든 주장의 단일 실패점이었다. 문항을 사람이 냈다는 건
**시스템이 잘 답하는 쪽으로 질문이 기울었을 수 있다**는 뜻이고, 그 위에서 나온 93% 는
"내가 낸 문제를 내가 채점한 점수"다. 문항을 수백 개로 늘려야 그 편향이 희석된다.

수기로 500문항을 만드는 건 현실적이지 않다(평가셋 제작이 개발 시간을 크게 잡아먹는다는 건
RAGAS·DeepEval 같은 도구가 존재하는 이유이기도 하다). 그래서 생성을 자동화한다.

## 순환을 어떻게 피했나 — 이 파일의 핵심 설계

질문은 **원본 청크**에서 만들고, 정답 라벨은 그 청크의 `source` 를 그대로 쓴다.
**검색기를 한 번도 부르지 않는다.** 검색 결과로 라벨을 만들면 "검색기가 찾은 것이 정답"이 되어
점수가 자기 자신을 증명하게 된다.

    청크 → (LLM) → 질문 → 라벨 = 그 청크의 source     ✅ 검색기 무관
    질문 → (검색기) → 결과 → 라벨                      ❌ 순환

## 남는 편향 — 정직하게 적어 둘 것

1. **생성 모델 = 답변 모델**이면 자기가 잘 답할 질문을 낼 수 있다. → Day 5 cross-judge 로 측정.
2. **어휘 중복** — 청크 표현을 그대로 베낀 질문은 BM25 가 공짜로 맞힌다. 프롬프트로 억제하고,
   실제 중복률(`lex_overlap`)을 문항마다 기록해 사후에 걸러낼 수 있게 한다.
3. **거짓 오답(false negative)** — 같은 답이 다른 파일에도 있으면 정답을 찾고도 miss 로 잡힌다.
   → 수동 검수 서브셋의 **불일치율**로 크기를 잰다(`eval/verify.py`).

실행:
    python -m eval.generate --profile eval --per-source 4 --per-chunk 2
    python -m eval.generate --profile eval --limit 20 --dry-run   # 프롬프트·비용 감 잡기
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.profiles import active_profile, available_profiles, use_profile
from eval import llm as L
from eval.datasets import ORIGIN_SYNTHETIC

OUT_PATH = Path(__file__).resolve().parent / "questions.synthetic.json"

# 청크가 너무 짧으면 질문거리가 안 되고, 너무 길면 여러 주제가 섞여 라벨이 흐려진다.
MIN_CHARS = 200
MAX_CHARS = 4000

PROMPT = """당신은 사내 코드베이스 어시스턴트의 **검색 평가셋**을 만드는 엔지니어입니다.

아래는 어떤 저장소에서 잘라낸 조각 하나입니다. 이 조각을 근거로 답할 수 있는
**자연스러운 질문 {n}개**를 만드세요.

지켜야 할 규칙:
1. 실제 동료가 물을 법한 말투로 쓰세요. ("~는 왜 그래?", "~는 어디에 있어?", "~는 어떻게 동작해?")
2. **조각의 문장을 그대로 베끼지 마세요.** 특징적인 표현을 복사하면 단어 매칭만으로 맞힐 수 있어
   평가가 무의미해집니다. 같은 내용을 **다른 말로** 물으세요.
3. 이 조각만 보고 답할 수 있는 질문이어야 합니다. 조각 밖 지식이 필요한 질문은 만들지 마세요.
4. 메타 질문 금지: "이 문서의 제목은?", "이 코드는 몇 줄이야?" 같은 것.
5. 파일명이나 함수명을 질문에 그대로 넣는 것은 **한 문항까지만** 허용합니다.
6. 한국어로 쓰세요.

출력은 **JSON 배열만**. 설명 금지.
["질문1", "질문2"]

[조각 위치] {where}
[조각]
{chunk}
"""


def _tokens(text: str) -> set:
    from app.retriever import _tokenize

    return {t for t in _tokenize(text) if len(t) > 1}


def lexical_overlap(question: str, chunk: str) -> float:
    """질문 토큰 중 청크에도 있는 비율.

    1.0 에 가까우면 질문이 조각을 베낀 것이라 BM25 가 거저 맞힌다 —
    그런 문항이 많으면 recall 이 높게 나와도 검색 성능의 증거가 되지 못한다.
    """
    q = _tokens(question)
    return len(q & _tokens(chunk)) / len(q) if q else 0.0


def sample_chunks(per_source: int, seed: int, limit: Optional[int] = None) -> List[Tuple[int, object]]:
    """소스 파일마다 최대 `per_source` 개씩 뽑는다(한 파일이 평가셋을 독점하지 않게).

    검색기는 부르지 않는다 — 인덱스에서 청크 본문과 메타데이터만 읽는다.
    """
    from app.retriever import _corpus

    docs, _, _ = _corpus()
    by_source: Dict[str, List[Tuple[int, object]]] = defaultdict(list)
    for i, d in enumerate(docs):
        text = d.page_content
        if not (MIN_CHARS <= len(text) <= MAX_CHARS):
            continue
        by_source[d.metadata.get("source", "?")].append((i, d))

    rng = random.Random(seed)
    picked: List[Tuple[int, object]] = []
    for src in sorted(by_source):
        items = by_source[src]
        rng.shuffle(items)
        picked.extend(items[:per_source])
    rng.shuffle(picked)
    return picked[:limit] if limit else picked


def _bucket(doc_type: str) -> str:
    return {"code": "in_scope_code", "commit": "in_scope_commit"}.get(doc_type, "in_scope")


def generate(per_source: int, per_chunk: int, seed: int, limit: Optional[int], dry_run: bool) -> dict:
    prof = active_profile()
    spec = L.generator_spec()
    picked = sample_chunks(per_source, seed, limit)

    print(f"[gen] 프로필={prof.name} · 청크 {len(picked)}개 × 질문 {per_chunk}개 "
          f"→ 최대 {len(picked) * per_chunk}문항 (모델={spec.label})")
    if dry_run:
        i, d = picked[0]
        print("\n--- 프롬프트 예시 ---\n")
        print(PROMPT.format(n=per_chunk, where=d.metadata.get("source"), chunk=d.page_content[:800]))
        est_min = len(picked) * L.DEFAULT_THROTTLE_SEC / 60
        print(f"\n예상 소요: 약 {est_min:.0f}분 (호출 {len(picked)}회 × {L.DEFAULT_THROTTLE_SEC}s)")
        return {}

    out: Dict[str, List[dict]] = defaultdict(list)
    seen: set = set()
    failed = 0

    for n, (idx, d) in enumerate(picked, 1):
        src = d.metadata.get("source", "?")
        sec = d.metadata.get("section", "")
        where = f"{src}" + (f" > {sec}" if sec else "")
        raw = L.ask(spec, PROMPT.format(n=per_chunk, where=where, chunk=d.page_content[:MAX_CHARS]))
        parsed = L.parse_json(raw)
        if not isinstance(parsed, list):
            failed += 1
            print(f"[gen] {n}/{len(picked)} {src} — 파싱 실패")
            continue

        added = 0
        for q in parsed:
            if not isinstance(q, str) or len(q.strip()) < 6:
                continue
            key = "".join(q.split())
            if key in seen:
                continue
            seen.add(key)
            out[_bucket(d.metadata.get("doc_type", "doc"))].append({
                "q": q.strip(),
                "expected": [src],
                "origin": ORIGIN_SYNTHETIC,
                "axis": d.metadata.get("doc_type", "doc"),
                "chunk_index": idx,
                "lex_overlap": round(lexical_overlap(q, d.page_content), 3),
            })
            added += 1
        print(f"[gen] {n}/{len(picked)} {src} — {added}문항")

    total = sum(len(v) for v in out.values())
    print(f"\n[gen] 생성 {total}문항 (파싱 실패 {failed}건 / 중복 제거 후)")
    return {
        "_comment": "LLM 이 원본 청크에서 생성한 합성 평가셋. 라벨=청크의 source(검색기 미사용).",
        "_generated_by": spec.label,
        "_corpus_profile": prof.name,
        "_corpus_ref": prof.git_ref,
        "_params": {"per_source": per_source, "per_chunk": per_chunk, "seed": seed},
        **out,
        "out_of_scope": [],
    }


def summarize(data: dict) -> None:
    """생성된 셋의 품질 지표 — 숫자가 없으면 '많이 만들었다'밖에 말할 게 없다."""
    cases = [c for k in ("in_scope", "in_scope_code", "in_scope_commit") for c in data.get(k, [])]
    if not cases:
        return
    ov = sorted(c["lex_overlap"] for c in cases)
    med = ov[len(ov) // 2]
    high = sum(1 for x in ov if x >= 0.8)
    per_axis: Dict[str, int] = defaultdict(int)
    for c in cases:
        per_axis[c["axis"]] += 1
    srcs = {tuple(c["expected"])[0] for c in cases}
    print(f"[gen] 축별: {dict(sorted(per_axis.items()))} · 정답 소스 {len(srcs)}개")
    print(f"[gen] 어휘 중복률 중앙값 {med:.2f} · 0.8 이상 {high}문항({high/len(cases):.0%})")
    if high / len(cases) > 0.3:
        print("      ⚠️ 베껴 쓴 질문 비중이 높다 — BM25 가 거저 맞히므로 recall 이 부풀 수 있다.")


def main() -> None:
    ap = argparse.ArgumentParser(description="합성 평가셋 생성(원본 청크 → LLM 질문)")
    ap.add_argument("--profile", choices=available_profiles(),
                    help="코퍼스 프로필(기본: .env). 게이트와 맞추려면 eval 을 쓸 것")
    ap.add_argument("--per-source", type=int, default=4, help="소스 파일당 뽑을 청크 수")
    ap.add_argument("--per-chunk", type=int, default=2, help="청크당 만들 질문 수")
    ap.add_argument("--seed", type=int, default=20260901, help="샘플링 시드(재현용)")
    ap.add_argument("--limit", type=int, default=None, help="청크 수 상한(맛보기용)")
    ap.add_argument("--dry-run", action="store_true", help="프롬프트와 예상 시간만 출력")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()
    if args.profile:
        use_profile(args.profile)

    data = generate(args.per_source, args.per_chunk, args.seed, args.limit, args.dry_run)
    if not data:
        return
    summarize(data)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gen] 저장: {args.out}")


if __name__ == "__main__":
    main()
