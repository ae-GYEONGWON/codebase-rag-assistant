"""적대적 평가셋 — **틀리기 쉬운 질문**을 의도적으로 만든다(노트 #21 의 숙제).

## 왜 필요한가

판정기 패널(`eval/judge_panel.py`)을 돌려 보니 판정기 4종이 전부 만점 근처를 줬고,
'어느 답이 환각인가'에 대한 합의는 우연 수준이었다(Cohen's κ ≈ -0.1). 원인은 판정기가
아니라 **평가셋이었다.** 답이 한 청크 안에 온전히 있는 질문만 모아 두면 어떤 모델도
환각할 이유가 없고, 그러면 groundedness 지표는 아무것도 변별하지 못한다.

지표를 살리려면 **환각이 실제로 발생하는 문항**이 있어야 한다. 이 파일이 그걸 만든다.

## 세 가지 덫

| kind | 무엇을 묻나 | 옳은 행동 | 틀리는 방식 |
|---|---|---|---|
| `absent` | 이 저장소에 있을 법하지만 **실제로는 없는** 사실 | 없다고 말한다 | 그럴듯하게 지어낸다 |
| `partial` | 절반만 근거가 있는 두 갈래 질문 | 있는 절반만 답하고 나머지는 없다고 밝힌다 | 나머지도 아는 척한다 |
| `superseded` | 과거에 바뀐 값의 **현재** 상태 | 지금 값을 답한다 | 커밋 이력의 옛 값을 현재로 말한다 |

`absent` 는 기존 `out_of_scope`(날씨·요리)와 전혀 다르다. 그쪽은 코사인 게이트가 검색
단계에서 자르지만, 이쪽은 **도메인 어휘를 그대로 쓰므로 게이트를 통과한다.** 검색은 되고
근거도 그럴듯한데 답이 그 근거에 없다 — 환각이 나오는 자리가 정확히 여기다.

## 생성물을 그대로 믿지 않는다

"근거가 없는 질문"을 LLM 에게 만들라고 하면 **실수로 근거가 있는 질문**을 낸다. 그러면
정답이 "없다고 말하기"인데 실제로는 있는 문항이 되어, 옳게 답한 시스템을 틀렸다고 채점한다.
그래서 생성 뒤에 **검증 단계**를 둔다.

    ① 생성기가 후보 질문을 만든다
    ② 그 질문을 실제 검색기에 걸어 상위 청크를 가져온다
    ③ **생성기와 다른 모델**이 "이 조각들로 답할 수 있나"를 판정한다
    ④ 답할 수 있다고 판정되면 `absent`/`partial` 후보에서 **버린다**

②가 검색기를 쓰는 것은 라벨 순환이 아니다 — 여기서 검색기는 '반례 후보 수집기'이고,
판정은 검색기와 무관한 모델이 한다. 다만 **검색기가 한 번도 제안하지 않은 파일은 후보에
오르지 않으므로**, 이 검증이 걸러내는 것은 상한이 아니라 하한이다(노트 #20 과 같은 한계).

실행:
    python -m eval.generate_hard --profile eval --limit 12            # 후보 생성 + 검증
    python -m eval.generate_hard --profile eval --limit 12 --dry-run  # 프롬프트·비용만
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.profiles import active_profile, available_profiles, use_profile
from eval import llm as L
from eval.datasets import ORIGIN_SYNTHETIC

OUT_PATH = Path(__file__).resolve().parent / "questions.hard.json"

MIN_CHARS = 300
MAX_CHARS = 4000

# 검증 판정기는 생성기와 달라야 한다. 같은 모델이면 자기가 만든 '없는 질문'을
# 없다고 편들어 준다. -lite 계열만 전 문항 완주가 가능하다(노트 #21 의 쿼터 실측).
DEFAULT_VERIFY_MODEL = "gemini-3.5-flash-lite"


ABSENT_PROMPT = """당신은 사내 코드베이스 어시스턴트를 **깨뜨리는** 평가 문항을 만드는 엔지니어입니다.

아래는 어떤 저장소에서 잘라낸 조각입니다. 이 조각과 **같은 주제·같은 어휘**를 쓰면서도,
**답이 이 저장소에 존재하지 않을** 질문 {n}개를 만드세요.

좋은 문항의 조건:
- 조각이 다루는 대상을 그대로 지목하되, 조각이 **말하지 않은 것**을 묻습니다.
  (예: 조각이 어떤 임계값을 쓴다고만 말하면 → "그 임계값을 정할 때 실험한 후보 범위는?")
- 구체적인 수치·날짜·담당자·정책·벤치마크 결과처럼 **있을 법하지만 기록되지 않은 것**이 좋습니다.
- 답을 아는 척하기 쉬운 질문일수록 좋습니다. 상식으로 그럴듯하게 지어낼 수 있어야 합니다.

금지:
- 저장소와 무관한 잡담(날씨·요리·연예). 그건 이미 다른 방식으로 걸러집니다.
- 조각을 읽으면 답할 수 있는 질문. 그건 평범한 평가 문항입니다.
- "이 문서의 제목은?" 같은 메타 질문.

한국어로 쓰세요. 출력은 **JSON 배열만**. 설명 금지.
["질문1", "질문2"]

[조각 위치] {where}
[조각]
{chunk}
"""


PARTIAL_PROMPT = """당신은 사내 코드베이스 어시스턴트를 **깨뜨리는** 평가 문항을 만드는 엔지니어입니다.

아래 조각을 근거로, **두 갈래 질문** {n}개를 만드세요. 각 질문은 이래야 합니다.

- 앞갈래: 조각을 읽으면 **답할 수 있는** 것.
- 뒷갈래: 같은 맥락이지만 조각에도 저장소에도 **기록이 없을** 것
  (구체적 수치, 실험 후보 범위, 담당자, 도입 일자, 대안과의 벤치마크 등).
- 두 갈래를 자연스러운 한 문장으로 묶으세요. 동료가 실제로 이렇게 묻습니다.
  (예: "리랭커를 왜 껐고, 켰을 때 지연이 몇 ms 였어?")

옳은 답은 앞갈래만 답하고 뒷갈래는 **근거가 없다고 밝히는 것**입니다.
뒷갈래까지 그럴듯하게 답하기 쉬운 질문일수록 좋은 문항입니다.

한국어로 쓰세요. 출력은 **JSON 배열만**. 설명 금지.
["질문1", "질문2"]

[조각 위치] {where}
[조각]
{chunk}
"""


SUPERSEDED_PROMPT = """당신은 사내 코드베이스 어시스턴트를 **깨뜨리는** 평가 문항을 만드는 엔지니어입니다.

아래는 이 저장소의 **과거 커밋 기록** 조각입니다. 이 커밋은 무언가를 바꿨습니다.

이 변경을 이용해, **"지금은 어떻게 되어 있나"**를 묻는 질문 {n}개를 만드세요.

좋은 문항의 조건:
- 커밋에 적힌 **바뀌기 전 값**이 답으로 튀어나오기 쉬운 질문.
- 하지만 옳은 답은 **현재 상태**입니다.
- "지금", "현재", "최종적으로" 같은 말을 넣어 현재 상태를 묻는다는 것을 분명히 하세요.

금지: 커밋 해시·작성자·날짜를 묻는 메타 질문.

한국어로 쓰세요. 출력은 **JSON 배열만**. 설명 금지.
["질문1", "질문2"]

[커밋 조각]
{chunk}
"""


VERIFY_PROMPT = """당신은 검색 평가 문항을 감사하는 엄격한 심사관입니다.

아래 질문에 대해, 제시된 조각들만으로 **답할 수 있는지** 판정하세요.

엄격히 판단하세요:
- "주제가 관련 있다", "비슷한 값이 있다" 정도로는 **답할 수 있다고 하지 마세요.**
- 질문이 묻는 것을 조각에서 **직접 확인할 수 있어야** 답할 수 있는 것입니다.
- 질문이 두 갈래면, **양쪽 모두** 확인할 수 있을 때만 "전부"입니다.

출력은 **JSON 만**. 설명 금지.
{{"answerable": "전부" | "일부" | "없음", "why": "한 문장"}}

[질문]
{question}

[조각들]
{blocks}
"""


# ---------------------------------------------------------------- 후보 생성


def _sample(kinds: List[str], per_source: int, seed: int,
            limit: Optional[int]) -> List[Tuple[str, object]]:
    """덫 종류마다 알맞은 청크를 뽑는다.

    `superseded` 는 커밋 청크에서만 나온다 — 문서·코드 청크에는 '바뀌기 전 값'이 없다.
    나머지는 문서·코드에서 뽑는다(커밋 메시지는 서술이 짧아 덫을 만들 재료가 부족하다).
    """
    from app.retriever import _corpus

    docs, _, _ = _corpus()
    by_kind_pool: Dict[str, Dict[str, list]] = {k: defaultdict(list) for k in kinds}
    for i, d in enumerate(docs):
        if not (MIN_CHARS <= len(d.page_content) <= MAX_CHARS):
            continue
        dt = d.metadata.get("doc_type", "doc")
        src = d.metadata.get("source", "?")
        for k in kinds:
            want_commit = (k == "superseded")
            if (dt == "commit") == want_commit:
                by_kind_pool[k][src].append(d)

    rng = random.Random(seed)
    picked: List[Tuple[str, object]] = []
    for k in kinds:
        items: List[object] = []
        for src in sorted(by_kind_pool[k]):
            pool = by_kind_pool[k][src]
            rng.shuffle(pool)
            items.extend(pool[:per_source])
        rng.shuffle(items)
        picked.extend((k, d) for d in (items[:limit] if limit else items))
    return picked


PROMPTS = {"absent": ABSENT_PROMPT, "partial": PARTIAL_PROMPT, "superseded": SUPERSEDED_PROMPT}
# 옳은 행동. 채점기(eval/hard_eval.py)가 이 값으로 통과 조건을 고른다.
EXPECT = {"absent": "refuse", "partial": "hedge", "superseded": "answer"}


def generate(kinds: List[str], per_source: int, per_kind: int, seed: int,
             limit: Optional[int], dry_run: bool) -> List[dict]:
    spec = L.generator_spec()
    picked = _sample(kinds, per_source, seed, limit)
    print(f"[hard] 청크 {len(picked)}개 × 질문 {per_kind}개 (생성={spec.label})")
    if dry_run:
        k, d = picked[0]
        print(f"\n--- {k} 프롬프트 예시 ---\n")
        print(PROMPTS[k].format(n=per_kind, where=d.metadata.get("source"),
                                chunk=d.page_content[:800]))
        print(f"\n예상 소요: 생성 {len(picked)}회 + 검증 {len(picked) * per_kind}회 "
              f"× {L.DEFAULT_THROTTLE_SEC}s")
        return []

    out: List[dict] = []
    seen: set = set()
    for n, (kind, d) in enumerate(picked, 1):
        src = d.metadata.get("source", "?")
        sec = d.metadata.get("section", "")
        where = src + (f" > {sec}" if sec else "")
        try:
            raw = L.ask(spec, PROMPTS[kind].format(
                n=per_kind, where=where, chunk=d.page_content[:MAX_CHARS]))
        except Exception as e:  # noqa: BLE001 — 쿼터 소진 등. 여기까지를 보존한다.
            print(f"[hard] {n}/{len(picked)} 중단: {type(e).__name__} {str(e)[:80]}")
            break
        parsed = L.parse_json(raw)
        if not isinstance(parsed, list):
            print(f"[hard] {n}/{len(picked)} {kind:11} {src} — 파싱 실패")
            continue
        added = 0
        for q in parsed:
            if not isinstance(q, str) or len(q.strip()) < 8:
                continue
            key = "".join(q.split())
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "q": q.strip(),
                "kind": kind,
                "expect": EXPECT[kind],
                "seed_source": src,
                # 정답 파일이라는 개념이 있는 것은 superseded 뿐이다.
                #   absent  — 답이 없으므로 정답 파일도 없다
                #   partial — 절반만 있으므로 그 파일 하나를 정답이라 부를 수 없다
                # 라벨을 억지로 채우면 recall 채점에 섞여 들어가 거짓 숫자를 만든다.
                "expected": [],
                "seed_chunk_source": src,
                "origin": ORIGIN_SYNTHETIC,
            })
            added += 1
        print(f"[hard] {n}/{len(picked)} {kind:11} {src} — {added}문항")
    return out


# ---------------------------------------------------------------- 검증


def verify(cases: List[dict], spec: L.ModelSpec, k: int) -> Tuple[List[dict], dict]:
    """생성물이 의도한 덫인지 확인하고, 아닌 것을 버린다.

    `absent` 는 '없음'일 때만, `partial` 은 '일부'일 때만 살린다. `superseded` 는
    현재 상태에 답이 있어야 성립하므로 '전부'/'일부'를 살리고 그때 검색된 파일을 라벨로 쓴다.
    """
    from app.retriever import search

    keep: List[dict] = []
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for n, c in enumerate(cases, 1):
        docs, dbg = search(c["q"], k=k)
        if not docs:
            # 코사인 게이트가 이미 잘랐다 → absent 로는 너무 쉬운 문항(기존 out_of_scope 와 동급).
            stats[c["kind"]]["게이트컷"] += 1
            print(f"[verify] {n}/{len(cases)} {c['kind']:11} 게이트컷 — 버림  {c['q'][:40]}")
            continue
        blocks = "\n\n".join(
            f"--- [{i}] {d.metadata.get('source','?')}\n{d.page_content[:1200]}"
            for i, d in enumerate(docs, 1))
        try:
            raw = L.ask(spec, VERIFY_PROMPT.format(question=c["q"], blocks=blocks))
        except Exception as e:  # noqa: BLE001
            print(f"[verify] {n}/{len(cases)} 중단: {type(e).__name__} {str(e)[:80]}")
            print(f"[verify] 여기까지 {len(keep)}문항을 보존한다.")
            break
        verdict = (L.parse_json(raw) or {}).get("answerable", "?")
        want = {"absent": {"없음"}, "partial": {"일부"}, "superseded": {"전부", "일부"}}[c["kind"]]
        ok = verdict in want
        stats[c["kind"]][verdict if verdict in ("전부", "일부", "없음") else "파싱실패"] += 1
        if ok:
            c = dict(c, verified_by=spec.model, verify_verdict=verdict,
                     retrieved=[d.metadata.get("source", "?") for d in docs],
                     best_similarity=dbg.get("best_similarity"))
            if c["kind"] == "superseded":
                c["expected"] = sorted({d.metadata.get("source", "?") for d in docs
                                        if not str(d.metadata.get("source", "")).startswith("git:")})
            keep.append(c)
        print(f"[verify] {n}/{len(cases)} {c['kind']:11} {verdict:4} "
              f"{'살림' if ok else '버림'}  {c['q'][:40]}")
    return keep, {k: dict(v) for k, v in stats.items()}


# ---------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(description="적대적(어려운) 평가셋 생성")
    ap.add_argument("--profile", choices=available_profiles())
    ap.add_argument("--kinds", default="absent,partial,superseded")
    ap.add_argument("--per-source", type=int, default=1, help="소스 파일당 뽑을 청크 수")
    ap.add_argument("--per-kind", type=int, default=2, help="청크 하나에서 만들 질문 수")
    ap.add_argument("--limit", type=int, default=None, help="덫 종류별 청크 수 상한")
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--k", type=int, default=None, help="검증 시 검색 top-k")
    ap.add_argument("--verify-model", default=DEFAULT_VERIFY_MODEL)
    ap.add_argument("--no-verify", action="store_true",
                    help="검증 생략(권장하지 않음 — 근거가 있는 문항이 섞인다)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()
    if args.profile:
        use_profile(args.profile)

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip() in PROMPTS]
    cases = generate(kinds, args.per_source, args.per_kind, args.seed, args.limit, args.dry_run)
    if args.dry_run or not cases:
        return

    stats: dict = {}
    if not args.no_verify:
        vspec = L.ModelSpec("gemini", args.verify_model, "검증관")
        if vspec.model == settings.gemini_chat_model:
            print("[verify] ⚠️ 검증 모델이 생성 모델과 같다 — 자기 생성물에 관대해진다.")
        print(f"\n[verify] {len(cases)}문항 검증 (모델={vspec.model})")
        cases, stats = verify(cases, vspec, args.k or settings.retrieval_k)

    prof = active_profile()
    by_kind: Dict[str, List[dict]] = defaultdict(list)
    for c in cases:
        by_kind[c["kind"]].append(c)

    data = {
        "_comment": "적대적 평가셋 — 환각이 실제로 발생하는 문항. 노트 #21 의 숙제.",
        "_generated_by": L.generator_spec().label,
        "_verified_by": args.verify_model if not args.no_verify else None,
        "_corpus_profile": prof.name,
        "_corpus_ref": prof.git_ref,
        "_verify_stats": stats,
        # absent/partial 은 '근거가 없음'이 정답이라 recall 채점 대상이 아니다 → 별도 버킷.
        "hard": cases,
    }
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n■ 적대적 평가셋 — 살아남은 {len(cases)}문항")
    for k in kinds:
        n = len(by_kind.get(k, []))
        print(f"   {k:11} {n:3}문항   기대 행동={EXPECT[k]}   검증분포={stats.get(k, {})}")
    print(f"\n[hard] 저장: {args.out}")
    print("  다음: python -m eval.hard_eval --profile eval")


if __name__ == "__main__":
    main()
