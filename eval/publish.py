"""측정 결과를 **제품 안에서 보이도록** 스냅샷으로 발행한다.

## 왜 필요한가

이 프로젝트의 실제 차별점은 챗봇이 아니라 **평가 체계**다. 그런데 그 결과가 전부
`docs/*.md` 와 `eval/reports/` 안에만 있어서, 데모를 3분 클릭하는 사람에게는
"출처 붙는 RAG 챗봇" 하나로만 보인다. 가장 강한 자산이 화면에서 비가시였다.

그래서 `/eval` 대시보드를 만들고, 이 파일이 그 페이지가 읽을 데이터를 만든다.

## 왜 `eval/reports/` 를 직접 읽지 않는가

`eval/reports/` 는 **git 제외**다(실행할 때마다 갱신되는 산출물이라). clone 직후에는
비어 있으므로, 대시보드가 그걸 직접 읽으면 **다른 사람 화면에서는 빈 페이지**가 된다.
포트폴리오에서 그건 치명적이다 — 보러 온 사람이 보는 것이 정확히 빈 화면이다.

그래서 **선별한 스냅샷**을 `eval/published/summary.json` 으로 만들어 git 에 넣는다.
리포트 원본을 통째로 커밋하지 않는 이유는 두 가지다.

1. 크기 — `panel-answers.json` 류는 답변·근거 본문을 통째로 담아 수 MB 다.
2. 선별이 곧 설계 — 무엇을 보여줄지 고르는 일을 자동화하면 화면이 데이터 덤프가 된다.

## 스냅샷이라는 사실을 숨기지 않는다

생성 시각과 커밋 해시를 함께 넣고 화면에도 띄운다. 지금 코드로 다시 잰 값이 아닐 수
있다는 것을 감추면, 이 프로젝트가 내내 지켜 온 원칙(수치에 조건을 붙여 말한다)을
정작 그 수치를 보여주는 화면에서 어기는 것이 된다.

실행:
    python -m eval.publish            # eval/reports/ → eval/published/summary.json
    python -m eval.publish --check    # 스냅샷이 낡았는지만 확인(CI 용, 쓰기 없음)
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

EVAL_DIR = Path(__file__).resolve().parent
REPORTS = EVAL_DIR / "reports"
PUBLISHED = EVAL_DIR / "published"
SUMMARY = PUBLISHED / "summary.json"
DECISIONS = PUBLISHED / "decisions.json"
STORY = PUBLISHED / "story.json"


def _load(name: str) -> Optional[dict]:
    p = REPORTS / name
    if not p.exists():
        print(f"[publish] 건너뜀 — {name} 없음 (해당 측정을 먼저 실행하세요)")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


NOTES_MD = EVAL_DIR.parent / "docs" / "engineering-notes.md"
# 브랜치는 main 고정 — 커밋 해시로 박으면 링크가 화석이 된다.
NOTES_URL = ("https://github.com/ae-GYEONGWON/codebase-rag-assistant"
             "/blob/main/docs/engineering-notes.md")


def _gh_anchor(heading: str) -> str:
    """GitHub 이 `## 제목` 에 붙이는 앵커 id 를 만든다.

    ★ 줄 번호(`#L78`)를 쓰면 안 된다. GitHub 은 마크다운을 기본으로 **Preview 모드**로
      보여주는데, 그 화면은 줄 앵커를 무시한다(실측: 파일 맨 위로만 간다).
      제목 앵커는 Preview 에서 정확히 동작하고, 렌더링된 상태로 읽히므로 더 낫다.

    규칙: 소문자화 → 영숫자·공백·하이픈·유니코드 문자만 남김 → 공백을 하이픈으로.
    제목을 고치면 앵커가 깨지지만, 그때도 **파일 자체로는 도달**하므로 손해가 제한적이다.
    """
    import re
    import unicodedata

    t = heading.strip().lower()
    t = "".join(c for c in t
                if c.isalnum() or c in " -_" or unicodedata.category(c).startswith("L"))
    return re.sub(r"\s", "-", t)


def _note_index() -> Dict[int, dict]:
    """`docs/engineering-notes.md` 의 `## N. 제목` 을 훑어 번호 → (제목, 링크) 로 만든다.

    손으로 제목을 옮겨 적지 않는 이유는 **어긋나기 때문**이다. 노트 제목을 고쳤는데
    대시보드에 옛 제목이 남으면, 수치의 근거를 보여주겠다는 화면이 거짓말을 하게 된다.
    """
    import re

    if not NOTES_MD.exists():
        print(f"[publish] 경고 — {NOTES_MD.name} 이 없습니다(노트 링크가 비어 나갑니다)")
        return {}
    out: Dict[int, dict] = {}
    for i, line in enumerate(NOTES_MD.read_text(encoding="utf-8").splitlines(), 1):
        m = re.match(r"^##\s+(\d+)\.\s+(.+?)\s*$", line)
        if m:
            n = int(m.group(1))
            title = m.group(2)
            out[n] = {"n": n, "title": title,
                      "url": f"{NOTES_URL}#{_gh_anchor(f'{n}. {title}')}",
                      "line": i}
    return out


COMMIT_URL = "https://github.com/ae-GYEONGWON/codebase-rag-assistant/commit/"


def _commit_subject(sha: str) -> Optional[str]:
    """커밋 제목 한 줄. 저장소 밖에서 발행하거나 얕은 클론이면 없을 수 있다."""
    try:
        r = subprocess.run(["git", "log", "-1", "--format=%s", sha],
                           capture_output=True, text=True, encoding="utf-8", check=True)
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _resolve_notes(ref: str, index: Dict[int, dict]) -> List[dict]:
    """"노트 #13, #17" / "커밋 b856b47" 같은 문자열 → [{kind, label, title, url}, ...].

    화면에는 번호나 해시가 아니라 **제목**을 보여준다. 처음 보는 사람에게 '노트 #7' 은
    아무 뜻도 없는 기호다 — 무엇에 대한 기록인지가 보여야 근거 구실을 한다.

    커밋도 받는 이유는 **모든 측정이 노트에 있지는 않기 때문**이다. 없는 것을 노트인 척
    적으면 근거를 보여주겠다는 화면이 거짓 인용을 하게 된다(실제로 한 번 그랬다).
    """
    import re

    out: List[dict] = []
    for n in (int(x) for x in re.findall(r"#(\d+)", ref or "")):
        if n in index:
            d = index[n]
            out.append({"kind": "note", "label": f"노트 #{n}", **d})
    for sha in re.findall(r"\b([0-9a-f]{7,40})\b", ref or ""):
        subject = _commit_subject(sha)
        if subject:
            out.append({"kind": "commit", "label": f"커밋 {sha[:7]}",
                        "title": subject, "url": COMMIT_URL + sha})
    return out


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 — git 이 없어도 발행은 되어야 한다
        return ""


def _retrieval(name: str, label: str, note: str) -> Optional[dict]:
    """검색 평가 리포트 → 화면에 필요한 것만. miss 목록은 길어서 개수만 남긴다."""
    d = _load(name)
    if not d:
        return None
    return {
        "key": name.replace(".json", ""),
        "label": label,
        "note": note,
        "profile": d.get("profile"),
        "dataset": (d.get("dataset") or {}).get("path"),
        "origins": (d.get("dataset") or {}).get("origins", {}),
        "k": d.get("k"),
        "corpus": d.get("corpus", {}),
        "created_at": d.get("created_at"),
        "suites": [
            {"title": s["title"], "n": s["n"],
             "rows": [{"name": r["name"], "recall": r["recall"], "mrr": r["mrr"],
                       "misses": len(r.get("misses") or [])} for r in s["rows"]]}
            for s in d.get("suites", [])
        ],
        "out_of_scope": {"total": d.get("out_of_scope_total"),
                         "rejected": d.get("out_of_scope_rejected")},
    }


def _judges(name: str = "judge-panel.json") -> Optional[dict]:
    d = _load(name)
    if not d:
        return None
    return {
        "generator": d.get("generator"),
        "threshold": d.get("threshold"),
        "judges": [{"judge": j["judge"], "self_judge": j["self_judge"], "n": j["n"],
                    "mean": j["mean"], "flag_rate": j["flag_rate"]} for j in d.get("judges", [])],
        "pairs": [{"a": p["a"], "b": p["b"], "n": p["n_common"],
                   "mean_diff": p["mean_diff"], "ci95": p["ci95"],
                   "mad": p["mad"], "kappa": p["kappa"], "spearman": p["spearman"],
                   "underpowered": p.get("underpowered", False)}
                  for p in d.get("pairs", [])],
        "caveat": d.get("caveat"),
    }


def _hard() -> Optional[dict]:
    d = _load("hard-eval.json")
    if not d:
        return None
    # 실제로 지어낸 사례는 화면에 그대로 싣는다 — 실패를 보여주는 것이 이 페이지의 목적이다.
    fabricated = [{"kind": r["kind"], "q": r["q"], "why": r.get("why", ""),
                   "fabricated": (r.get("fabricated") or [])[:2]}
                  for r in d.get("records", []) if r.get("pass") is False]
    return {
        "generator": d.get("generator"), "judge": d.get("judge"),
        "n": d.get("n"), "pass_rate": d.get("pass_rate"),
        "by_kind": d.get("by_kind", {}),
        "rule_vs_llm": d.get("absent_rule_vs_llm", {}),
        "failures": fabricated,
    }


def _ann() -> Optional[dict]:
    d = _load("ann-threshold.json")
    if not d:
        return None
    out = {"k": d.get("k"), "dim": d.get("dim"),
           "scaling": [{"n": r["n"], "mb": r["matrix_mb"],
                        "p50": (r.get("brute") or {}).get("p50_ms"),
                        "p95": (r.get("brute") or {}).get("p95_ms")}
                       for r in d.get("results", []) if not r.get("failed")]}
    real = d.get("real")
    if real:
        out["real"] = {"n": real["n"], "n_queries": real["n_queries"],
                       "brute_p50": real["brute"]["p50_ms"],
                       "sweep": real["hnsw"]["sweep"]}
    return out


def build() -> dict:
    decisions: List[dict] = []
    if DECISIONS.exists():
        decisions = json.loads(DECISIONS.read_text(encoding="utf-8")).get("decisions", [])
        index = _note_index()
        missing = []
        for d in decisions:
            d["notes"] = _resolve_notes(d.get("note", ""), index)
            if d.get("note") and not d["notes"]:
                missing.append(d["title"])
        if missing:
            print(f"[publish] 경고 — 노트를 못 찾은 항목: {', '.join(missing)}")
    else:
        print(f"[publish] 경고 — {DECISIONS.name} 이 없습니다(기각 판단 목록이 비어 나갑니다)")

    retrieval = [r for r in (
        _retrieval("private.json", "private — 규모 있는 실제 코드베이스",
                   "3,619 청크. 이 프로젝트의 대표 수치."),
        _retrieval("eval.json", "eval — CI 회귀 게이트 기준",
                   "코퍼스를 태그로 고정해 커밋해도 값이 안 움직인다(노트 #18)."),
        _retrieval("eval-synthetic.json", "합성 258문항 — 규모를 키운 재측정",
                   "수기 20문항의 자가 라벨 편향을 희석하려고 생성(노트 #19)."),
        _retrieval("demo.json", "demo — 라이브 데모 코퍼스",
                   "워킹트리를 보므로 커밋마다 커진다. 게이트에는 쓰지 않는다."),
    ) if r]

    story: dict = {}
    if STORY.exists():
        story = json.loads(STORY.read_text(encoding="utf-8"))
        # 장이 참조하는 판단 제목이 실제로 있는지 확인한다. 오타 하나로 카드가
        # 조용히 사라지면, 그 판단은 화면에서 없었던 일이 된다.
        titles = {d["title"] for d in decisions}
        for ch in story.get("chapters", []):
            unknown = [t for t in ch.get("decisions", []) if t not in titles]
            if unknown:
                print(f"[publish] 경고 — {ch['n']}장이 없는 판단을 참조: {unknown}")
    else:
        print(f"[publish] 경고 — {STORY.name} 이 없습니다(서사 없이 표만 나갑니다)")

    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "story": story,
        "git_sha": _git_sha(),
        "retrieval": retrieval,
        # 평범한 문항과 적대적 문항, **둘 다** 싣는다. 하나만 보이면 이야기가 성립하지 않는다 —
        # "평가셋이 쉬워서"라는 진단이 틀렸다는 것이 어려운 쪽 결과로만 드러나기 때문이다.
        "judges": _judges(),
        "judges_hard": _judges("judge-panel-hard.json"),
        "hard": _hard(),
        "ann": _ann(),
        "decisions": decisions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="측정 결과 스냅샷 발행")
    ap.add_argument("--check", action="store_true",
                    help="쓰지 않고 기존 스냅샷과 달라졌는지만 알려준다")
    args = ap.parse_args()

    data = build()
    PUBLISHED.mkdir(parents=True, exist_ok=True)

    if args.check:
        if not SUMMARY.exists():
            raise SystemExit("스냅샷이 없습니다 — python -m eval.publish 를 실행하세요.")
        old = json.loads(SUMMARY.read_text(encoding="utf-8"))

        # **지금 다시 만들 수 있는 것만** 비교한다.
        # CI 는 `eval` 프로필만 돌리므로 reports/ 에 eval.json 하나만 남는다. 그 상태에서
        # 스냅샷 전체를 비교하면 "판정기·적대적 절이 없다"는 이유로 항상 낡음이 되어,
        # 아무도 못 고치는 실패가 매번 뜬다. 그건 게이트가 아니라 소음이다.
        # 없는 것은 "확인하지 않았다"이지 "달라졌다"가 아니다.
        def _stable(x: Any) -> Any:
            """실행할 때마다 달라지는 필드를 뺀다 — 값이 같은지만 본다."""
            if isinstance(x, dict):
                return {k: _stable(v) for k, v in x.items() if k != "created_at"}
            if isinstance(x, list):
                return [_stable(v) for v in x]
            return x

        checked, stale = [], []
        # 검색 절은 **항목 단위**로 본다. 프로필마다 리포트가 따로라, 이번에 다시 만든
        # 프로필만 비교해야 한다(안 그러면 나머지 3벌이 없다는 이유로 낡음이 된다).
        old_ret = {r["key"]: r for r in old.get("retrieval", [])}
        for r in data.get("retrieval", []):
            checked.append(f"retrieval:{r['key']}")
            if _stable(old_ret.get(r["key"])) != _stable(r):
                stale.append(f"retrieval:{r['key']}")

        # decisions 는 손으로 쓰는 것이라 재생성 대상이 아니다.
        for key in ("judges", "judges_hard", "hard", "ann"):
            fresh = data.get(key)
            if not fresh:                              # 이번에 만들지 못한 절 → 판단 보류
                continue
            checked.append(key)
            if _stable(old.get(key)) != _stable(fresh):
                stale.append(key)

        if not checked:
            print("[publish] 비교할 리포트가 없습니다 — 확인을 건너뜁니다.")
            raise SystemExit(0)
        print(f"[publish] 확인한 절: {', '.join(checked)}")
        if stale:
            print(f"[publish] 낡았습니다 → {', '.join(stale)} — python -m eval.publish 로 갱신하세요.")
            raise SystemExit(1)
        print("[publish] 스냅샷이 최신입니다.")
        raise SystemExit(0)

    SUMMARY.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    n_rows = sum(len(s["rows"]) for r in data["retrieval"] for s in r["suites"])
    print(f"[publish] 저장: {SUMMARY}")
    print(f"  검색 평가 {len(data['retrieval'])}벌({n_rows}행) · "
          f"판정기 {len((data.get('judges') or {}).get('judges', []))}종 · "
          f"적대적 {(data.get('hard') or {}).get('n', 0)}문항 · "
          f"기각 판단 {len(data['decisions'])}건")


if __name__ == "__main__":
    main()
