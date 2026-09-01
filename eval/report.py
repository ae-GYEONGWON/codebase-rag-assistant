"""평가 결과를 **파일로 남기는** 계층 — 회귀 게이트의 그릇.

## 왜 필요했나

지금까지 측정치는 콘솔에만 찍히고 커밋 메시지로 옮겨 적었다. 그러면:

1. **회귀를 자동으로 못 잡는다** — 이전 값과 비교할 대상이 없다.
2. 숫자의 출처가 사람의 손을 거친다 — 옮겨 적다 틀리면 아무도 모른다(실제로 겪은 사고다).
3. 어떤 커밋의 숫자인지 남지 않는다.

그래서 실행 결과를 **기계용(JSON)과 사람용(Markdown)** 두 벌로 떨군다.
JSON 은 CI 가 baseline 과 비교하는 데 쓰고, Markdown 은 아티팩트·PR 코멘트용이다.

## 회귀 게이트

`compare()` 가 baseline 대비 하락을 찾는다. 판정 대상은 **운영 파이프라인 행**과
**범위 밖 거절률** — 즉 실제 서비스 경로만 본다(비교용 행은 하락해도 게이트를 막지 않는다).

임계는 0 이 아니라 작은 허용치를 둔다. 임베딩이 플랫폼·BLAS 구현에 따라 마지막 자리에서
갈릴 수 있고, 그때 동점 청크의 순서가 뒤집히면 recall 이 한 문항 단위로 흔들린다.
"어느 머신에서든 같은 코퍼스"는 보장했지만(→ engineering-notes #16),
"어느 머신에서든 같은 부동소수점"까지는 보장할 수 없다.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPORT_DIR = Path(__file__).resolve().parent / "reports"
BASELINE_DIR = Path(__file__).resolve().parent / "baselines"

# 게이트가 보는 행 — 실제 서비스가 타는 경로
PRIMARY_ROW = "운영 파이프라인"

# 허용치(절대값). 이보다 큰 하락만 회귀로 본다.
TOL_RECALL = 0.01
TOL_MRR = 0.03


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


@dataclass
class RetrieverRow:
    name: str
    recall: float
    mrr: float
    misses: List[str] = field(default_factory=list)


@dataclass
class Suite:
    title: str
    n: int
    rows: List[RetrieverRow] = field(default_factory=list)

    def row(self, name: str) -> Optional[RetrieverRow]:
        return next((r for r in self.rows if r.name == name), None)


@dataclass
class EvalReport:
    profile: str
    collection: str
    k: int
    git_sha: str
    created_at: str
    corpus: Dict[str, int]
    dataset: Dict[str, object]
    suites: List[Suite] = field(default_factory=list)
    out_of_scope_total: int = 0
    out_of_scope_rejected: int = 0

    # --- 조회 ---
    def suite(self, title: str) -> Optional[Suite]:
        return next((s for s in self.suites if s.title == title), None)

    @property
    def rejection_rate(self) -> float:
        return self.out_of_scope_rejected / (self.out_of_scope_total or 1)

    # --- 직렬화 ---
    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_markdown(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path

    def to_markdown(self) -> str:
        L: List[str] = []
        L.append(f"# 검색 평가 리포트 — `{self.profile}` 프로필")
        L.append("")
        L.append(f"- 커밋 `{self.git_sha}` · {self.created_at} · k={self.k}")
        corpus = " / ".join(f"{k} {v}" for k, v in self.corpus.items())
        L.append(f"- 코퍼스: **{sum(self.corpus.values())}청크** ({corpus}) · 컬렉션 `{self.collection}`")
        origins = self.dataset.get("origins") or {}
        L.append(f"- 평가셋: {self.dataset.get('path')} — 라벨 출처 {origins or '미기재'}")
        L.append("")
        for s in self.suites:
            L.append(f"## {s.title} — {s.n}문항")
            L.append("")
            L.append("| retriever | recall@k | MRR | miss |")
            L.append("|---|---:|---:|---:|")
            for r in s.rows:
                mark = "**" if r.name == PRIMARY_ROW else ""
                L.append(f"| {mark}{r.name}{mark} | {r.recall:.0%} | {r.mrr:.2f} | {len(r.misses)} |")
            L.append("")
            misses = (s.row(PRIMARY_ROW) or RetrieverRow("", 0, 0)).misses
            if misses:
                L.append("<details><summary>운영 파이프라인 미스 문항</summary>")
                L.append("")
                for m in misses:
                    L.append(f"- {m}")
                L.append("")
                L.append("</details>")
                L.append("")
        L.append(
            f"## 범위 밖 거절 — {self.out_of_scope_rejected}/{self.out_of_scope_total} "
            f"({self.rejection_rate:.0%})"
        )
        L.append("")
        L.append("> 운영 파이프라인 = RRF + 심볼슬롯 + MMR(λ=1.0 → no-op) + 범위밖 게이트")
        L.append("")
        return "\n".join(L)


def load(path: Path) -> EvalReport:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    suites = [
        Suite(title=s["title"], n=s["n"], rows=[RetrieverRow(**r) for r in s["rows"]])
        for s in d.get("suites", [])
    ]
    d = {**d, "suites": suites}
    return EvalReport(**d)


def compare(baseline: EvalReport, current: EvalReport) -> List[str]:
    """baseline 대비 회귀 목록. 비어 있으면 통과.

    비교 대상은 **운영 파이프라인 행과 범위 밖 거절률**뿐이다.
    vector only / bm25 only 는 비교용 대조군이라 하락해도 서비스 품질과 무관하다.
    """
    problems: List[str] = []

    if baseline.profile != current.profile:
        problems.append(
            f"프로필 불일치: baseline={baseline.profile} vs current={current.profile}"
        )
        return problems

    for b_suite in baseline.suites:
        c_suite = current.suite(b_suite.title)
        if c_suite is None:
            problems.append(f"[{b_suite.title}] 스위트가 사라짐")
            continue
        b_row, c_row = b_suite.row(PRIMARY_ROW), c_suite.row(PRIMARY_ROW)
        if b_row is None or c_row is None:
            continue
        if c_row.recall < b_row.recall - TOL_RECALL:
            problems.append(
                f"[{b_suite.title}] recall {b_row.recall:.0%} → {c_row.recall:.0%} "
                f"(허용치 {TOL_RECALL:.0%})"
            )
        if c_row.mrr < b_row.mrr - TOL_MRR:
            problems.append(
                f"[{b_suite.title}] MRR {b_row.mrr:.2f} → {c_row.mrr:.2f} (허용치 {TOL_MRR:.2f})"
            )
        new_misses = set(c_row.misses) - set(b_row.misses)
        if new_misses:
            problems.append(
                f"[{b_suite.title}] 새로 실패한 문항 {len(new_misses)}건: "
                + " / ".join(sorted(new_misses)[:3])
            )

    if current.rejection_rate < baseline.rejection_rate - TOL_RECALL:
        problems.append(
            f"[범위 밖 거절] {baseline.rejection_rate:.0%} → {current.rejection_rate:.0%}"
        )
    return problems


def new_report(profile, collection, k, corpus, dataset) -> EvalReport:
    return EvalReport(
        profile=profile,
        collection=collection,
        k=k,
        git_sha=_git_sha(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        corpus=corpus,
        dataset=dataset,
    )
