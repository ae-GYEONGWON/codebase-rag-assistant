"""평가 데이터셋 로딩 — "무엇으로 평가하는가"의 단일 진입점.

평가셋은 **코퍼스 프로필에 딸린 자원**이다. 질문과 정답 경로가 인덱싱 대상에 종속되므로,
프로필이 바뀌면 평가셋도 함께 바뀌어야 한다(안 그러면 정답 파일이 코퍼스에 없어
recall 0% 가 나오고, 그 0% 의 원인을 검색 품질로 오해하게 된다).

    demo    → eval/questions.demo.json   git 추적. 어느 PC·CI 에서도 동일한 셋.
    private → eval/questions.json        대상 코드베이스 정보를 담아 git 제외.

## 스키마

```json
{
  "in_scope":       [{"q": "...", "expect_sources": ["a.md", "b.md"]}],
  "in_scope_code":  [{"q": "...", "expect_sources": ["app/x.py"]}],
  "multihop":       [{"q": "...", "expect_axes": {"doc": [...], "code": [...]}}],
  "out_of_scope":   ["오늘 날씨 어때?"]
}
```

`expect_sources` 는 **OR** 채점(하나만 맞으면 정답), `multihop` 의 `expect_axes` 는
축별 **AND** 채점(모든 축을 채워야 정답)이다.

## 라벨 출처(origin)

문항마다 선택 필드 ``origin`` 을 둔다: ``manual``(수기) | ``synthetic``(LLM 생성) |
``synthetic-verified``(LLM 생성 후 사람이 검수). 자가 라벨 편향을 수치화하려면
**어떤 라벨이 어디서 왔는지**를 데이터가 스스로 말해야 한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from app.profiles import active_profile

EVAL_DIR = Path(__file__).resolve().parent
TEMPLATE = EVAL_DIR / "questions.example.json"

# origin 값 — 문항 라벨이 어디서 왔는지
ORIGIN_MANUAL = "manual"
ORIGIN_SYNTHETIC = "synthetic"
ORIGIN_SYNTHETIC_VERIFIED = "synthetic-verified"


@dataclass(frozen=True)
class QuestionSet:
    """평가셋 한 벌. 하네스는 이 객체만 본다."""

    path: Path
    profile: str
    in_scope: List[dict]
    in_scope_code: List[dict]
    multihop: List[dict]
    out_of_scope: List[str]

    @property
    def n_total(self) -> int:
        return len(self.in_scope) + len(self.in_scope_code) + len(self.multihop) + len(self.out_of_scope)

    def origin_counts(self) -> Dict[str, int]:
        """라벨 출처별 문항 수 — 자가 라벨 비중을 리포트에 남기기 위한 것."""
        counts: Dict[str, int] = {}
        for case in [*self.in_scope, *self.in_scope_code, *self.multihop]:
            key = case.get("origin", ORIGIN_MANUAL)
            counts[key] = counts.get(key, 0) + 1
        if self.out_of_scope:
            counts[ORIGIN_MANUAL] = counts.get(ORIGIN_MANUAL, 0) + len(self.out_of_scope)
        return dict(sorted(counts.items()))

    def summary(self) -> str:
        origins = ", ".join(f"{k} {v}" for k, v in self.origin_counts().items())
        return (
            f"[{self.profile}] {self.path.name}: 문서 {len(self.in_scope)} · "
            f"코드 {len(self.in_scope_code)} · 멀티홉 {len(self.multihop)} · "
            f"범위밖 {len(self.out_of_scope)} (라벨: {origins})"
        )


def questions_path() -> Path:
    """활성 프로필의 평가셋 경로. 없으면 형식만 담은 템플릿으로 폴백."""
    p = active_profile().eval_questions
    return p if p.exists() else TEMPLATE


def load_questions(path: Path | None = None) -> QuestionSet:
    qpath = path or questions_path()
    data = json.loads(qpath.read_text(encoding="utf-8"))
    return QuestionSet(
        path=qpath,
        profile=active_profile().name,
        in_scope=data.get("in_scope", []),
        in_scope_code=data.get("in_scope_code", []),
        multihop=data.get("multihop", []),
        out_of_scope=data.get("out_of_scope", []),
    )
