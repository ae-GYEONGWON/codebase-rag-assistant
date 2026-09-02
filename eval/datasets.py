"""평가 데이터셋 로딩 — "무엇으로 평가하는가"의 단일 진입점.

평가셋은 **코퍼스 프로필에 딸린 자원**이다. 질문과 정답 경로가 인덱싱 대상에 종속되므로,
프로필이 바뀌면 평가셋도 함께 바뀌어야 한다(안 그러면 정답 파일이 코퍼스에 없어
recall 0% 가 나오고, 그 0% 의 원인을 검색 품질로 오해하게 된다).

    demo    → eval/questions.demo.json   git 추적. 어느 PC·CI 에서도 동일한 셋.
    private → eval/questions.json        대상 코드베이스 정보를 담아 git 제외.

## 스키마

```json
{
  "in_scope":       [{"q": "...", "expected": ["a.md", "b.md"], "origin": "manual"}],
  "in_scope_code":  [{"q": "...", "expected": ["app/x.py"]}],
  "multihop":       [{"q": "...", "hops": [{"axis": "doc",  "expected": [...]},
                                            {"axis": "code", "expected": [...]}]}],
  "out_of_scope":   ["오늘 날씨 어때?"]
}
```

`expected` 는 **OR** 채점(하나라도 top-k 에 잡히면 hit), `multihop` 의 `hops` 는
축별 **AND** 채점(모든 홉을 채워야 정답)이다. `axis` 는 `doc|code|commit`,
커밋 출처 형식은 `git:<short-hash>`.

★ 라벨은 **원본**(소스 grep · git log)에서 만든다. 평가 대상인 검색기의 출력으로
라벨을 만들면 순환이 되어 점수가 자기 자신을 증명하게 된다.

## 라벨 출처(origin)

문항마다 선택 필드 ``origin`` 을 둔다: ``manual``(수기) | ``synthetic``(LLM 생성) |
``synthetic-verified``(LLM 생성 후 사람이 검수). 자가 라벨 편향을 수치화하려면
**어떤 라벨이 어디서 왔는지**를 데이터가 스스로 말해야 한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    # 커밋 축("언제 왜 바뀌었나")은 문서·코드와 성격이 달라 따로 센다.
    in_scope_commit: List[dict] = field(default_factory=list)
    # 적대적 문항(eval/questions.hard.json). recall 채점 대상이 아니다 —
    # `absent` 는 정답이 "없다고 말하기"라서 정답 파일이라는 개념 자체가 없다.
    # 그래서 in_scope 에 합치지 않고 따로 둔다(합치면 recall 이 거짓으로 깎인다).
    hard: List[dict] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return (len(self.in_scope) + len(self.in_scope_code) + len(self.in_scope_commit)
                + len(self.multihop) + len(self.out_of_scope))

    def origin_counts(self) -> Dict[str, int]:
        """라벨 출처별 문항 수 — 자가 라벨 비중을 리포트에 남기기 위한 것."""
        counts: Dict[str, int] = {}
        for case in [*self.in_scope, *self.in_scope_code, *self.in_scope_commit, *self.multihop]:
            key = case.get("origin", ORIGIN_MANUAL)
            counts[key] = counts.get(key, 0) + 1
        if self.out_of_scope:
            counts[ORIGIN_MANUAL] = counts.get(ORIGIN_MANUAL, 0) + len(self.out_of_scope)
        return dict(sorted(counts.items()))

    def summary(self) -> str:
        origins = ", ".join(f"{k} {v}" for k, v in self.origin_counts().items())
        return (
            f"[{self.profile}] {self.path.name}: 문서 {len(self.in_scope)} · "
            f"코드 {len(self.in_scope_code)} · 커밋 {len(self.in_scope_commit)} · "
            f"멀티홉 {len(self.multihop)} · 범위밖 {len(self.out_of_scope)}"
            + (f" · 적대적 {len(self.hard)}" if self.hard else "")
            + f" (라벨: {origins})"
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
        in_scope_commit=data.get("in_scope_commit", []),
        multihop=data.get("multihop", []),
        out_of_scope=data.get("out_of_scope", []),
        hard=data.get("hard", []),
    )
