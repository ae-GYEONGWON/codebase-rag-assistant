"""질문 의도 판별 — 어느 축(문서/코드/커밋)을 묻는지 보고 검색 장치를 켜고 끈다.

## 왜 필요한가 (노트 #17 이 남긴 숙제)

심볼 슬롯(질문에 등장한 함수명의 코드 청크를 top-k 에 강제 편입)은 **한쪽만 좋게 한다.**

    코드 질문  MRR 0.44 → 0.53   (올라감)
    문서 질문  MRR 0.88 → 0.78   (깎임)

이유는 단순하다. "MMR 다양성 계수를 왜 1.0 으로 정했어?" 같은 문서 질문에도 `mmr` 이라는
식별자가 들어 있어서, 슬롯이 코드 청크를 상위에 끼워 넣고 정작 근거인 설계 노트를 밀어낸다.
장치가 나쁜 게 아니라 **켜야 할 때와 끌 때를 안 가린 것**이 문제였다.

## 왜 LLM 을 쓰지 않는가

의도 분류에 LLM 을 부르면 질문 하나에 호출이 한 번 더 붙는다. 검색 자체가 5ms 인데
분류에 1,400ms 를 쓰는 것은 앞뒤가 맞지 않고, 무료 티어에서는 쿼터까지 먹는다.
그래서 **규칙 기반**으로 두고, 규칙이 못 가르는 경우(`unknown`)에는 **기존 동작을 유지**한다.
판별이 틀렸을 때의 손해가 판별을 안 했을 때보다 크면 안 되므로, 애매하면 손대지 않는다.

## 판별 근거

한국어 질문은 축을 꽤 노골적으로 드러낸다.

    코드  "…는 코드에서 어떻게 구현돼 있어?" "…함수 어디 있어?"  → 구현·위치를 묻는다
    커밋  "언제 바뀌었어?" "왜 그 커밋에서…"                    → 변경·시점을 묻는다
    문서  "…를 왜 그렇게 정했어?" "근거가 뭐야?"                 → 이유·설계를 묻는다

이유 어휘와 구현 어휘가 **함께** 나오면("RRF 를 왜 쓰고 코드에서 어떻게 구현돼 있어?")
어느 한쪽으로 몰지 않고 `unknown` 으로 둔다. 멀티홉 질문을 한 축으로 접으면 반대쪽을 잃는다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

# 축을 가리키는 표지어. 조사·어미가 붙어도 잡히도록 어간만 적는다.
_CODE_WORDS = (
    "코드", "구현", "함수", "메서드", "메소드", "클래스", "모듈", "파일 어디", "어느 파일",
    "어디에 있", "어디 있", "어디야", "어디에", "시그니처", "인자", "리턴", "반환값",
    "정의돼", "정의되", "짜여", "작성돼",
)
# 커밋 축은 고정 문구로 잡으면 샌다. "최근에 바뀐" 은 걸리는데 "최근에 **가장 크게** 바뀐"
# 은 사이에 단어가 끼어 놓쳤다(실측). 문구를 계속 늘리는 것은 본 질문에만 맞추는 과적합이라,
# **시점어 + 변화어가 함께 나오는가**로 일반화한다. 둘이 같이 나오면 어순·수식어와 무관하다.
_COMMIT_WORDS = ("커밋", "변경 이력", "히스토리", "이력", "리비전")   # 명시 어휘(단독으로 충분)
_WHEN_WORDS = ("언제", "최근", "처음", "마지막", "예전", "이전", "그때", "지금까지")
_CHANGE_WORDS = ("바뀌", "바꾼", "바꿨", "바뀐", "변경", "추가", "도입", "삭제", "제거",
                 "생겼", "없어졌", "고쳤", "수정")
_DOC_WORDS = (
    "왜", "이유", "근거", "배경", "의도", "설계", "판단", "결정", "트레이드오프",
    "차이", "비교", "장단점", "한계", "정책", "원칙", "무엇을 하는", "어떤 시스템",
    # 비교 질문의 구어형. "뭐가 달라?" 는 실제 데모에서 가장 흔한 문서 질문인데
    # "차이" 만 등록하면 통째로 놓친다.
    "달라", "다른가", "다른 점", "다릅",
)

AXES = ("doc", "code", "commit")


@dataclass(frozen=True)
class Intent:
    """판별 결과. `axis=None` 이면 '모르겠음' — 호출부는 기존 동작을 유지해야 한다."""

    axis: Optional[str]          # doc | code | commit | None(unknown)
    reason: str                  # 무엇을 보고 그렇게 판단했는지(진단 패널 노출용)
    hits: Tuple[str, ...] = ()   # 걸린 표지어
    # 질문이 건드린 축 **전부**. axis 는 그중 하나로 좁힌 결과라 이 정보가 따로 필요하다 —
    # 축이 둘 이상이면 멀티홉이고, 그때는 검색 장치가 아니라 **검색 횟수**를 바꿔야 한다.
    axes: Tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.axis or "unknown"

    @property
    def is_multihop(self) -> bool:
        """축을 둘 이상 물었다 = 한 번의 검색으로는 한쪽이 밀린다(노트: 코드질문 미스의 원인)."""
        return len(self.axes) >= 2


def _matches(question: str, words: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(w for w in words if w in question)


def classify(question: str) -> Intent:
    """질문 → 의도. 규칙만 쓰며 LLM 을 부르지 않는다(지연 0에 가깝다)."""
    q = question.strip()
    if not q:
        return Intent(None, "빈 질문")

    code = _matches(q, _CODE_WORDS)
    doc = _matches(q, _DOC_WORDS)

    # 커밋: 명시 어휘 단독, 또는 (시점어 + 변화어) 동시 출현.
    # 변화어만으로는 잡지 않는다 — "설정을 어떻게 변경해?" 는 사용법 질문이지 이력 질문이 아니다.
    when = _matches(q, _WHEN_WORDS)
    change = _matches(q, _CHANGE_WORDS)
    commit = _matches(q, _COMMIT_WORDS) + ((when + change) if (when and change) else ())
    axes = tuple(a for a, hit in (("doc", doc), ("code", code), ("commit", commit)) if hit)

    # 커밋 표지어는 다른 축과 겹쳐도 우선한다 — "언제 바뀌었나"는 다른 축이 답할 수 없다.
    # 단, 구현 어휘가 같이 오면 멀티홉("왜 바뀌었고 지금 코드는?")이므로 접지 않는다.
    if commit and not code:
        return Intent("commit", f"변경·시점 표지어 {len(commit)}개", commit, axes)

    # 이유 어휘와 구현 어휘가 함께면 멀티홉 — 한 축으로 접으면 반대쪽 근거를 잃는다.
    if code and doc:
        return Intent(None, "구현·이유 어휘가 함께 등장(멀티홉으로 봄)", code + doc, axes)
    if code:
        return Intent("code", f"구현·위치 표지어 {len(code)}개", code, axes)
    if doc:
        return Intent("doc", f"이유·설계 표지어 {len(doc)}개", doc, axes)
    return Intent(None, "표지어 없음", (), axes)


def symbol_slots_for(question: str, default: int) -> Tuple[int, Intent]:
    """의도에 따라 심볼 슬롯 수를 정한다. 반환 = (슬롯 수, 판별 결과).

    - `code`   → 기본값 유지. 심볼 슬롯이 이기는 자리다.
    - `doc`    → **0**. 식별자가 질문에 섞여 있어도 설계 노트를 밀어내지 않게 한다.
    - `commit` → **0**. 커밋 질문의 정답은 커밋 청크이지 함수 본문이 아니다.
    - unknown  → 기본값 유지(손대지 않는다).
    """
    intent = classify(question)
    return (0 if intent.axis in ("doc", "commit") else default), intent
