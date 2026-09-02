"""BM25 토크나이저 — 문자 2-gram 근사와 형태소 분석기(kiwi) 중에서 고른다.

## 왜 2-gram 근사로 시작했나

한국어는 조사·어미가 붙어 어절을 그대로 토큰으로 쓰면 매칭이 어긋난다.

    질문 "리랭커를"   문서 "리랭커는"   → 어절 매칭 실패

형태소 분석기 없이 이걸 넘기는 값싼 방법이 **문자 2-gram** 이다. "리랭커"를 `리랭·랭커`
로 펼치면 조사가 무엇이든 어간 부분이 겹친다. 의존성 0, 사전 관리 0, 미등록어에 강하다.

## 2-gram 이 무엇을 잘못하나

값싼 대신 **정밀도**를 버린다. 흔한 음절 조합("하는", "이다", "에서")이 아무 문서에나
걸려 BM25 점수를 밀어 올린다. 실제로 범위 밖 질문("고양이 키우는 법")조차 BM25 16+ 를
받아서, 이 프로젝트는 BM25 를 **범위 밖 게이트로 쓰지 못하고** 순위 매기기에만 쓴다
(그 판단의 근거는 `app/config.py` 의 `min_similarity` 주석에 있다).

형태소 분석기는 "리랭커/NNG + 를/JKO" 로 갈라 **어간만** 남기므로 이 잡음이 준다.
대신 사전에 없는 신조어·식별자에 약하고, 설치 의존성과 초기화 비용이 붙는다.

## 그래서 고르지 않고 잰다

어느 쪽이 이 코퍼스에서 나은지는 코퍼스에 달렸다(노트 #13 과 같은 교훈). 그래서 둘 다
구현해 두고 `TOKENIZER` 로 갈아끼운 뒤 **평가 하네스로 비교**한다.

    TOKENIZER=ngram   문자 2-gram 근사(기본)
    TOKENIZER=kiwi    형태소 분석기(kiwipiepy 필요)

★ 토크나이저를 바꾸면 BM25 인덱스가 달라진다. 임베딩은 그대로이므로 **재인덱싱은 필요 없지만**,
  프로세스 안의 BM25 캐시는 새로 만들어야 한다(`retriever._corpus_at` 의 캐시 키에 포함).
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List

from app.config import settings

# 영문/숫자/밑줄 토큰(ERR7742, BATCH_DEADLINE …) 과 한글 덩어리를 분리
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[가-힣]+")

# 형태소 중 **내용어**만 남긴다. 조사(J*)·어미(E*)·접사(X*)·기호(S*)는 버린다 —
# 그것들이 남으면 2-gram 과 같은 잡음 문제가 형태소 단위로 재현될 뿐이다.
#
# SL(외국어)·SN(숫자)은 일부러 **뺐다.** ASCII·숫자는 앞의 정규식 패스가 이미 넣으므로
# 여기서 또 넣으면 같은 토큰이 두 번 들어가 BM25 의 단어 빈도(tf)가 두 배로 부풀려진다.
# 식별자가 실제보다 흔해 보이면 IDF 가 깎여 희귀 식별자 매칭이라는 강점이 무뎌진다.
_KIWI_KEEP_PREFIX = ("NN", "NP", "NR", "VV", "VA", "MM", "MAG")


def _ascii_expand(tok: str, out: List[str]) -> None:
    """snake_case 식별자는 조각으로도 색인.

    파일명 `composite_m9s5` 를 통째 토큰으로만 두면 "composite m9s5" 질의가 어디에도
    걸리지 않는다(본문 표기는 `m=9 / s=5`).
    """
    if "_" in tok:
        out.extend(p for p in tok.split("_") if p)


def tokenize_ngram(text: str) -> List[str]:
    """문자 2-gram 근사(기본). 형태소 분석기 없이 조사·어미 변형을 흡수한다."""
    tokens: List[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tokens.append(tok)
        if not tok.isascii() and len(tok) > 1:
            tokens.extend(tok[i : i + 2] for i in range(len(tok) - 1))
        else:
            _ascii_expand(tok, tokens)
    return tokens


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def tokenize_kiwi(text: str) -> List[str]:
    """형태소 분석 후 내용어만 남긴다.

    ASCII 식별자는 kiwi 가 `SL`(외국어) 로 잘 잡지만, 잘라 놓는 경우가 있어
    **원본 정규식 토큰도 함께** 넣는다. 식별자 정확매칭은 이 시스템의 핵심 능력이라
    (희귀 토큰 `ERR7742` 류) 분석기 판단에 통째로 맡기지 않는다.
    """
    tokens: List[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok.isascii():
            tokens.append(tok)
            _ascii_expand(tok, tokens)
    for t in _kiwi().tokenize(text):
        if not t.tag.startswith(_KIWI_KEEP_PREFIX):
            continue
        form = t.form.lower()
        # 한 글자 형태소는 버린다. '으', '수', '것' 같은 것들이 남으면 2-gram 이 겪던
        # "흔한 조각이 아무 문서에나 걸리는" 문제가 형태소 단위로 그대로 재현된다.
        if len(form) < 2:
            continue
        tokens.append(form)
    return tokens


_IMPLS = {"ngram": tokenize_ngram, "kiwi": tokenize_kiwi}


def name() -> str:
    """현재 토크나이저 이름. 캐시 키·리포트 표기에 쓴다."""
    n = (settings.tokenizer or "ngram").lower()
    return n if n in _IMPLS else "ngram"


def tokenize(text: str) -> List[str]:
    return _IMPLS[name()](text)
