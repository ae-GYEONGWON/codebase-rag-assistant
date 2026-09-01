"""평가용 LLM 호출 계층 — **모델을 갈아끼울 수 있게** 분리한 곳.

## 왜 별도 모듈인가

`app/rag.py` 의 `_llm()` 은 "서비스가 쓰는 모델 하나"를 돌려준다. 그건 서비스에는 맞지만
평가에는 부족하다. 평가에서는 **생성기와 판정기를 다른 모델로 두는 것**이 요구사항이기 때문이다.

같은 모델이 답을 만들고 그 답을 채점하면 자기 답에 후한 점수를 준다 —
**self-enhancement bias** 라는 이름이 붙은 알려진 현상이고, 표준 완화책이 바로 모델 분리다.
그래서 여기서는 모델을 `ModelSpec` 이라는 값으로 다루고, 호출부는 어떤 모델인지 모른 채 쓴다.

## 무료 티어 제약

Gemini 무료 티어는 분당 15요청이다. 평가는 문항 수에 비례해 호출이 늘어나므로
**선제 throttle + 429 재시도**를 호출 계층에 박아둔다(엔지니어링 노트 #12).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Optional

from app.config import settings
from app.rag import _text_of

text_of = _text_of  # 재수출 — Gemini 파트 배열 정규화(노트 #1)


@dataclass(frozen=True)
class ModelSpec:
    """평가에 쓸 모델 한 벌. 이름으로 다루면 리포트에 '누가 채점했는지'가 남는다."""

    provider: str          # gemini | openai
    model: str
    label: str = ""        # 리포트 표기용(생략 시 "provider:model")

    def __post_init__(self):
        if not self.label:
            object.__setattr__(self, "label", f"{self.provider}:{self.model}")

    @property
    def available(self) -> bool:
        if self.provider == "gemini":
            return settings.has_gemini
        if self.provider == "openai":
            return settings.has_openai
        return False


def generator_spec() -> ModelSpec:
    """서비스가 실제로 답변 생성에 쓰는 모델 — 평가의 '피험자'."""
    if settings.active_llm == "openai":
        return ModelSpec("openai", settings.openai_chat_model, "생성기")
    return ModelSpec("gemini", settings.gemini_chat_model, "생성기")


@lru_cache(maxsize=4)
def get_llm(spec: ModelSpec):
    """ModelSpec → LangChain 챗 모델. 같은 spec 은 재사용(모델 재생성 비용 회피)."""
    if spec.provider == "gemini":
        if not settings.has_gemini:
            raise RuntimeError("GOOGLE_API_KEY 가 없습니다(.env 확인).")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=spec.model, google_api_key=settings.google_api_key, temperature=0
        )
    if spec.provider == "openai":
        if not settings.has_openai:
            raise RuntimeError("OPENAI_API_KEY 가 없습니다(.env 확인).")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=spec.model, api_key=settings.openai_api_key, temperature=0)
    raise ValueError(f"알 수 없는 provider: {spec.provider}")


# 무료 티어 분당 15요청 → 호출 사이 최소 간격
DEFAULT_THROTTLE_SEC = 4.5


def call(fn: Callable[[], Any], throttle: float = DEFAULT_THROTTLE_SEC, tries: int = 5) -> Any:
    """LLM 호출을 선제 throttle + 429 재시도로 감싼다.

    429 응답에 서버가 알려주는 재시도 지연이 들어 있으면 그 값을 쓰고, 없으면 20초.
    """
    last: Optional[Exception] = None
    for _ in range(tries):
        time.sleep(throttle)
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                m = re.search(r"retry in ([\d.]+)", msg) or re.search(r"retryDelay'?: ?'?(\d+)", msg)
                wait = float(m.group(1)) if m else 20.0
                print(f"    (429 — {wait:.0f}s 대기 후 재시도)")
                time.sleep(wait + 1)
                last = e
                continue
            raise
    if last:
        raise last
    return fn()


def ask(spec: ModelSpec, prompt: str, throttle: float = DEFAULT_THROTTLE_SEC) -> str:
    """프롬프트 하나 → 텍스트 응답 하나."""
    return text_of(call(lambda: get_llm(spec).invoke(prompt), throttle).content)


def parse_json(raw: str) -> Optional[Dict[str, Any]]:
    """LLM 이 뱉은 JSON 을 최대한 살려 파싱. 실패하면 None(호출부가 세어서 리포트에 남긴다).

    모델이 코드펜스를 두르거나 앞뒤에 설명을 붙이는 일이 잦아 관용적으로 처리한다.
    """
    import json

    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass
    # 본문 어딘가에 박힌 첫 JSON 객체/배열을 건져 본다
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = txt.find(opener), txt.rfind(closer)
        if 0 <= i < j:
            try:
                return json.loads(txt[i : j + 1])
            except json.JSONDecodeError:
                continue
    return None
