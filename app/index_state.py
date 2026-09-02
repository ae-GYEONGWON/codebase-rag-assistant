"""인덱스 버전 도장 — 서빙 프로세스가 **재기동 없이** 인덱스 변경을 알아채게 한다.

## 왜 필요한가

검색기는 청크 본문·임베딩 행렬·BM25 인덱스를 프로세스 메모리에 캐시한다(`retriever._corpus`).
캐시가 없으면 질문마다 수천 청크를 Chroma 에서 다시 꺼내야 해서 쓸 수 없다.

그런데 이 캐시 때문에 **인덱싱과 서빙이 묶여 있었다.** 인덱싱은 별도 프로세스에서 도는데
서버는 예전 캐시를 계속 들고 있으니, 재인덱싱하면 서버를 내렸다 올려야 했다
(HANDOFF '알려진 한계'에 그대로 적혀 있던 항목).

## 어떻게 푸는가

`collection.count()` 로 판별하면 안 된다. 파일 한 줄만 고치면 청크 수는 그대로이고 **내용만**
바뀌는데, 그 경우를 통째로 놓친다. 그래서 인덱서가 쓰기를 끝낼 때마다 **버전 토큰 파일**에
새 값을 찍고, 검색기는 질문마다 그 파일을 읽어 토큰이 바뀌었을 때만 캐시를 다시 만든다.

토큰 읽기는 작은 파일 하나를 읽는 것이라 마이크로초 단위다 — 검색 자체가 5ms 인데
그 앞에 붙는 비용으로 무시할 수 있다. 반대로 DB 를 매번 조회하는 방식은 그 균형이 깨진다.

프로세스 간 통신 수단으로 파일을 쓰는 이유는 단순하다. 인덱서와 서버가 **이미 같은
`chroma_db/` 디렉터리를 공유**하고 있어서, 새 의존성(레디스·시그널) 없이 그 자리에 둘 수 있다.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path


def _path(chroma_dir: str, collection: str) -> Path:
    return Path(chroma_dir) / f".index_version_{collection}"


def stamp(chroma_dir: str, collection: str) -> str:
    """인덱스가 바뀌었음을 기록하고 새 버전 토큰을 반환한다(인덱서가 호출)."""
    p = _path(chroma_dir, collection)
    p.parent.mkdir(parents=True, exist_ok=True)
    token = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    p.write_text(token, encoding="utf-8")
    return token


def version(chroma_dir: str, collection: str) -> str:
    """현재 버전 토큰. 파일이 없으면 빈 문자열(= 아직 도장이 찍힌 적 없는 인덱스).

    읽기 실패는 예외로 올리지 않는다. 인덱서가 파일을 쓰는 **중간**에 검색이 들어올 수 있고,
    그때 서빙이 죽는 것보다 한 번 낡은 캐시를 쓰는 편이 낫다 — 다음 질문에서 갱신된다.
    """
    try:
        return _path(chroma_dir, collection).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def clear(chroma_dir: str, collection: str) -> None:
    """버전 파일 제거(컬렉션을 드롭할 때). 없으면 조용히 넘어간다."""
    try:
        os.remove(_path(chroma_dir, collection))
    except OSError:
        pass
