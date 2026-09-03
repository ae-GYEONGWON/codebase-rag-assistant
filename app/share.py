"""대화 공유 — 나눈 대화를 링크 하나로 남긴다.

## 왜 필요한가

지금까지 이 데모에는 "이 답변 좀 보세요"라고 건넬 수단이 없었다. 대화는 브라우저 메모리에만
있어서 화면을 캡처하는 것 말고는 남길 방법이 없고, 캡처는 근거 카드도 검색 진단도 접힌 채
찍힌다 — 이 시스템이 보여 주려는 것이 정확히 그 접힌 부분이다.

## 무엇을 저장하는가 — 받은 대로 두지 않는다

들어온 JSON 을 그대로 저장했다가 그대로 돌려주면, 저장소가 **남이 쓴 내용을 내 도메인에서
재생하는 통로**가 된다. 화면이 마크다운을 escape 한 뒤 렌더링하긴 하지만, 방어를 화면 한
겹에만 두면 그 화면을 고칠 때마다 이 통로를 같이 기억해야 한다.

그래서 서버가 **스키마를 강제**한다. 아는 필드만 아는 타입으로 남기고 나머지는 버린다.
남길 것을 고르는 기준은 "공유 링크가 무엇을 보여 줘야 하는가"다 — 답변만 있는 링크는 여느
챗봇 캡처와 다를 게 없으므로, 근거와 검색 진단은 **필드를 하나씩 적어서** 남긴다.

## id 를 내용 해시로 만들지 않는 이유

내용 해시면 같은 대화가 같은 링크가 되어 중복이 없어진다. 그런데 링크를 아는 사람만
볼 수 있는 구조에서 id 가 내용으로 정해지면, **내용을 아는 사람이 링크를 계산할 수 있다.**
공유 링크의 비밀은 예측 불가능성뿐이므로 난수를 쓴다.
"""
from __future__ import annotations

import json
import re
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List

from app.profiles import REPO_ROOT

STORE_DIR = REPO_ROOT / "share_store"

# 한 대화의 저장 상한. 넘으면 거부한다.
MAX_BYTES = 300_000
# 한 대화에 담을 수 있는 턴 수. 무한 대화를 통째로 올리는 것을 막는다.
MAX_TURNS = 60
# 저장소 전체 상한. 도달하면 **거부한다** — 오래된 것을 지우면 이미 배포된 링크가
# 조용히 죽는다. 죽은 링크보다 "지금은 저장할 수 없다"가 정직하다.
MAX_ITEMS = 2000

_ID_LEN = 16          # token_urlsafe(12) ≈ 16자
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16}$")   # token_urlsafe 가 쓰는 문자만


class ShareError(ValueError):
    """저장할 수 없는 요청. 라우터가 400 으로 바꾼다."""


class ShareNotFound(LookupError):
    """없는 공유 id. 라우터가 404 로 바꾼다."""


# --- 스키마 ----------------------------------------------------------------

def _s(v: Any, limit: int) -> str:
    return str(v)[:limit] if isinstance(v, (str, int, float)) else ""


def _source(raw: Any) -> Dict[str, str]:
    d = raw if isinstance(raw, dict) else {}
    return {
        "source": _s(d.get("source"), 300),
        "section": _s(d.get("section"), 300),
        "doc_type": _s(d.get("doc_type"), 20) or "doc",
        "snippet": _s(d.get("snippet"), 1200),
    }


def _num(v: Any) -> Any:
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _retrieval(raw: Any) -> Dict[str, Any]:
    """검색 진단. **아는 필드만** 남긴다.

    이 절을 통째로 버릴 수도 있었지만, 공유 링크가 보여 줘야 할 것이 정확히 이 부분이다 —
    답변만 있는 링크는 여느 챗봇 캡처와 다를 게 없다. 그래서 버리는 대신 스키마를 적는다.
    """
    d = raw if isinstance(raw, dict) else {}
    intent = d.get("intent") if isinstance(d.get("intent"), dict) else None
    picked = d.get("picked") if isinstance(d.get("picked"), list) else []
    return {
        "reason": _s(d.get("reason"), 40),
        "best_similarity": _num(d.get("best_similarity")),
        "best_bm25": _num(d.get("best_bm25")),
        "reranked": bool(d.get("reranked")),
        "symbol_hits": _num(d.get("symbol_hits")),
        "intent": (None if intent is None else {
            "axis": _s(intent.get("axis"), 40),
            "reason": _s(intent.get("reason"), 200),
            "symbol_slots": _num(intent.get("symbol_slots")),
        }),
        "picked": [
            {
                "source": _s(x.get("source"), 300),
                "section": _s(x.get("section"), 300),
                "similarity": _num(x.get("similarity")),
                "bm25": _num(x.get("bm25")),
                "rerank": _num(x.get("rerank")),
            }
            for x in picked[:20] if isinstance(x, dict)
        ],
    }


def _turn(raw: Any) -> Dict[str, Any]:
    d = raw if isinstance(raw, dict) else {}
    role = _s(d.get("role"), 16)
    if role not in ("user", "assistant"):
        raise ShareError(f"알 수 없는 역할입니다: {role!r}")
    srcs = d.get("sources")
    return {
        "role": role,
        "content": _s(d.get("content"), 40_000),
        "sources": [_source(x) for x in srcs[:20]] if isinstance(srcs, list) else [],
        "retrieval": _retrieval(d.get("retrieval")),
        "rewrite": _s((d.get("rewrite") or {}).get("rewritten")
                      if isinstance(d.get("rewrite"), dict) else "", 2000),
        "route": _s((d.get("route") or {}).get("reason")
                    if isinstance(d.get("route"), dict) else "", 500),
        "mode": _s(d.get("mode"), 40),
        "scope": _s(d.get("scope"), 20),
    }


def _normalize(payload: Any) -> Dict[str, Any]:
    d = payload if isinstance(payload, dict) else {}
    turns = d.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ShareError("공유할 대화가 비어 있습니다")
    if len(turns) > MAX_TURNS:
        raise ShareError(f"대화가 너무 깁니다({len(turns)}턴 · 상한 {MAX_TURNS}턴)")
    return {
        "title": _s(d.get("title"), 200) or "제목 없는 대화",
        "created": time.strftime("%Y-%m-%d %H:%M", time.localtime()),
        "turns": [_turn(t) for t in turns],
    }


# --- 저장 · 조회 ------------------------------------------------------------

def _path(share_id: str) -> Path:
    """id → 파일 경로. **id 는 우리가 만든 난수만 유효하다.**

    경로를 만들기 전에 모양을 검사한다. 사용자가 준 문자열로 경로를 조립하는 자리라
    여기서 걸러 두지 않으면 `../` 가 그대로 파일 시스템에 닿는다.
    """
    if not _ID_RE.match(share_id or ""):
        raise ShareNotFound(f"형식이 올바르지 않은 공유 id 입니다: {(share_id or '')[:40]!r}")
    return STORE_DIR / f"{share_id}.json"


def save(payload: Any) -> str:
    data = _normalize(payload)
    blob = json.dumps(data, ensure_ascii=False)
    if len(blob.encode("utf-8")) > MAX_BYTES:
        raise ShareError(f"대화가 너무 큽니다(상한 {MAX_BYTES // 1000}KB)")

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    if sum(1 for _ in STORE_DIR.glob("*.json")) >= MAX_ITEMS:
        raise ShareError("공유 저장소가 가득 찼습니다. 관리자가 정리해야 합니다.")

    share_id = secrets.token_urlsafe(12)[:_ID_LEN]
    _path(share_id).write_text(blob, encoding="utf-8")
    return share_id


def load(share_id: str) -> Dict[str, Any]:
    path = _path(share_id)
    if not path.is_file():
        raise ShareNotFound(f"없는 공유 링크입니다: {share_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def count() -> int:
    return sum(1 for _ in STORE_DIR.glob("*.json")) if STORE_DIR.exists() else 0


def list_ids() -> List[str]:
    """진단·정리용. 화면에는 쓰지 않는다 — 링크를 아는 사람만 보는 구조를 깬다."""
    return sorted(p.stem for p in STORE_DIR.glob("*.json")) if STORE_DIR.exists() else []
