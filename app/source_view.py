"""근거 원문 조회 — 답변에 붙은 출처를 **파일 전문으로** 열어 준다.

## 왜 필요한가

이 시스템은 "근거를 보여주는 RAG" 를 표방하는데, 화면이 보여 준 것은 220자 발췌뿐이었다.
발췌는 그 발췌가 맞는지 확인할 수단이 아니다 — 앞뒤가 잘려 있으면 반대 뜻이어도 알 수 없다.
근거를 **검증**하려면 원문으로 갈 수 있어야 하고, 그 동선이 없으면 주장과 화면이 어긋난다.

## 경로 파라미터를 받아 파일을 읽는 위험

`?path=` 를 받아 그대로 여는 엔드포인트는 경로 탈출(`../../.env`)의 교과서적 표적이다.
`..` 를 걸러 내거나 경로를 정규화해 루트 밑인지 보는 방식이 흔한데, 심볼릭 링크·유니코드
정규화·드라이브 상대경로(Windows `C:foo`)까지 다 막았는지는 **증명하기 어렵다.**

그래서 검사하지 않고 **허용목록**으로 간다 — 지금 프로필이 실제로 인덱싱한 파일 목록에
있는 표시이름만 연다. 목록에 없는 문자열은 어떤 모양이든 404 다. 우회할 표면이 없다.

부수효과로 제품 의미도 맞는다: **근거로 인용될 수 있는 것만 열린다.**

## 라인 강조 — 재인덱싱 없이

청크 메타데이터에 라인 번호가 없다. 넣으려면 전량 재인덱싱이고, 그러면 청크 id(내용 해시)가
바뀌어 회귀 게이트의 기준선이 함께 움직인다(노트 #18 의 원칙 — 코퍼스 변경은 명시적 행위여야
한다). 조회는 초당 몇 번 일어나는 일이 아니므로 **조회 시점에** 찾는다:

- 코드: `section` 이 심볼명이므로 ast 로 다시 파싱해 그 심볼의 줄 범위를 얻는다.
- 문서: `section` 이 마크다운 헤더 제목이므로 그 헤더 줄부터 다음 동급 헤더 전까지.

못 찾으면 강조 없이 전문만 준다. 강조는 편의지 정확성의 근거가 아니다.
"""
from __future__ import annotations

import ast
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.fs_utils import GitSnapshotSource, GitTrackedSource
from app.profiles import REPO_ROOT, active_profile

# 원문 전송 상한. 이보다 큰 파일은 잘라서 보내고 잘렸다고 알린다 — 브라우저에서
# 수 MB 짜리 텍스트를 한 번에 렌더링하면 화면이 멈춘다.
MAX_BYTES = 400_000

_COMMIT_RE = re.compile(r"^git:([0-9a-f]{4,40})$")


class SourceNotFound(LookupError):
    """허용목록에 없는 출처. 라우터는 이걸 404 로 바꾼다."""


# --- 허용목록 ---------------------------------------------------------------

def _file_map() -> Dict[str, Path]:
    """표시이름 → 실제 경로. **인덱싱 대상 파일만** 들어 있다.

    캐시하지 않는다. 증분 인덱싱(노트 #25)으로 코퍼스가 서버 기동 중에도 바뀌므로,
    캐시하면 방금 추가한 파일의 근거를 열 수 없는 상태가 생긴다. 파일 목록 수집은
    git ls-files 한 번 또는 디렉터리 순회라 조회당 비용이 문제 되지 않는다.
    """
    from app.code_loader import _display_name as code_name
    from app.code_loader import _iter_code_files
    from app.loader import _display_name as doc_name
    from app.loader import _iter_files

    out: Dict[str, Path] = {}
    for p in _iter_files():
        out[doc_name(p)] = p
    for p in _iter_code_files():
        out[code_name(p)] = p
    return out


# --- 원격 링크 --------------------------------------------------------------

@lru_cache(maxsize=1)
def _remote_base() -> Optional[str]:
    """`https://github.com/<owner>/<repo>` 형태의 원격 주소. 없으면 None.

    SSH 주소(`git@github.com:owner/repo.git`)도 웹 주소로 바꾼다 — 개발자는 SSH 로
    clone 하지만 링크를 누르는 사람은 브라우저에 있다.
    """
    try:
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not url:
        return None
    if url.startswith("git@"):                       # git@host:owner/repo.git
        host, _, path = url[4:].partition(":")
        url = f"https://{host}/{path}"
    if url.endswith(".git"):
        url = url[:-4]
    return url if url.startswith("http") else None


@lru_cache(maxsize=8)
def _remote_ref(ref: str) -> str:
    """링크에 박을 ref. `HEAD` 는 커밋 해시로 굳힌다.

    `main` 을 박으면 브랜치가 움직인 뒤 링크가 다른 줄을 가리킨다 — 근거 링크가
    시간이 지나 조용히 틀려지는 것이 근거가 없는 것보다 나쁘다.
    """
    if ref != "HEAD":
        return ref
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return sha or "HEAD"
    except (OSError, subprocess.SubprocessError):
        return "HEAD"


def _remote_url(display: str, line: Optional[int]) -> Optional[str]:
    """이 저장소를 코퍼스로 쓰는 프로필(demo·eval)에서만 원격 링크를 만든다.

    private 프로필의 지식원은 외부 로컬 코드베이스라 공개 주소가 없다. 없는데 만들어
    붙이면 404 로 가는 링크가 되고, 그건 근거로서 링크가 없는 것보다 나쁘다.
    """
    prof = active_profile()
    src = prof.code or prof.docs
    if not isinstance(src, (GitTrackedSource, GitSnapshotSource)):
        return None
    base = _remote_base()
    if not base:
        return None
    anchor = f"#L{line}" if line else ""
    return f"{base}/blob/{_remote_ref(prof.git_ref)}/{display}{anchor}"


def _commit_url(short: str) -> Optional[str]:
    base = _remote_base()
    return f"{base}/commit/{short}" if base else None


# --- 강조 범위 --------------------------------------------------------------

def _code_span(text: str, symbol: str) -> Optional[Tuple[int, int]]:
    """심볼명 → (시작줄, 끝줄). 1-based, 양끝 포함.

    청크의 `section` 은 `_segments()` 가 붙인 이름이라 `Klass.method` 나 `(module)`
    같은 모양일 수 있다. 마지막 마디를 이름으로 보고 찾되, 클래스 안의 메서드는
    그 클래스 안에서만 찾는다 — 같은 이름의 메서드가 여러 클래스에 있으면
    전역 탐색은 엉뚱한 줄을 강조한다.
    """
    if not symbol:
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    def _find(body, name: str):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and node.name == name:
                return node
        return None

    parts = [p for p in symbol.split(".") if p and not p.startswith("(")]
    if not parts:
        return None
    node = _find(tree.body, parts[0])
    for part in parts[1:]:
        if node is None:
            return None
        node = _find(getattr(node, "body", []), part)
    if node is None:
        return None
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return start, (node.end_lineno or node.lineno)


def _doc_span(text: str, heading: str) -> Optional[Tuple[int, int]]:
    """마크다운 헤더 제목 → 그 절의 줄 범위(헤더 줄부터 다음 동급 이상 헤더 직전까지)."""
    if not heading:
        return None
    lines = text.splitlines()
    target = heading.strip()
    start = level = None
    for i, line in enumerate(lines, start=1):
        m = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
        if m and m.group(2).strip() == target:
            start, level = i, len(m.group(1))
            break
    if start is None:
        return None
    for j in range(start, len(lines)):
        m = re.match(r"^(#{1,6})\s+", lines[j])
        if m and len(m.group(1)) <= level:
            return start, j            # lines[j] 는 다음 절의 헤더 → 그 앞줄까지
    return start, len(lines)


# --- 조회 ------------------------------------------------------------------

def _language_of(display: str) -> str:
    ext = display.rsplit(".", 1)[-1].lower() if "." in display else ""
    return {"py": "python", "md": "markdown", "json": "json", "yml": "yaml",
            "yaml": "yaml", "toml": "toml", "sh": "bash", "html": "html",
            "js": "javascript", "css": "css"}.get(ext, "text")


def read_commit(short: str) -> Dict:
    """커밋 근거(`git:abc1234`) → 전체 메시지 + 변경 파일.

    diff 본문은 싣지 않는다. 커밋 하나가 수천 줄인 경우가 있고, 화면에서 필요한 것은
    "무엇이 왜 바뀌었나"라 메시지와 파일 목록이면 충분하다. 그 이상은 원격 링크로 간다.
    """
    from app.git_loader import _git

    prof = active_profile()
    repo = prof.git_repos[0] if prof.git_repos else REPO_ROOT
    try:
        body = _git(repo, "show", "--no-patch",
                    "--pretty=format:%H%n%h%n%an%n%ad%n%s%n%n%b", "--date=short", short)
        names = _git(repo, "show", "--name-status", "--pretty=format:", short).strip()
    except Exception as e:  # noqa: BLE001 — 없는 커밋·git 부재를 404 로 바꾼다
        raise SourceNotFound(f"커밋을 읽을 수 없습니다: {short}") from e
    if not body.strip():
        raise SourceNotFound(f"없는 커밋입니다: {short}")

    lines = body.splitlines()
    full_sha, short_sha, author, date = (lines + ["", "", "", ""])[:4]
    message = "\n".join(lines[4:]).strip()
    changed = [ln for ln in names.splitlines() if ln.strip()]

    text = (f"커밋 {short_sha}  ({date})\n작성자 {author}\n\n{message}\n"
            + ("\n변경 파일\n" + "\n".join(f"  {c}" for c in changed) if changed else ""))
    return {
        "ref": f"git:{short_sha}",
        "doc_type": "commit",
        "display": f"커밋 {short_sha}",
        "language": "text",
        "text": text,
        "line_count": len(text.splitlines()),
        "highlight": None,
        "truncated": False,
        "remote_url": _commit_url(full_sha or short_sha),
    }


def read_source(ref: str, section: str = "") -> Dict:
    """출처 문자열 → 원문 + 강조 범위 + 원격 링크.

    `ref` 는 답변의 근거 카드에 찍힌 `source` 값 그대로다(예 `app/rag.py`, `git:1a2b3c4`).
    """
    ref = (ref or "").strip()
    if not ref:
        raise SourceNotFound("출처가 비어 있습니다")

    m = _COMMIT_RE.match(ref)
    if m:
        return read_commit(m.group(1))

    path = _file_map().get(ref)
    if path is None or not path.is_file():
        raise SourceNotFound(f"인덱싱된 출처가 아닙니다: {ref}")

    raw = path.read_bytes()
    truncated = len(raw) > MAX_BYTES
    text = raw[:MAX_BYTES].decode("utf-8", errors="replace")

    span = None
    if not truncated:                  # 잘린 본문에서 계산한 줄 범위는 틀릴 수 있다
        doc_type = "code" if path.suffix == ".py" else "doc"
        span = _code_span(text, section) if doc_type == "code" else _doc_span(text, section)

    return {
        "ref": ref,
        "doc_type": "code" if path.suffix == ".py" else "doc",
        "display": ref,
        "language": _language_of(ref),
        "text": text,
        "line_count": len(text.splitlines()),
        "highlight": ({"start": span[0], "end": span[1]} if span else None),
        "truncated": truncated,
        "remote_url": _remote_url(ref, span[0] if span else None),
    }


def indexed_refs() -> List[str]:
    """지금 열 수 있는 출처 목록(테스트·진단용)."""
    return sorted(_file_map())
