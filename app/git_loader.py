"""git 히스토리 → 커밋 단위 청크.

문서·코드가 "지금 어떤가"를 말한다면, git 이력은 **"언제, 왜 바뀌었나"**를 말한다.
"이 파라미터 언제 도입됐어?", "SL 캡은 왜 없앴어?" 같은 질문의 답은 코드가 아니라
커밋 메시지·변경 이력에 있다.

커밋 하나 = 청크 하나. 본문에 제목·설명·변경 파일 목록을 담고, **커밋 날짜를
메타데이터(commit_date)**로 실어 최신성 가중치·폐기 인지 랭킹의 근거로 쓴다.
diff 전문은 넣지 않는다(노이즈·용량). 파일 목록과 메시지가 "왜"의 대부분을 담는다.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.profiles import active_profile

# 커밋 사이 구분자(메시지에 나올 리 없는 문자열)
_SEP = "\x1e===COMMIT===\x1e"
_FIELD = "\x1f"  # 필드 구분자


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if out.returncode != 0:
        raise RuntimeError(f"git 실패({repo}): {out.stderr.strip()[:200]}")
    return out.stdout


def _load_repo(repo: Path, limit: int) -> List[Document]:
    if not (repo / ".git").exists():
        print(f"[git] 경고: git 저장소 아님 → {repo}")
        return []

    # %H 해시 %ad 날짜 %an 저자 %s 제목 %b 본문 — 필드는 _FIELD, 커밋은 _SEP 로 구분
    fmt = _FIELD.join(["%h", "%ad", "%an", "%s", "%b"]) + _SEP
    raw = _git(
        repo,
        "log",
        f"--max-count={limit}",
        "--date=short",
        f"--pretty=format:{fmt}",
    )

    docs: List[Document] = []
    for block in raw.split(_SEP):
        block = block.strip()
        if not block:
            continue
        parts = block.split(_FIELD)
        if len(parts) < 5:
            continue
        short, date, author, subject, body = parts[:5]

        # 변경 파일 목록(이름만) — "왜 바뀌었나"의 핵심 신호
        files = _git(repo, "show", "--name-only", "--pretty=format:", short).strip()
        file_list = [f for f in files.splitlines() if f.strip()]
        files_text = "\n".join(f"  - {f}" for f in file_list[:40])
        if len(file_list) > 40:
            files_text += f"\n  - … 외 {len(file_list) - 40}개"

        content = (
            f"# 커밋 {short} ({date})\n"
            f"제목: {subject}\n"
            + (f"\n{body.strip()}\n" if body.strip() else "")
            + (f"\n변경 파일:\n{files_text}" if file_list else "")
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": f"git:{short}",
                    "section": subject[:80],
                    "doc_type": "commit",
                    "commit_date": date,
                    "author": author,
                },
            )
        )

    print(f"[git] {repo.name}: 커밋 {len(docs)}개")
    return docs


def load_git() -> List[Document]:
    prof = active_profile()
    if not prof.index_git:
        return []
    docs: List[Document] = []
    for repo in prof.git_repos:
        docs.extend(_load_repo(repo, prof.git_max_commits))
    return docs
