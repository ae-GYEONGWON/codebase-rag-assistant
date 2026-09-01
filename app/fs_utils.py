"""지식원 파일 열거 — "무엇이 코퍼스인가"를 정하는 단 한 곳.

로더(문서·코드)마다 제각각 rglob 을 돌리던 것을 전략 객체로 뽑았다.
코퍼스 프로필([[app.profiles]])이 이 전략을 골라 끼운다.

두 가지 전략:

- ``DirSource``        지정 폴더를 rglob. 대상 코드베이스가 이 repo 밖에 있을 때(private).
- ``GitTrackedSource`` ``git ls-files`` 로 **추적 파일만**. demo 프로필용.

★ demo 를 '추적 파일'로 정의한 이유 — 측정 재현성.
   폴더 walk 로 잡으면 로컬에만 있는 개인 메모(.gitignore 대상)가 코퍼스에 섞인다.
   그러면 **같은 코드인데 PC·CI 마다 청크 수와 recall 이 달라져** 회귀 게이트가 성립하지 않는다.
   추적 파일 집합은 커밋 해시로 고정되므로 어느 머신에서든 동일하다.
   유출 방지(비공개 파일은 애초에 gitignore 대상)는 그 결정에 딸려오는 부산물이지 목적이 아니다.
"""
from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path
from typing import Iterable, List, Protocol, Sequence, Tuple

# 코퍼스에서 항상 제외 — 가상환경·캐시·산출물. 여기 없으면 .venv 안의 수천 개
# 서드파티 .md/.py 가 코퍼스로 딸려 들어온다(폴더 walk 방식의 실제 함정).
#
# ⚠️ 도구 설정 폴더(.claude 등)는 넣지 말 것 — 지식원이 그런 폴더 **안**에 있을 수 있다.
#    (실제로 private 프로필의 문서축이 `<repo>/.claude/memory` 라 넣었다가 문서 0개가 됐다.)
#    이 repo 자신의 로컬 노트는 demo 가 git 추적 파일만 보므로 이미 제외된다.
SKIP_DIRS = frozenset(
    {
        "__pycache__",
        ".venv",
        "venv",
        ".git",
        "node_modules",
        "build",
        "dist",
        "chroma_db",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "htmlcov",
        ".idea",
        ".vscode",
    }
)


def _is_skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


class FileSource(Protocol):
    """지식원 파일을 열거하고 표시용 이름을 붙이는 전략."""

    def list_files(self, globs: Sequence[str]) -> List[Path]:
        """glob 에 맞는 파일 경로를 중복 없이 정렬해 반환."""
        ...

    def display_name(self, path: Path) -> str:
        """출처 표기에 쓸 상대경로(예: ``app/retriever.py``)."""
        ...


class DirSource:
    """지정한 폴더들을 rglob 으로 훑는다.

    ``name_base``:
      - ``"root"``   루트 기준 상대경로 → ``memory/foo.md``
      - ``"parent"`` 루트의 **부모** 기준 → ``app/retriever.py`` (코드용. 최상위 패키지명을 남긴다)
    """

    def __init__(self, roots: Iterable[Path], name_base: str = "root") -> None:
        self.roots: Tuple[Path, ...] = tuple(Path(r) for r in roots)
        self.name_base = name_base

    def list_files(self, globs: Sequence[str]) -> List[Path]:
        seen: set[Path] = set()
        files: List[Path] = []
        for root in self.roots:
            if not root.exists():
                print(f"[fs] 경고: 경로 없음 → {root}")
                continue
            for pattern in globs:
                for path in root.rglob(pattern):
                    if not path.is_file() or path in seen or _is_skipped(path):
                        continue
                    seen.add(path)
                    files.append(path)
        return sorted(files)

    def display_name(self, path: Path) -> str:
        for root in self.roots:
            base = root.parent if self.name_base == "parent" else root
            try:
                return path.relative_to(base).as_posix()
            except ValueError:
                continue
        return path.name

    def __repr__(self) -> str:  # 프로필 요약 출력용
        return f"DirSource({', '.join(str(r) for r in self.roots)})"


class GitTrackedSource:
    """``git ls-files`` 가 반환하는 **추적 파일**만 코퍼스로 삼는다.

    ``subdirs`` 를 주면 repo 안에서 그 하위 경로로 한 번 더 좁힌다.
    """

    def __init__(self, repo_root: Path, subdirs: Sequence[str] = ()) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.subdirs = tuple(subdirs)

    def _tracked(self) -> List[Path]:
        out = subprocess.run(
            ["git", "-C", str(self.repo_root), "ls-files", "-z"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if out.returncode != 0:
            raise RuntimeError(
                f"git ls-files 실패({self.repo_root}): {out.stderr.strip()[:200]}\n"
                "demo 프로필은 git 저장소 안에서만 동작합니다."
            )
        return [self.repo_root / rel for rel in out.stdout.split("\0") if rel]

    def list_files(self, globs: Sequence[str]) -> List[Path]:
        files: List[Path] = []
        for path in self._tracked():
            rel = path.relative_to(self.repo_root).as_posix()
            if self.subdirs and not any(
                rel == s or rel.startswith(s.rstrip("/") + "/") for s in self.subdirs
            ):
                continue
            if _is_skipped(path):
                continue
            if not any(fnmatch.fnmatch(path.name, g) for g in globs):
                continue
            # 추적 목록에 있으나 워킹트리에 없는 경우(체크아웃 중 삭제 등)는 건너뛴다.
            if not path.is_file():
                continue
            files.append(path)
        return sorted(files)

    def display_name(self, path: Path) -> str:
        try:
            return path.relative_to(self.repo_root).as_posix()
        except ValueError:
            return path.name

    def __repr__(self) -> str:
        scope = f", subdirs={list(self.subdirs)}" if self.subdirs else ""
        return f"GitTrackedSource({self.repo_root}{scope})"
