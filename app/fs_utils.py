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
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Sequence, Tuple

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


def _excluded(rel: str, prefixes: Sequence[str]) -> bool:
    """저장소 기준 상대경로 접두사로 제외.

    ⚠️ 폴더 **이름**으로 전역 제외하지 않는다 — 대상 코드베이스에 같은 이름의 폴더가 있으면
    지식이 통째로 사라진다(engineering-notes #16 에서 `.claude` 로 실제로 겪었다).
    """
    return any(rel == p.rstrip("/") or rel.startswith(p) for p in prefixes)


class GitTrackedSource:
    """``git ls-files`` 가 반환하는 **추적 파일**만 코퍼스로 삼는다.

    ``subdirs`` 를 주면 repo 안에서 그 하위 경로로 한 번 더 좁힌다.
    """

    def __init__(self, repo_root: Path, subdirs: Sequence[str] = (),
                 exclude: Sequence[str] = ()) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.subdirs = tuple(subdirs)
        self.exclude = tuple(exclude)

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
            if _is_skipped(path) or _excluded(rel, self.exclude):
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


class GitSnapshotSource:
    """특정 커밋/태그 시점의 저장소를 **스냅샷으로 굳혀** 코퍼스로 삼는다.

    ★ 왜 필요한가 — 회귀 게이트가 성립하려면 코퍼스가 움직이면 안 된다.

    demo 코퍼스는 이 저장소 자기 자신이라, **커밋할 때마다 코퍼스가 바뀐다.**
    그러면 평가 점수의 변화가 '검색 코드가 나빠져서'인지 '문서를 한 편 더 써서'인지
    구분할 수 없다. 실제로 엔지니어링 노트를 한 편 추가했더니 그 주제를 묻는 문항이
    검색에서 밀려 recall 이 떨어졌다(→ engineering-notes #18).

    그래서 평가용 코퍼스는 태그로 고정한다. `git archive <ref>` 로 그 시점 파일들을
    캐시 디렉터리에 풀어놓고 그 위에서 인덱싱하므로, 워킹트리가 아무리 바뀌어도
    **같은 ref 면 언제나 같은 코퍼스**다.
    """

    def __init__(self, repo_root: Path, ref: str, cache_root: Path,
                 exclude: Sequence[str] = ()) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.ref = ref
        self.cache_root = Path(cache_root)
        self.exclude = tuple(exclude)
        self._dir: Optional[Path] = None

    # --- 스냅샷 준비 ---
    def _resolve(self) -> str:
        out = subprocess.run(
            ["git", "-C", str(self.repo_root), "rev-parse", "--verify", f"{self.ref}^{{commit}}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if out.returncode != 0:
            raise RuntimeError(
                f"평가용 코퍼스 ref 를 찾을 수 없습니다: {self.ref!r}\n"
                f"  → 태그를 만드세요: git tag {self.ref} <커밋>\n"
                f"  (평가 코퍼스는 고정돼야 합니다 — engineering-notes #18)"
            )
        return out.stdout.strip()

    def materialize(self) -> Path:
        """ref 시점 파일들을 캐시에 풀고 그 경로를 반환(이미 있으면 재사용)."""
        if self._dir is not None:
            return self._dir
        sha = self._resolve()
        target = self.cache_root / sha[:12]
        marker = target / ".snapshot_ok"
        if not marker.exists():
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)
            archive = target.with_suffix(".zip")
            out = subprocess.run(
                ["git", "-C", str(self.repo_root), "archive", "--format=zip",
                 "-o", str(archive), sha],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if out.returncode != 0:
                raise RuntimeError(f"git archive 실패: {out.stderr.strip()[:200]}")
            with zipfile.ZipFile(archive) as z:
                z.extractall(target)
            archive.unlink(missing_ok=True)
            marker.write_text(sha, encoding="utf-8")
            print(f"[fs] 평가 코퍼스 스냅샷 생성: {self.ref} → {sha[:12]}")
        self._dir = target
        return target

    # --- FileSource 규격 ---
    def list_files(self, globs: Sequence[str]) -> List[Path]:
        root = self.materialize()
        files = DirSource([root]).list_files(globs)
        if not self.exclude:
            return files
        return [f for f in files if not _excluded(f.relative_to(root).as_posix(), self.exclude)]

    def display_name(self, path: Path) -> str:
        """스냅샷 경로를 지우고 저장소 기준 상대경로로 되돌린다.

        정답 라벨(`app/retriever.py` 등)이 스냅샷 경로에 오염되면 안 되므로 반드시 필요하다.
        """
        try:
            return path.relative_to(self.materialize()).as_posix()
        except ValueError:
            return path.name

    def __repr__(self) -> str:
        return f"GitSnapshotSource({self.repo_root} @ {self.ref})"
