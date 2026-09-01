"""코퍼스 프로필 — "어떤 지식원을 인덱싱하는가"를 한 덩어리로 묶은 설정.

## 왜 필요했나

지식원 경로가 `.env` 에 흩어져 있어서 대상 코드베이스가 **로컬 절대경로에 묶여 있었다**.
결과:

1. 다른 PC 에서 clone 해도 인덱싱조차 못 한다(그 경로가 없으므로).
2. CI 러너에도 그 경로가 없으니 **회귀 게이트를 돌릴 코퍼스가 없다**.
3. 공개 데모에 비공개 코퍼스를 올릴 수 없다.

셋 다 같은 원인이라 한 번에 푼다. 프로필 = (지식원 · 컬렉션 · 평가셋) 한 벌.

## 기본 프로필

- ``demo``    이 저장소 **자기 자신**. git 추적 파일만 → 어느 머신·CI 에서든 동일한 코퍼스.
              공개 데모용이며 유출 위험이 0 이다(추적 파일 = 이미 공개된 파일).
- ``private`` `.env` 로 지정한 외부 코드베이스. 기존 동작 그대로.

두 프로필은 **같은 chroma 디렉터리 안에서 컬렉션 이름으로 분리**한다.
디렉터리를 나누면 프로필을 바꿀 때마다 전체 재인덱싱이 필요하지만,
컬렉션으로 나누면 두 인덱스가 공존해 전환이 즉시 이뤄진다.

## 확장

새 축(예: 로그, 티켓)을 붙일 때는 아래 ``@register`` 함수 하나를 추가하면 된다.
로더·인덱서·평가는 프로필만 보고 동작하므로 손댈 곳이 없다.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from app.config import Settings, settings
from app.fs_utils import DirSource, FileSource, GitSnapshotSource, GitTrackedSource

# 이 저장소의 루트(app/ 의 부모). demo 프로필의 기준점.
REPO_ROOT = Path(__file__).resolve().parents[1]

# 이 저장소를 코퍼스로 쓸 때(demo·eval) 제외할 경로.
# 평가 산출물은 지식원이 아니라 **이 시스템이 만들어 낸 결과물**이다. 코퍼스에 넣으면
# "평가 결과를 검색해서 평가 결과를 설명하는" 자기참조가 생기고, 문항 텍스트가 코퍼스에
# 섞여 recall 이 부풀 수도 있다(평가셋 질문이 코퍼스 안에 있게 되므로).
SELF_CORPUS_EXCLUDE = ("eval/verification/", "eval/reports/", "eval/baselines/")

# 평가셋 위치도 프로필에 딸린 자원이다(질문·정답 경로가 코퍼스에 종속되므로).
EVAL_DIR = REPO_ROOT / "eval"

# 평가용 코퍼스 스냅샷 캐시(git 제외). ref 당 한 벌.
SNAPSHOT_CACHE = REPO_ROOT / ".eval_corpus"


@dataclass(frozen=True)
class CorpusProfile:
    """인덱싱 대상 한 벌. 로더·인덱서·평가가 참조하는 단일 진실."""

    name: str
    description: str

    docs: Optional[FileSource]
    doc_globs: Tuple[str, ...]

    code: Optional[FileSource]
    code_globs: Tuple[str, ...]

    git_repos: Tuple[Path, ...]
    git_max_commits: int

    collection_name: str
    chroma_dir: str

    eval_questions: Path

    # 코퍼스를 고정할 ref(eval 프로필). 나머지 프로필은 워킹트리 기준이라 HEAD.
    git_ref: str = "HEAD"

    # 시작 화면의 추천 질문. **코퍼스에 실제로 답이 있는 것만** 넣는다 —
    # 클릭했는데 "찾을 수 없습니다"가 나오면 데모가 그 자리에서 끝난다.
    suggestions: Tuple[str, ...] = ()

    # --- 파생 ---
    @property
    def index_docs(self) -> bool:
        return self.docs is not None

    @property
    def index_code(self) -> bool:
        return self.code is not None

    @property
    def index_git(self) -> bool:
        return bool(self.git_repos)

    def summary(self) -> str:
        parts = [
            f"docs={self.docs if self.index_docs else 'off'}",
            f"code={self.code if self.index_code else 'off'}",
            f"git={[str(r) for r in self.git_repos] if self.index_git else 'off'}",
        ]
        return f"[{self.name}] collection={self.collection_name} · " + " · ".join(parts)


# --- 레지스트리 -------------------------------------------------------------

_BUILDERS: Dict[str, Callable[[Settings], CorpusProfile]] = {}


def register(name: str) -> Callable[[Callable[[Settings], CorpusProfile]], Callable]:
    """프로필 빌더 등록 데코레이터. 새 코퍼스 축은 여기 하나만 추가하면 된다."""

    def deco(fn: Callable[[Settings], CorpusProfile]):
        _BUILDERS[name] = fn
        return fn

    return deco


def available_profiles() -> List[str]:
    return sorted(_BUILDERS)


@register("demo")
def _demo(cfg: Settings) -> CorpusProfile:
    """이 저장소 자기 자신 = '자기 자신을 아는 어시스턴트'.

    면접·데모에서 유리한 성질: 면접관이 답변을 `docs/engineering-notes.md` 와
    직접 대조해 **검증할 수 있다**. 규모 자랑이 아니라 동작 시연이 데모의 목적이다.
    """
    tracked = GitTrackedSource(REPO_ROOT, exclude=SELF_CORPUS_EXCLUDE)
    return CorpusProfile(
        name="demo",
        description="이 저장소 자기 자신(git 추적 파일). 공개 데모·CI 용, 유출 위험 0.",
        docs=tracked,
        doc_globs=("*.md",),
        code=tracked if cfg.index_code else None,
        code_globs=("*.py",),
        git_repos=(REPO_ROOT,) if cfg.index_git else (),
        git_max_commits=cfg.git_max_commits,
        collection_name="corpus_demo",
        chroma_dir=cfg.chroma_dir,
        eval_questions=EVAL_DIR / "questions.demo.json",
        suggestions=(
            "이 프로젝트는 무엇을 하는 시스템이야?",
            "하이브리드 검색은 왜 쓰고 얼마나 좋아졌어?",
            "리랭커를 왜 기본으로 껐어?",
            "평가 코퍼스를 태그로 고정한 이유가 뭐야?",
            "브루트포스 검색에서 ANN 으로 언제 바꿔야 해?",
            "코퍼스 프로필은 코드에서 어떻게 구현돼 있어?",
        ),
    )


@register("eval")
def _eval(cfg: Settings) -> CorpusProfile:
    """★회귀 게이트 전용 — 저장소를 **태그 시점으로 고정한** 스냅샷.

    demo 는 워킹트리를 보므로 커밋할 때마다 코퍼스가 커진다. 그 위에서 회귀를 재면
    "검색이 나빠졌다"와 "문서를 한 편 더 썼다"가 구분되지 않는다(engineering-notes #18).
    평가는 움직이지 않는 코퍼스 위에서만 의미가 있으므로 여기서 ref 를 못 박는다.

    코퍼스를 의도적으로 갱신하려면 태그를 옮기고 baseline 을 다시 만든다 — 그 두 동작이
    **명시적이어야** 한다는 것이 이 프로필의 존재 이유다.
    """
    snap = GitSnapshotSource(REPO_ROOT, cfg.eval_corpus_ref, SNAPSHOT_CACHE,
                             exclude=SELF_CORPUS_EXCLUDE)
    return CorpusProfile(
        name="eval",
        description=f"저장소 스냅샷 @ {cfg.eval_corpus_ref} — 회귀 게이트용 고정 코퍼스.",
        docs=snap,
        doc_globs=("*.md",),
        code=snap if cfg.index_code else None,
        code_globs=("*.py",),
        git_repos=(REPO_ROOT,) if cfg.index_git else (),
        git_max_commits=cfg.git_max_commits,
        git_ref=cfg.eval_corpus_ref,
        collection_name=f"corpus_eval_{cfg.eval_corpus_ref.replace('.', '_').replace('/', '_')}",
        chroma_dir=cfg.chroma_dir,
        eval_questions=EVAL_DIR / "questions.demo.json",
        suggestions=(
            "이 프로젝트는 무엇을 하는 시스템이야?",
            "하이브리드 검색은 왜 쓰고 얼마나 좋아졌어?",
            "리랭커를 왜 기본으로 껐어?",
            "평가 코퍼스를 태그로 고정한 이유가 뭐야?",
            "브루트포스 검색에서 ANN 으로 언제 바꿔야 해?",
            "코퍼스 프로필은 코드에서 어떻게 구현돼 있어?",
        ),
    )


@register("private")
def _private(cfg: Settings) -> CorpusProfile:
    """`.env` 로 지정한 외부 코드베이스. 기존 측정치는 전부 이 프로필 기준이다."""
    return CorpusProfile(
        name="private",
        description="`.env` 의 KNOWLEDGE_DIRS/CODE_DIRS/GIT_REPOS 로 지정한 비공개 코드베이스.",
        docs=DirSource(cfg.knowledge_dir_list, name_base="root"),
        doc_globs=tuple(cfg.glob_list),
        code=DirSource(cfg.code_dir_list, name_base="parent") if cfg.index_code else None,
        code_globs=tuple(cfg.code_glob_list),
        git_repos=tuple(cfg.git_repo_list) if cfg.index_git else (),
        git_max_commits=cfg.git_max_commits,
        collection_name=cfg.collection_name,
        chroma_dir=cfg.chroma_dir,
        eval_questions=EVAL_DIR / "questions.json",
    )


def build_profile(name: str, cfg: Settings = settings) -> CorpusProfile:
    key = (name or "").strip().lower()
    if key not in _BUILDERS:
        raise ValueError(
            f"알 수 없는 CORPUS_PROFILE: {name!r}. 사용 가능: {', '.join(available_profiles())}"
        )
    return _BUILDERS[key](cfg)


_override: Optional[str] = None


@lru_cache(maxsize=1)
def active_profile() -> CorpusProfile:
    """활성 프로필. `.env` 의 CORPUS_PROFILE, 또는 `use_profile()` 로 덮어쓴 값."""
    return build_profile(_override or settings.corpus_profile)


def use_profile(name: str) -> CorpusProfile:
    """프로필을 런타임에 전환(CLI `--profile` 용).

    ⚠️ **작업 시작 전에** 호출할 것. 검색기(`app.retriever._corpus`)와 벡터스토어는
    프로필별 결과를 lru_cache 로 들고 있어, 이미 조회가 일어난 뒤 바꾸면 캐시가 어긋난다.
    """
    global _override
    build_profile(name)  # 이름 검증을 먼저 (잘못된 값이면 여기서 실패)
    _override = name
    active_profile.cache_clear()
    return active_profile()
