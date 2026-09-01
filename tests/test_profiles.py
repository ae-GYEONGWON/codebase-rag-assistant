"""코퍼스 프로필 · 파일 열거 전략 테스트.

인덱스나 임베딩 모델이 필요 없는 순수 로직이라 CI 에서 그대로 돈다.
"""
from pathlib import Path

import pytest

from app.fs_utils import SKIP_DIRS, DirSource, GitTrackedSource
from app.profiles import REPO_ROOT, available_profiles, build_profile


def test_기본_프로필_두개가_등록돼있다():
    assert set(available_profiles()) >= {"demo", "private"}


def test_알수없는_프로필은_사용가능목록과_함께_실패한다():
    with pytest.raises(ValueError) as e:
        build_profile("없는프로필")
    assert "demo" in str(e.value)


def test_프로필별_컬렉션이_분리된다():
    """같은 chroma_dir 를 쓰더라도 컬렉션이 겹치면 두 코퍼스가 섞인다."""
    assert build_profile("demo").collection_name != build_profile("private").collection_name


def test_demo_는_이_저장소를_git_추적_기반으로_본다():
    prof = build_profile("demo")
    assert isinstance(prof.docs, GitTrackedSource)
    assert prof.docs.repo_root == REPO_ROOT
    assert prof.eval_questions.name == "questions.demo.json"


def test_demo_문서에_README_는_있고_추적안되는_파일은_없다():
    """demo 코퍼스 = git 추적 파일. 로컬에만 있는 파일이 섞이면 측정이 머신 종속이 된다."""
    prof = build_profile("demo")
    names = {prof.docs.display_name(p) for p in prof.docs.list_files(prof.doc_globs)}
    assert "README.md" in names
    # .gitignore 로 제외된 개인 문서는 코퍼스에 들어오면 안 된다
    assert "docs/interview-prep.md" not in names
    assert "docs/learning-progress.md" not in names


def test_demo_코드에_추적되는_py_만_들어온다():
    prof = build_profile("demo")
    names = {prof.code.display_name(p) for p in prof.code.list_files(prof.code_globs)}
    assert "app/retriever.py" in names
    assert all(n.endswith(".py") for n in names)


def test_skip_dirs_에_도구설정폴더를_넣지_않는다():
    """회귀 방지 — `.claude` 를 넣었더니 지식원이 그 안에 있던 프로필의 문서가 0개가 됐다.

    지식원 루트가 점(.)으로 시작하는 폴더 안에 있을 수 있으므로, 제외 목록은
    '가상환경·캐시·산출물'로만 한정한다.
    """
    assert ".claude" not in SKIP_DIRS
    assert ".venv" in SKIP_DIRS and "__pycache__" in SKIP_DIRS


def test_dirsource_표시이름_기준(tmp_path: Path):
    root = tmp_path / "pkg"
    (root / "sub").mkdir(parents=True)
    f = root / "sub" / "m.py"
    f.write_text("x = 1", encoding="utf-8")

    # 문서용: 루트 기준 → 'sub/m.py'
    assert DirSource([root], name_base="root").display_name(f) == "sub/m.py"
    # 코드용: 루트의 부모 기준 → 최상위 패키지명을 남긴다 → 'pkg/sub/m.py'
    assert DirSource([root], name_base="parent").display_name(f) == "pkg/sub/m.py"


def test_dirsource_는_캐시폴더를_건너뛴다(tmp_path: Path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.py").write_text("", encoding="utf-8")
    (tmp_path / "real.py").write_text("", encoding="utf-8")

    files = DirSource([tmp_path]).list_files(["*.py"])
    assert [f.name for f in files] == ["real.py"]


def test_gittrackedsource_는_git_아닌_경로에서_명확히_실패한다(tmp_path: Path):
    with pytest.raises(RuntimeError) as e:
        GitTrackedSource(tmp_path).list_files(["*.py"])
    assert "demo" in str(e.value)  # 원인과 해결을 같이 알려줄 것


def test_eval_프로필은_고정_스냅샷을_쓴다():
    """회귀 게이트용 코퍼스는 워킹트리가 아니라 ref 스냅샷이어야 한다(engineering-notes #18)."""
    from app.fs_utils import GitSnapshotSource

    prof = build_profile("eval")
    assert isinstance(prof.docs, GitSnapshotSource)
    assert prof.git_ref == prof.docs.ref
    # demo(워킹트리)와 컬렉션이 겹치면 두 코퍼스가 섞인다
    assert prof.collection_name != build_profile("demo").collection_name


def test_eval_스냅샷은_워킹트리_수정에_영향받지_않는다(tmp_path: Path):
    """이 테스트가 이 설계의 전부다 — 코퍼스가 움직이면 회귀 판정이 성립하지 않는다."""
    prof = build_profile("eval")
    snap_dir = prof.docs.materialize()
    target = REPO_ROOT / "README.md"
    snap_copy = snap_dir / "README.md"
    assert snap_copy.exists()
    # 스냅샷 파일은 워킹트리 파일과 다른 실체다(같은 경로를 가리키지 않는다)
    assert snap_copy.resolve() != target.resolve()


def test_알수없는_ref_는_해결방법까지_알려준다():
    from app.fs_utils import GitSnapshotSource
    from app.profiles import SNAPSHOT_CACHE

    src = GitSnapshotSource(REPO_ROOT, "존재하지-않는-태그", SNAPSHOT_CACHE)
    with pytest.raises(RuntimeError) as e:
        src.list_files(["*.md"])
    assert "git tag" in str(e.value)
