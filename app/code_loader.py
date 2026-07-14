"""소스코드(.py) → AST 기반 청크.

문서와 달리 코드는 **의미 단위가 함수/클래스**다. 고정 길이로 자르면 함수가 허리에서
잘려 "이 코드가 무엇을 하는지" 알 수 없는 조각이 생긴다. 그래서 ast 로 파싱해
top-level 함수·클래스(큰 클래스는 메서드) 단위로 자른다.

또 하나 중요한 점: 코드 조각만 임베딩하면 **어느 파일의 무슨 함수인지가 사라진다**.
각 청크 앞에 `파일 > 심볼` 컨텍스트 헤더를 붙여 임베딩·BM25 양쪽이 위치 정보를 갖게 한다.
(Contextual Retrieval 의 경량 버전 — LLM 호출 없이 구조 정보만으로 같은 효과를 노림)
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

_SKIP_DIRS = {"__pycache__", ".venv", "venv", ".git", "node_modules", "build", "dist"}


def _iter_code_files() -> List[Path]:
    seen: set[Path] = set()
    files: List[Path] = []
    for root in settings.code_dir_list:
        if not root.exists():
            print(f"[code] 경고: 경로 없음 → {root}")
            continue
        for pattern in settings.code_glob_list:
            for path in root.rglob(pattern):
                if not path.is_file() or path in seen:
                    continue
                if any(part in _SKIP_DIRS for part in path.parts):
                    continue
                seen.add(path)
                files.append(path)
    return sorted(files)


def _display_name(path: Path) -> str:
    """코드 루트의 **부모** 기준 상대경로 → 'app/broker/broker_main.py' 처럼 보이게."""
    for root in settings.code_dir_list:
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        return (Path(root.name) / rel).as_posix()
    return path.name


def _segments(tree: ast.Module, src_lines: List[str]) -> Iterator[Tuple[str, str]]:
    """(심볼명, 소스조각) 을 순회. 모듈 도입부 → top-level 함수/클래스 순."""
    body = list(tree.body)
    defs = [n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]

    # 1) 모듈 도입부: 첫 def 이전(도크스트링·상수·import). 상수 정의가 여기 몰려 있어 검색 가치가 크다.
    first_line = min((n.lineno for n in defs), default=len(src_lines) + 1)
    head = "\n".join(src_lines[: first_line - 1]).strip()
    if head:
        yield "(module)", head

    # 2) 함수/클래스. 큰 클래스는 메서드 단위로 쪼갠다(클래스 통째로는 청크가 너무 커짐).
    for node in defs:
        seg = ast.get_source_segment("\n".join(src_lines), node) or ""
        if not seg.strip():
            continue
        if isinstance(node, ast.ClassDef) and len(seg) > settings.chunk_size:
            methods = [
                m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            # 클래스 선언부(docstring·클래스 변수)는 따로 남긴다
            first_m = min((m.lineno for m in methods), default=node.end_lineno or node.lineno)
            decl = "\n".join(src_lines[node.lineno - 1 : first_m - 1]).strip()
            if decl:
                yield node.name, decl
            for m in methods:
                m_seg = ast.get_source_segment("\n".join(src_lines), m) or ""
                if m_seg.strip():
                    yield f"{node.name}.{m.name}", m_seg
        else:
            yield node.name, seg


def load_code() -> List[Document]:
    """소스코드 전체를 읽어 청크 리스트로 반환. 문법 오류 파일은 건너뛴다."""
    if not settings.index_code:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: List[Document] = []
    files = _iter_code_files()
    skipped = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            skipped += 1
            continue

        rel = _display_name(path)
        src_lines = text.splitlines()

        for symbol, seg in _segments(tree, src_lines):
            # 긴 함수는 재분할하되, 조각마다 컨텍스트 헤더를 다시 붙인다.
            for piece in splitter.split_text(seg):
                header = f"# 파일: {rel}\n# 심볼: {symbol}\n"
                chunks.append(
                    Document(
                        page_content=header + piece,
                        metadata={
                            "source": rel,
                            "path": str(path),
                            "section": symbol,
                            "doc_type": "code",
                        },
                    )
                )

    print(f"[code] 파일 {len(files)}개(파싱실패 {skipped}) → 청크 {len(chunks)}개")
    return chunks


def list_code_sources() -> List[str]:
    return [_display_name(p) for p in _iter_code_files()] if settings.index_code else []
