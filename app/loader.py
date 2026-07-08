"""지식원(.md) 파일을 읽어 헤더 인지 청크로 분할.

- Markdown 헤더(#, ##, ###)를 메타데이터로 보존 → 답변 출처가 "파일 > 섹션" 으로 정확히 찍힘
- 큰 섹션은 RecursiveCharacterTextSplitter 로 chunk_size 단위 재분할
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import settings

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _iter_files() -> List[Path]:
    """설정된 폴더들에서 glob 에 맞는 파일 경로를 중복 없이 수집."""
    seen: set[Path] = set()
    files: List[Path] = []
    for root in settings.knowledge_dir_list:
        if not root.exists():
            print(f"[loader] 경고: 경로 없음 → {root}")
            continue
        for pattern in settings.glob_list:
            for path in root.rglob(pattern):
                if path.is_file() and path not in seen:
                    seen.add(path)
                    files.append(path)
    return sorted(files)


def load_and_split() -> List[Document]:
    """지식원 전체를 읽어 청크 리스트로 반환."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS, strip_headers=False
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )

    chunks: List[Document] = []
    files = _iter_files()
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = _display_name(path)

        # 1) 헤더 단위 1차 분할
        try:
            sections = header_splitter.split_text(text)
        except Exception:
            sections = [Document(page_content=text)]
        if not sections:
            sections = [Document(page_content=text)]

        # 2) 큰 섹션 2차 분할 + 메타데이터 부여
        for sec in sections:
            section_title = sec.metadata.get("h3") or sec.metadata.get("h2") or sec.metadata.get("h1") or ""
            for piece in char_splitter.split_text(sec.page_content):
                chunks.append(
                    Document(
                        page_content=piece,
                        metadata={
                            "source": rel,
                            "path": str(path),
                            "section": section_title,
                        },
                    )
                )

    print(f"[loader] 파일 {len(files)}개 → 청크 {len(chunks)}개")
    return chunks


def _display_name(path: Path) -> str:
    """지식원 루트 기준 상대경로(없으면 파일명)로 표시용 이름 생성."""
    for root in settings.knowledge_dir_list:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            continue
    return path.name
