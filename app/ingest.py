"""지식원 → 벡터 DB(Chroma) 인덱싱.

사용:
    python -m app.ingest          # 인덱스 (재)생성
    python -m app.ingest --reset  # 기존 컬렉션 삭제 후 재생성
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from langchain_chroma import Chroma

from app.config import settings
from app.embeddings import get_embeddings
from app.loader import load_and_split


def build_index(reset: bool = False) -> int:
    """지식원을 읽어 Chroma 컬렉션을 (재)생성. 인덱싱된 청크 수 반환."""
    chroma_path = Path(settings.chroma_dir)
    if reset and chroma_path.exists():
        shutil.rmtree(chroma_path)
        print(f"[ingest] 기존 인덱스 삭제: {chroma_path}")

    docs = load_and_split()
    if not docs:
        raise RuntimeError(
            "인덱싱할 문서가 없습니다. .env 의 KNOWLEDGE_DIRS / FILE_GLOBS 를 확인하세요."
        )

    embeddings = get_embeddings()
    print(f"[ingest] 임베딩 제공자: {settings.embedding_provider} — 임베딩 계산 중...")

    # from_documents 는 넘긴 문서로 컬렉션을 채우고 디스크에 영속화한다.
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_dir,
    )
    print(f"[ingest] 완료 — 청크 {len(docs)}개를 '{settings.collection_name}' 에 인덱싱")
    return len(docs)


def get_vectorstore() -> Chroma:
    """기존 인덱스를 로드해 조회용 벡터스토어 반환."""
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="지식원 → Chroma 인덱싱")
    parser.add_argument("--reset", action="store_true", help="기존 인덱스 삭제 후 재생성")
    args = parser.parse_args()
    build_index(reset=args.reset)
