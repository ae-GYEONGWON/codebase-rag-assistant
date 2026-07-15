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

from app.code_loader import load_code
from app.config import settings
from app.embeddings import get_embeddings
from app.git_loader import load_git
from app.loader import load_and_split


def build_index(reset: bool = False) -> int:
    """지식원을 읽어 Chroma 컬렉션을 (재)생성. 인덱싱된 청크 수 반환."""
    chroma_path = Path(settings.chroma_dir)
    if reset and chroma_path.exists():
        shutil.rmtree(chroma_path)
        print(f"[ingest] 기존 인덱스 삭제: {chroma_path}")

    docs = load_and_split() + load_code() + load_git()
    if not docs:
        raise RuntimeError(
            "인덱싱할 문서가 없습니다. .env 의 KNOWLEDGE_DIRS / FILE_GLOBS / CODE_DIRS 를 확인하세요."
        )

    embeddings = get_embeddings()
    print(f"[ingest] 임베딩 제공자: {settings.embedding_provider} — 임베딩 계산 중...")

    # 청크가 많아 한 번에 넣으면 Chroma 배치 상한(약 5461)에 걸린다 → 나눠서 add.
    store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
    )
    batch = 2000
    for i in range(0, len(docs), batch):
        store.add_documents(docs[i : i + batch])
        print(f"[ingest]   … {min(i + batch, len(docs))}/{len(docs)}")

    n_code = sum(1 for d in docs if d.metadata.get("doc_type") == "code")
    n_commit = sum(1 for d in docs if d.metadata.get("doc_type") == "commit")
    n_doc = len(docs) - n_code - n_commit
    print(
        f"[ingest] 완료 — 청크 {len(docs)}개(문서 {n_doc} / 코드 {n_code} / 커밋 {n_commit})를 "
        f"'{settings.collection_name}' 에 인덱싱"
    )
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
