"""지식원 → 벡터 DB(Chroma) 인덱싱.

인덱싱 대상은 **코퍼스 프로필**(app/profiles.py)이 결정한다. 이 모듈은 "무엇을 넣을지"를
모르고, 프로필이 준 로더 결과를 담기만 한다.

사용:
    python -m app.ingest                      # 활성 프로필로 인덱싱
    python -m app.ingest --reset              # 이 프로필의 컬렉션만 지우고 재생성
    python -m app.ingest --profile demo       # 프로필을 지정해 인덱싱
"""
from __future__ import annotations

import argparse
from typing import Optional

from langchain_chroma import Chroma

from app.code_loader import load_code
from app.config import settings
from app.embeddings import get_embeddings
from app.git_loader import load_git
from app.loader import load_and_split
from app.profiles import CorpusProfile, active_profile, available_profiles, use_profile


def _store(profile: CorpusProfile) -> Chroma:
    return Chroma(
        collection_name=profile.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=profile.chroma_dir,
    )


def build_index(reset: bool = False, profile: Optional[CorpusProfile] = None) -> int:
    """지식원을 읽어 프로필의 컬렉션을 (재)생성. 인덱싱된 청크 수 반환."""
    prof = profile or active_profile()
    print(f"[ingest] {prof.summary()}")

    store = _store(prof)
    if reset:
        # 디렉터리를 통째로 지우지 않는다 — 프로필들이 같은 chroma_dir 를 공유하므로
        # rmtree 하면 다른 프로필의 인덱스까지 날아간다. 컬렉션 단위로만 드롭.
        try:
            store.delete_collection()
            print(f"[ingest] 기존 컬렉션 삭제: {prof.collection_name}")
        except Exception as e:  # 컬렉션이 아직 없으면 정상 상황
            print(f"[ingest] 삭제 건너뜀({type(e).__name__}) — 기존 컬렉션 없음")
        store = _store(prof)

    docs = load_and_split() + load_code() + load_git()
    if not docs:
        raise RuntimeError(
            f"인덱싱할 문서가 없습니다(프로필={prof.name}). "
            "demo 라면 git 저장소인지, private 이라면 .env 의 경로를 확인하세요."
        )

    print(f"[ingest] 임베딩 제공자: {settings.embedding_provider} — 임베딩 계산 중...")

    # 청크가 많아 한 번에 넣으면 Chroma 배치 상한(약 5461)에 걸린다 → 나눠서 add.
    batch = 2000
    for i in range(0, len(docs), batch):
        store.add_documents(docs[i : i + batch])
        print(f"[ingest]   … {min(i + batch, len(docs))}/{len(docs)}")

    n_code = sum(1 for d in docs if d.metadata.get("doc_type") == "code")
    n_commit = sum(1 for d in docs if d.metadata.get("doc_type") == "commit")
    n_doc = len(docs) - n_code - n_commit
    print(
        f"[ingest] 완료 — 청크 {len(docs)}개(문서 {n_doc} / 코드 {n_code} / 커밋 {n_commit})를 "
        f"'{prof.collection_name}' 에 인덱싱"
    )
    return len(docs)


def get_vectorstore() -> Chroma:
    """활성 프로필의 인덱스를 로드해 조회용 벡터스토어 반환."""
    return _store(active_profile())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="지식원 → Chroma 인덱싱")
    parser.add_argument("--reset", action="store_true", help="이 프로필의 컬렉션 삭제 후 재생성")
    parser.add_argument(
        "--profile",
        choices=available_profiles(),
        help="코퍼스 프로필(기본: .env 의 CORPUS_PROFILE)",
    )
    args = parser.parse_args()
    if args.profile:
        use_profile(args.profile)
    build_index(reset=args.reset)
