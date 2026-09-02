"""지식원 → 벡터 DB(Chroma) 인덱싱.

인덱싱 대상은 **코퍼스 프로필**(app/profiles.py)이 결정한다. 이 모듈은 "무엇을 넣을지"를
모르고, 프로필이 준 로더 결과를 담기만 한다.

## 전체 재색인 vs 증분

기본은 **증분**이다. 청크 id 를 내용 해시로 두면 "무엇이 바뀌었나"가 집합 연산이 된다.

    새 id 집합 - 기존 id 집합 = 추가할 것
    기존 id 집합 - 새 id 집합 = 지울 것
    교집합                    = 손대지 않을 것  ← 임베딩을 다시 계산하지 않는다

임베딩이 인덱싱 비용의 거의 전부이므로, 문서 한 줄을 고쳤을 때 3,600 청크를 다시 임베딩할
이유가 없다. 파일 단위 mtime 비교가 아니라 **청크 단위 내용 해시**를 쓰는 이유는, 한 파일을
고쳐도 그 파일의 청크 대부분은 그대로이기 때문이다.

쓰기를 마치면 **버전 도장**을 찍는다(app/index_state.py) — 돌고 있는 서버가 재기동 없이
새 인덱스를 집어 든다.

사용:
    python -m app.ingest                      # 활성 프로필로 증분 인덱싱
    python -m app.ingest --full               # 증분 없이 전량 재색인(정합성 의심 시)
    python -m app.ingest --reset              # 이 프로필의 컬렉션만 지우고 재생성
    python -m app.ingest --dry-run            # 무엇이 바뀌는지만 계산(쓰기 없음)
    python -m app.ingest --profile demo       # 프로필을 지정해 인덱싱
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from typing import Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app import index_state
from app.code_loader import load_code
from app.config import settings
from app.embeddings import get_embeddings
from app.git_loader import load_git
from app.loader import load_and_split
from app.profiles import CorpusProfile, active_profile, available_profiles, use_profile

# Chroma 배치 상한(약 5461)에 걸리지 않게 나눠 넣는다.
_ADD_BATCH = 2000
# delete 는 id 목록을 SQL IN 절로 푸는데, 너무 길면 sqlite 변수 한도(999)에 걸린다.
_DEL_BATCH = 500


def _store(profile: CorpusProfile) -> Chroma:
    return Chroma(
        collection_name=profile.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=profile.chroma_dir,
    )


def chunk_id(doc: Document, ordinal: int = 0) -> str:
    """청크의 **내용 주소**. 같은 내용이면 언제 어디서 만들어도 같은 id 가 된다.

    출처·섹션까지 넣는 이유: 똑같은 문장이 두 파일에 있을 때 하나로 합쳐지면
    출처 하나가 사라진다. 그러면 "어디서 나온 말이냐"에 답할 수 없다.

    `ordinal` 은 한 파일·한 섹션 안에서 **완전히 동일한 청크**가 두 번 나올 때만 쓰인다
    (반복되는 상용구 등). 그런 경우까지 하나로 접으면 삭제 판정이 어긋난다.
    """
    m = doc.metadata
    key = "|".join([
        str(m.get("source", "")), str(m.get("section", "")),
        str(m.get("doc_type", "")), str(ordinal), doc.page_content,
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def assign_ids(docs: List[Document]) -> List[str]:
    """청크 목록 → id 목록. 중복 내용은 ordinal 을 올려 구분한다."""
    seen: Counter = Counter()
    ids: List[str] = []
    for d in docs:
        base = (d.metadata.get("source", ""), d.metadata.get("section", ""),
                d.metadata.get("doc_type", ""), d.page_content)
        n = seen[base]
        seen[base] += 1
        ids.append(chunk_id(d, n))
    return ids


def _existing_ids(store: Chroma) -> set:
    """컬렉션에 이미 들어 있는 id 집합. 본문·임베딩은 가져오지 않는다(메모리·시간 절약)."""
    got = store.get(include=[])
    return set(got.get("ids") or [])


def plan(docs: List[Document], existing: set) -> Tuple[List[int], List[str], int]:
    """(추가할 청크 위치, 지울 id, 그대로 둘 개수) — 쓰기 없이 계산만 한다."""
    ids = assign_ids(docs)
    new_set = set(ids)
    add_positions = [i for i, cid in enumerate(ids) if cid not in existing]
    delete_ids = sorted(existing - new_set)
    keep = len(new_set & existing)
    return add_positions, delete_ids, keep


def build_index(reset: bool = False, profile: Optional[CorpusProfile] = None,
                full: bool = False, dry_run: bool = False) -> Dict[str, int]:
    """지식원을 읽어 프로필의 컬렉션을 갱신. 변경 요약을 dict 로 반환."""
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
        index_state.clear(prof.chroma_dir, prof.collection_name)
        store = _store(prof)

    docs = load_and_split() + load_code() + load_git()
    if not docs:
        raise RuntimeError(
            f"인덱싱할 문서가 없습니다(프로필={prof.name}). "
            "demo 라면 git 저장소인지, private 이라면 .env 의 경로를 확인하세요."
        )

    ids = assign_ids(docs)
    existing = set() if (reset or full) else _existing_ids(store)
    add_pos, del_ids, keep = plan(docs, existing)

    n_code = sum(1 for d in docs if d.metadata.get("doc_type") == "code")
    n_commit = sum(1 for d in docs if d.metadata.get("doc_type") == "commit")
    n_doc = len(docs) - n_code - n_commit
    mode = "전량" if (reset or full) else "증분"
    print(f"[ingest] {mode} — 총 {len(docs)}청크(문서 {n_doc} / 코드 {n_code} / 커밋 {n_commit})")
    print(f"[ingest]   추가 {len(add_pos)} · 삭제 {len(del_ids)} · 유지 {keep}"
          + ("  ← 유지분은 임베딩을 다시 계산하지 않는다" if keep else ""))

    summary = {"total": len(docs), "added": len(add_pos), "deleted": len(del_ids), "kept": keep,
               "doc": n_doc, "code": n_code, "commit": n_commit}
    if dry_run:
        print("[ingest] --dry-run — 쓰기 없이 종료")
        return summary
    if not add_pos and not del_ids:
        print("[ingest] 바뀐 것이 없습니다. 그대로 둡니다.")
        return summary

    # 삭제를 먼저 한다. 추가 후에 지우면, 내용이 같아 id 가 겹치는 청크를 방금 넣고
    # 곧바로 지우는 순서가 나올 수 있다.
    for i in range(0, len(del_ids), _DEL_BATCH):
        store.delete(ids=del_ids[i : i + _DEL_BATCH])
    if del_ids:
        print(f"[ingest]   삭제 완료 {len(del_ids)}건")

    if add_pos:
        print(f"[ingest] 임베딩 제공자: {settings.embedding_provider} — {len(add_pos)}청크 계산 중...")
        for i in range(0, len(add_pos), _ADD_BATCH):
            batch = add_pos[i : i + _ADD_BATCH]
            store.add_documents([docs[j] for j in batch], ids=[ids[j] for j in batch])
            print(f"[ingest]   … {min(i + _ADD_BATCH, len(add_pos))}/{len(add_pos)}")

    token = index_state.stamp(prof.chroma_dir, prof.collection_name)
    print(f"[ingest] 완료 — '{prof.collection_name}' (버전 {token})")
    print("[ingest] 돌고 있는 서버는 다음 질문에서 새 인덱스를 집어 든다(재기동 불필요).")
    return summary


def get_vectorstore() -> Chroma:
    """활성 프로필의 인덱스를 로드해 조회용 벡터스토어 반환."""
    return _store(active_profile())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="지식원 → Chroma 인덱싱(기본: 증분)")
    parser.add_argument("--reset", action="store_true", help="이 프로필의 컬렉션 삭제 후 재생성")
    parser.add_argument("--full", action="store_true",
                        help="증분 없이 전량 재색인(기존 청크를 지우지는 않음 — 정합성 의심 시)")
    parser.add_argument("--dry-run", action="store_true", help="무엇이 바뀌는지만 계산(쓰기 없음)")
    parser.add_argument(
        "--profile",
        choices=available_profiles(),
        help="코퍼스 프로필(기본: .env 의 CORPUS_PROFILE)",
    )
    args = parser.parse_args()
    if args.profile:
        use_profile(args.profile)
    build_index(reset=args.reset, full=args.full, dry_run=args.dry_run)
