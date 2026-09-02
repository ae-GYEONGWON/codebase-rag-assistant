"""증분 인덱싱의 집합 연산 + 인덱스 버전 도장 테스트.

임베딩·Chroma 없이 순수 로직만 고정한다. 여기서 틀리면 **말없이 데이터가 사라지거나**
(지워야 할 것을 안 지움 / 안 지워야 할 것을 지움) 임베딩을 헛돈다.
"""
from langchain_core.documents import Document

from app import index_state
from app.ingest import assign_ids, chunk_id, plan


def _doc(text, source="a.md", section="", doc_type="doc"):
    return Document(page_content=text,
                    metadata={"source": source, "section": section, "doc_type": doc_type})


# --- 내용 주소 ---------------------------------------------------------------

def test_같은_내용이면_같은_id():
    assert chunk_id(_doc("본문")) == chunk_id(_doc("본문"))


def test_내용이_바뀌면_id가_바뀐다():
    assert chunk_id(_doc("본문")) != chunk_id(_doc("본문 수정"))


def test_출처가_다르면_다른_id():
    # 같은 문장이 두 파일에 있을 때 하나로 합쳐지면 출처 하나가 사라진다.
    assert chunk_id(_doc("공통 문장", source="a.md")) != chunk_id(_doc("공통 문장", source="b.md"))


def test_섹션이_다르면_다른_id():
    assert chunk_id(_doc("본문", section="§1")) != chunk_id(_doc("본문", section="§2"))


def test_한_파일에_완전히_같은_청크가_둘이면_구분된다():
    # ordinal 이 없으면 두 청크가 한 id 로 접히고, 그러면 삭제 판정이 어긋난다.
    ids = assign_ids([_doc("반복되는 상용구"), _doc("반복되는 상용구")])
    assert len(set(ids)) == 2


# --- 변경 계획 ---------------------------------------------------------------

def test_처음이면_전부_추가():
    docs = [_doc("가"), _doc("나")]
    add, delete, keep = plan(docs, existing=set())
    assert (len(add), delete, keep) == (2, [], 0)


def test_바뀐_것이_없으면_아무것도_안_한다():
    docs = [_doc("가"), _doc("나")]
    add, delete, keep = plan(docs, existing=set(assign_ids(docs)))
    assert (add, delete, keep) == ([], [], 2)


def test_한_청크만_고치면_그것만_추가하고_옛것만_지운다():
    before = [_doc("가"), _doc("나"), _doc("다")]
    existing = set(assign_ids(before))
    after = [_doc("가"), _doc("나 수정"), _doc("다")]
    add, delete, keep = plan(after, existing)
    assert keep == 2                      # 손대지 않은 두 청크는 임베딩을 다시 안 한다
    assert len(add) == 1 and len(delete) == 1
    assert after[add[0]].page_content == "나 수정"
    assert delete[0] == chunk_id(_doc("나"))


def test_파일이_사라지면_그_청크가_삭제된다():
    before = [_doc("가", source="a.md"), _doc("나", source="b.md")]
    existing = set(assign_ids(before))
    add, delete, keep = plan([_doc("가", source="a.md")], existing)
    assert (add, keep) == ([], 1)
    assert delete == [chunk_id(_doc("나", source="b.md"))]


def test_청크_순서가_바뀌어도_재색인하지_않는다():
    # 파일 안에서 문단 순서만 바뀐 경우. 내용 주소라 순서에 영향받지 않아야 한다.
    docs = [_doc("가"), _doc("나")]
    existing = set(assign_ids(docs))
    add, delete, keep = plan([_doc("나"), _doc("가")], existing)
    assert (add, delete, keep) == ([], [], 2)


# --- 버전 도장 ---------------------------------------------------------------

def test_도장을_찍기_전에는_빈_버전(tmp_path):
    assert index_state.version(str(tmp_path), "c1") == ""


def test_도장을_찍으면_버전이_바뀐다(tmp_path):
    v1 = index_state.stamp(str(tmp_path), "c1")
    assert index_state.version(str(tmp_path), "c1") == v1
    v2 = index_state.stamp(str(tmp_path), "c1")
    assert v2 != v1 and index_state.version(str(tmp_path), "c1") == v2


def test_컬렉션마다_버전이_분리된다(tmp_path):
    # 프로필들이 같은 chroma_dir 를 공유하므로, 한쪽 재색인이 다른 쪽 캐시를 버리면 안 된다.
    index_state.stamp(str(tmp_path), "c1")
    assert index_state.version(str(tmp_path), "c2") == ""


def test_clear는_없어도_조용히_넘어간다(tmp_path):
    index_state.clear(str(tmp_path), "없는컬렉션")     # 예외가 나면 실패
    index_state.stamp(str(tmp_path), "c1")
    index_state.clear(str(tmp_path), "c1")
    assert index_state.version(str(tmp_path), "c1") == ""
