# stock_prod RAG 챗봇 — 작업 현황 (HANDOFF)

> 목적: `D:\stock_prod\.claude\memory` 문서를 지식원으로, 프로젝트에 대해 질문하면
> **문서 근거 + 출처**와 함께 답하는 RAG 챗봇. (이력서 포트폴리오 겸용)
> 위치: `D:\stock_prod_rag` (stock_prod 와 분리된 독립 repo, git init 완료)
> 최종 작업일: 2026-07-08

---

## 현재 상태: **동작 검증 완료 (MVP 완성)**

파이프라인 전 구간 로컬 실행 통과:
- 문서 로드·분할: 52개 .md → **911 청크** (전 파일 커버)
- 벡터 인덱싱: Chroma 영속화 (`./chroma_db`, 컬렉션 `stock_prod_memory`)
- 검색 정확도: "운용 모드?"→`prod_status.md>E모드`, "zombie recovery?"→`zombie_recovery.md`, "sl 캡?"→`0.3% 캡` 전부 정확
- FastAPI: `/health` `/chat` `/`(웹UI) 정상, UI 마크다운 렌더링 + XSS escape 검증(node)

## 아키텍처

```
.md 문서(memory/**) → LangChain(MarkdownHeader+Recursive 분할) → Chroma(임베딩 영속)
                    → 질문 유사검색 top-k → OpenAI gpt-4o-mini(근거기반 프롬프트) → FastAPI + 웹UI
```

## 파일 구조 (git 추적 12개)
```
app/config.py      pydantic-settings, .env 로드. knowledge_dir_list/glob_list/has_openai 헬퍼
app/loader.py      .md → 헤더 인지 청크. metadata: source(상대경로)/path/section
app/embeddings.py  get_embeddings(): openai(text-embedding-3-small) | hf(로컬 ko-sroberta)
app/ingest.py      build_index(reset) / get_vectorstore(). CLI: python -m app.ingest [--reset]
app/rag.py         answer(q)->{answer,sources,mode}. 근거기반 SYSTEM_PROMPT, extractive 폴백
app/main.py        FastAPI: /health /chat /(UI). 포트는 실행인자로만(하드코딩 X)
web/index.html     단일페이지 챗UI + 자체 마크다운 렌더러(외부CDN 無)
requirements.txt  .env.example  .gitignore  README.md
```
`.venv/`, `chroma_db/`, `.env` 는 gitignore.

## 실행
```powershell
cd D:\stock_prod_rag
.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8090   # 8000은 다른 서버가 점유 → 8090 사용
```
`/` 는 요청마다 index.html 재로드 → UI 수정 시 서버 재시작 없이 F5.

## ⚠️ 현재 .env 상태 (중요)
지금은 **키 없이 동작하도록** 테스트 모드로 설정됨:
- `EMBEDDING_PROVIDER=hf` (로컬 임베딩, torch 설치돼 있음)
- `LLM_PROVIDER=extractive` (LLM 미사용, 검색 발췌만 반환)
- chroma_db 는 **hf 임베딩으로 인덱싱된 상태**.

### OpenAI 로 "진짜 생성 답변" 켜기 (다음 단계 최우선)
`.env` 3줄 변경 후 **반드시 재인덱싱**(임베딩 제공자 바뀌면 벡터 불일치):
```
OPENAI_API_KEY=sk-...
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=openai
```
```powershell
python -m app.ingest --reset
```
→ answer() 의 openai 경로(rag.py)는 아직 실키로 미검증. 키 넣고 첫 검증 필요.

## TODO / 다음 작업
1. **OpenAI 키 발급 → openai 모드로 전환·재인덱싱·답변 품질 검증** (rag.py openai 분기 실측)
2. 포트폴리오용: 아키텍처 구조도(Data Flow Diagram) 1장 + 웹UI 스크린샷
3. GitHub 공개 repo push (현재 로컬 커밋만)
4. (선택) `POST /ingest` 관리 엔드포인트, `docs/` 폴더도 지식원에 추가(KNOWLEDGE_DIRS)
5. 이력서(D_data 경력기술서)에 한 줄 삽입 — README 하단 "이력서용 한 줄" 참고

## 메모
- Windows 콘솔 cp949 → 한글 print 시 mojibake. `PYTHONIOENCODING=utf-8` 로 실행하면 정상.
- curl 로 한글 body 보낼 때 Windows 인코딩 이슈로 400 발생 가능. 웹UI/파이썬 클라이언트는 정상.
- 임베딩 hf↔openai 전환은 **항상 --reset 재인덱싱** 동반해야 함.
