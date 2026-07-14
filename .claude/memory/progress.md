# stock_prod RAG 챗봇 — 작업 현황 (HANDOFF)

> 목적: `D:\stock_prod\.claude\memory` 문서를 지식원으로, 프로젝트에 대해 질문하면
> **문서 근거 + 출처**와 함께 답하는 RAG 챗봇. (이력서 포트폴리오 겸용)
> 위치: `D:\stock_prod_rag` (stock_prod 와 분리된 독립 repo, git init 완료)
> 최종 작업일: 2026-07-14

## 포트폴리오 고도화 로드맵 (사용자와 합의)
완성도 3대 문제 = ①질문 스코프 불명 ②답 품질 ③가독성. 뿌리는 extractive(LLM 미연결).
- **Tier 1 (체감)**: LLM 연결 + 예시질문 칩 + 지식 패널 + 스트리밍 → **구현 완료(2026-07-14)**
- **Tier 2 (정확도)**: 하이브리드검색+리랭커, 인용/스니펫 UX, 범위 밖 처리 → TODO
- **Tier 3 (도장)**: 평가 하네스(지표), README 다이어그램/스크린샷, Docker/테스트 → TODO
- **LLM 결정**: 벤더 무관 → **무료 Gemini 티어** 채택(비용 0). 임베딩은 로컬 hf(무료) 유지.

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
- `EMBEDDING_PROVIDER=hf` (로컬 무료 임베딩, torch 설치돼 있음). chroma_db 는 hf 로 인덱싱됨.
- `LLM_PROVIDER=gemini` 이지만 `GOOGLE_API_KEY` 가 비어 있어 **active_llm=extractive 로 자동 폴백** 중.
- config 의 `active_llm` 프로퍼티가 키 유무로 provider 자동 결정(gemini→키없으면 extractive).

### Gemini 로 "진짜 생성 답변" 켜기 (사용자 액션 필요)
1. https://aistudio.google.com/apikey 에서 **무료 키** 발급(결제수단 미등록이면 과금 불가 = 0원).
2. `.env` 의 `GOOGLE_API_KEY=` 뒤에 붙여넣기. (임베딩은 그대로 hf → **재인덱싱 불필요**)
3. 서버 재시작. `/health` 의 `llm_provider` 가 `gemini` 로 바뀌면 성공. 스트리밍 토큰이 여러 개로 쪼개짐.
→ Gemini 실호출 경로(rag `_llm()` gemini 분기, `gemini-2.5-flash`)는 **키 없어 아직 실측 미완**. 키 넣고 첫 검증 필요.
   (2.5-flash 오류 시 `.env` GEMINI_CHAT_MODEL=gemini-2.0-flash 로 교체)

## Tier 1 구현 내역 (2026-07-14, 커밋됨)
- rag.py: provider 스위치(gemini|openai|extractive), `answer()` + `stream_answer()`(이벤트 제너레이터), 가독성·[n]인용 프롬프트
- main.py: `POST /chat/stream`(NDJSON: sources→token…→done), `GET /topics`(파일목록+예시질문), /health active_llm 표기
- loader.py: `list_sources()` 추가
- web/index.html: 예시질문 칩, "이 봇이 아는 것" 지식패널, **토큰 스트리밍 수신**(fetch reader + 실시간 마크다운 렌더)
- config.py: google_api_key, gemini_chat_model, `has_gemini`/`active_llm`
- 검증: /health·/topics·/chat/stream 전부 extractive 폴백으로 정상(파이썬 클라 테스트). Gemini 실키만 미검증.

## TODO / 다음 작업
1. (사용자) Gemini 무료 키 넣고 gemini 모드 실검증.
2. **Tier 2**: 하이브리드검색(BM25+벡터)+MMR+리랭커 / 출처 클릭시 원문 스니펫·[n]각주 링크 / 범위 밖(유사도 임계) 처리.
3. **Tier 3**: 평가 하네스(질문셋+기대출처 → retrieval@k·정답률 스크립트), README 아키텍처 다이어그램+데모 GIF, Dockerfile, pytest.
4. GitHub 공개 repo push (현재 로컬 커밋만).
5. 지식원 문서가 52→54개로 늘어난 상태 → 최신 반영하려면 `python -m app.ingest --reset`.
6. 이력서(D_data 경력기술서)에 한 줄 삽입 — README 하단 "이력서용 한 줄" 참고.

## 메모
- Windows 콘솔 cp949 → 한글 print 시 mojibake. `PYTHONIOENCODING=utf-8` 로 실행하면 정상.
- curl 로 한글 body 보낼 때 Windows 인코딩 이슈로 400 발생 가능. 웹UI/파이썬 클라이언트는 정상.
- 임베딩 hf↔openai 전환은 **항상 --reset 재인덱싱** 동반. (LLM provider 만 바꾸는 건 재인덱싱 불필요)
