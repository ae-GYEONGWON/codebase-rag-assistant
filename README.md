# stock_prod 지식 어시스턴트 (RAG)

운영 중인 KOSPI200 선물·옵션 자동매매 시스템(`stock_prod`)의 **문서 · 소스코드 · git 이력**을
지식원으로 삼아, *"지금 운용 모드가 뭐지?"*, *"apply_vix_sl 함수는 뭘 해?"*, *"SL 캡은 언제 왜 없앴어?"*
같은 질문에 **근거 + 출처와 함께** 답하는 RAG 챗봇.

단순 "문서 넣고 질문하는 챗봇"이 아니라, **검색 품질과 답변 충실도를 자체 평가 하네스로 수치화**하고
그 수치로 설계를 결정한 것이 핵심이다. 전 구간 **무료 스택**(Gemini 무료 티어 + 로컬 임베딩)으로 운영비 0원.

---

## 무엇이 다른가 (측정으로 증명)

| 지표 | 벡터 단독 | **하이브리드(본 시스템)** |
|---|---:|---:|
| 검색 recall@5 (전체 28문항) | 75% | **93%** |
| MRR | 0.53 | **0.68** |
| — 문서 질문 20 | 85% | **100%** |
| — 코드 질문 8 | 50% | **75%** |
| 답변 groundedness (환각 없음, LLM-judge) | — | **0.962** |
| 범위 밖 질문 거절률 | — | **83%** (5/6) |

> 지식원: 문서 1,022 + 코드 2,331 + git 커밋 266 = **3,619 청크**.
> 측정 재현: `python -m eval.run_eval` (검색), `python -m eval.faithfulness` (답변 충실도).

구현하며 마주친 문제와 해결 과정 14건은 **[엔지니어링 노트](docs/engineering-notes.md)** 에 정리.

---

## 아키텍처

```
지식원 ─┬ 문서(.md)   → 헤더 인지 분할
        ├ 코드(.py)   → AST 함수/클래스 분할 + '파일>심볼' 컨텍스트 헤더
        └ git 이력    → 커밋 단위(메시지+변경파일+날짜)
            │
            ▼  로컬 임베딩(ko-sroberta, 무료)
        Chroma 벡터 DB (영속) ── 3,619 청크
            │
   질문 ─┬─→ 벡터 검색(코사인)  ─┐
         └─→ BM25(어휘)         ─┴─ RRF 융합 ─→ 심볼 정확매칭 ─→ 범위밖 게이트(cos≥0.35)
                                                                        │
                                              근거(문서+코드 혼합, 불일치 시 코드 기준)
                                                                        ▼
                              Gemini 3.1-flash-lite (근거기반 프롬프트) ─→ FastAPI 토큰 스트리밍 ─→ 웹 UI
```

### 검색 파이프라인 (하이브리드)
- **BM25(어휘) + 벡터(의미)를 RRF 로 융합** — 임베딩은 `RC4025` 같은 희귀 식별자에 약하고,
  BM25 는 표현이 다른 질문에 약하다. 서로의 약점을 메운다.
- **심볼 정확 매칭** — 질문에 코드 심볼명(`apply_vix_sl`)이 있으면 그 함수 본문을 top-k 에 보장.
  한국어 질문 ↔ 영어 코드의 임베딩 유사도 저하를 보완.
- **범위 밖 게이트** — 코사인 유사도 임계(0.35)로 잡담·무관 질문을 검색 단계에서 거절(환각 차단).
- **리랭커** — cross-encoder 를 구현했으나 **측정 결과 이 코퍼스에선 해로워 기본 비활성화**
  (`USE_RERANKER` 로 실험 가능). 자세한 근거는 엔지니어링 노트 참고.

### 생성
- **Gemini `gemini-3.1-flash-lite` 무료 티어**, temperature=0, 근거 기반 프롬프트로 환각 억제.
- **문서와 코드가 어긋나면 명시하고 코드를 기준**으로 답한다(예: 문서엔 남았으나 코드에선 제거된 설정).
- 키가 없으면 검색 발췌만 반환하는 `extractive` 모드로 자동 폴백.

### UX
- 토큰 스트리밍, 출처 각주 `[n]` 클릭 시 원문 스니펫 펼침, 출처에 **DOC/CODE 배지**.
- **개발자 모드 토글** — 끄면(비개발자) 코드를 산문으로 설명, 켜면 실제 코드를 ```python 블록으로 인용.
- **피드백 👍/👎** — 👎 받은 질문은 평가셋 확장 후보로 로그에 축적.

---

## 빠른 시작

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # bash: source .venv/Scripts/activate
pip install -r requirements.txt

copy .env.example .env              # bash: cp .env.example .env
# .env 의 GOOGLE_API_KEY 에 무료 키 입력 → https://aistudio.google.com/apikey

python -m app.ingest --reset               # 지식원(문서+코드+git) → 벡터 DB 인덱싱
uvicorn app.main:app --port 8123           # http://127.0.0.1:8123
```

> **비용 0원**: 임베딩은 로컬, 생성은 Gemini 무료 티어(결제수단 미등록 시 과금 불가).
> 키 없이도 `extractive` 모드로 동작.

### Docker
컨테이너는 **쿼리 서빙 전용**(인덱싱은 지식원이 있는 호스트에서). 호스트에서 `app.ingest` 로 만든
`chroma_db` 를 마운트한다.
```bash
python -m app.ingest --reset       # 먼저 호스트에서 인덱싱
docker compose up --build          # → http://127.0.0.1:8123
```

### 테스트
```bash
python -m pytest                   # 순수 로직 + 검색 통합(chroma 있으면)
```

---

## 설정 (`.env`)

| 키 | 기본값 | 설명 |
|---|---|---|
| `GOOGLE_API_KEY` | (필수) | Gemini 무료 키. 없으면 extractive 폴백 |
| `GEMINI_CHAT_MODEL` | `gemini-3.1-flash-lite` | 생성 모델 |
| `EMBEDDING_PROVIDER` | `hf` | `hf`(로컬 무료) \| `openai` |
| `KNOWLEDGE_DIRS` | `D:/stock_prod/.claude/memory` | 문서 지식원(쉼표 구분) |
| `CODE_DIRS` | `D:/stock_prod/app,...` | 코드 지식원 (`INDEX_CODE=false` 로 끄기) |
| `GIT_REPOS` | `D:/stock_prod` | git 이력 지식원 (`INDEX_GIT=false` 로 끄기) |
| `USE_RERANKER` | `false` | 리랭커(측정상 비활성 권장) |

> **다른 프로젝트 재사용**: `KNOWLEDGE_DIRS`/`CODE_DIRS`/`GIT_REPOS` 만 바꾸면 어떤 코드베이스에도 붙는다.
> 지식원이 바뀌면 재인덱싱(`python -m app.ingest --reset`).

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET  | `/health`      | 상태·활성 LLM 확인 |
| GET  | `/topics`      | 지식원 규모 + 예시 질문 |
| POST | `/chat`        | `{question, dev_mode?}` → `{answer, sources[], mode, retrieval}` |
| POST | `/chat/stream` | 같은 입력 → NDJSON 토큰 스트림(`sources`/`token`/`done`) |
| POST | `/feedback`    | `{question, verdict}` 👍/👎 로그 |
| GET  | `/`            | 데모 웹 챗 UI |

## 프로젝트 구조

```
app/
  loader.py       .md → 헤더 인지 청크
  code_loader.py  .py → AST 함수/클래스 청크 + 컨텍스트 헤더
  git_loader.py   git log → 커밋 청크(날짜 메타)
  embeddings.py   로컬 hf | openai 임베딩
  ingest.py       문서+코드+git 통합 인덱싱 → Chroma
  retriever.py    BM25+벡터 RRF → 심볼매칭 → MMR → 범위밖 게이트
  reranker.py     cross-encoder(기본 off, 측정 근거)
  rag.py          근거기반 생성 + 스트리밍, 문서/코드 혼합 프롬프트
  feedback.py     👍/👎 로그
  main.py         FastAPI 엔드포인트 + 기동 워밍업
eval/
  questions.json  평가셋(문서 20 + 코드 8 + 범위밖 6)
  run_eval.py     검색 평가(recall@k·MRR·거절률, 리트리버별 비교)
  faithfulness.py 답변 groundedness(LLM-as-judge)
tests/            pytest (토크나이저·정규화·config·AST·검색통합)
docs/
  engineering-notes.md  구현 중 마주친 문제 14건
```

---

### 이력서용 한 줄
> **운영 중인 자동매매 시스템의 문서·소스코드·git 이력(3,600+ 청크)을 대상으로 한 RAG 어시스턴트 구축** —
> BM25+벡터 하이브리드 검색(RRF)과 코드 심볼 매칭으로 검색 recall 을 벡터 단독 75%→93% 개선,
> 자체 평가 하네스로 검색 정확도와 답변 충실도(groundedness 0.96)를 수치화. AST 코드 청킹,
> 범위 밖 질문 거절, 문서·코드 불일치 판별, 토큰 스트리밍 서빙까지 무료 스택(Gemini·Chroma·FastAPI)으로 구현.
