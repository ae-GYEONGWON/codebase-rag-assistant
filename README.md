# stock_prod 문서 RAG 챗봇

KOSPI200 선물·옵션 자동매매 시스템(`stock_prod`)의 **프로젝트 문서(.claude/memory)** 를 지식원으로 삼아,
"이 로직 왜 이렇게 짰지?", "지금 운용 모드가 뭐지?" 같은 질문에 **문서 근거 + 출처와 함께** 답하는 RAG 챗봇.

> **핵심 특징**: 검색된 문서 발췌에만 근거해 답하고(환각 억제, 출처 표기), **토큰 스트리밍** 응답,
> **예시 질문 칩**과 **"이 봇이 아는 것" 지식 패널**로 무엇을 물어볼 수 있는지 안내.
> LLM은 **무료(Gemini)** 로 동작하며, 키가 없으면 검색 발췌만 반환하는 extractive 모드로 자동 폴백.

## 아키텍처

```
 .md 문서(memory/**)      LangChain                 Chroma             FastAPI
 ─────────────────  →  헤더 인지 청크 분할   →   벡터 DB(임베딩)  →  /chat·/chat/stream → 웹 UI
                       + Recursive split        (영속 저장)          근거기반 답변 + 출처
                                                       ↑
                                     질문 임베딩(로컬 무료) → 유사 top-k 검색 → LLM(Gemini) 스트리밍
```

- **수집/분할**: `MarkdownHeaderTextSplitter`(헤더를 메타데이터로 보존) → `RecursiveCharacterTextSplitter`
- **임베딩**: 로컬 무료 `jhgan/ko-sroberta-multitask` (기본) — 비용 0, 오프라인 가능
- **벡터 DB**: Chroma (로컬 영속)
- **생성(LLM)**: **Gemini `gemini-2.5-flash` 무료 티어** (근거 기반 프롬프트, temperature=0). openai 대안 지원.
- **서빙**: FastAPI(+토큰 스트리밍) + 단일 페이지 웹 UI(자체 마크다운 렌더러)

## 빠른 시작

```bash
cd D:/stock_prod_rag
python -m venv .venv
.venv/Scripts/activate            # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env              # Windows cmd: copy .env.example .env
# .env 의 GOOGLE_API_KEY 에 무료 키 입력 → https://aistudio.google.com/apikey

python -m app.ingest --reset              # 문서 → 벡터 DB 인덱싱
uvicorn app.main:app --reload --port 8090 # http://127.0.0.1:8090 접속
```

> **비용**: 임베딩은 로컬(무료), 생성은 Gemini 무료 티어 → **0원**. 결제수단 미등록이면 과금 자체가 불가능.
> **키 없이도** 실행됨 — 그 경우 검색된 문서 발췌만 보여주는 `extractive` 모드로 자동 동작.

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET  | `/health`       | 상태·설정(활성 LLM) 확인 |
| GET  | `/topics`       | 지식원 파일 목록 + 예시 질문 |
| POST | `/chat`         | `{"question":"..."}` → `{answer, sources[], mode}` (단발) |
| POST | `/chat/stream`  | 같은 입력 → NDJSON 토큰 스트림(`sources`/`token`/`done`) |
| GET  | `/`             | 데모 웹 챗 UI |

## 다른 프로젝트에 재사용
`.env` 의 `KNOWLEDGE_DIRS` 만 바꾸면 어떤 마크다운 문서 폴더에도 그대로 붙는다.
```
KNOWLEDGE_DIRS=D:/other_project/docs,D:/other_project/wiki
```
> 임베딩 제공자(hf↔openai)를 바꾸거나 문서가 바뀌면 재인덱싱: `python -m app.ingest --reset`

---

### 이력서용 한 줄
> **LangChain·Chroma(Vector DB) 기반 도메인 특화 RAG 파이프라인 구축 및 FastAPI 스트리밍 챗봇 API 서빙** —
> 마크다운 문서를 헤더 인지 청크로 분할·임베딩하고, 근거 기반 프롬프트로 환각을 억제(출처 표기)하며
> 토큰 스트리밍 UI로 서빙한 문서 질의응답 시스템 구현.
