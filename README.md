# stock_prod 문서 RAG 챗봇

KOSPI200 선물·옵션 자동매매 시스템(`stock_prod`)의 **프로젝트 문서(.claude/memory)** 를 지식원으로 삼아,
"이 로직 왜 이렇게 짰지?", "지금 운용 모드가 뭐지?" 같은 질문에 **문서 근거 + 출처와 함께** 답하는 RAG 챗봇.

> **핵심 특징**: 검색된 문서 발췌에만 근거해 답하고, 근거가 없으면 "문서에서 찾을 수 없다"고 답함(환각 억제).
> 답변마다 근거가 된 `파일 > 섹션` 출처를 표기.

## 아키텍처

```
 .md 문서(52개)          LangChain                Chroma            FastAPI
 ─────────────   →   헤더 인지 청크 분할   →   벡터 DB(임베딩)  →   /chat  →  웹 UI
 (memory/**)         + Recursive split       (영속 저장)         근거기반 답변 + 출처
                                                      ↑
                                          질문 임베딩 → 유사 top-k 검색 → LLM(gpt-4o-mini)
```

- **수집/분할**: `MarkdownHeaderTextSplitter`(헤더를 메타데이터로 보존) → `RecursiveCharacterTextSplitter`
- **임베딩**: OpenAI `text-embedding-3-small` (또는 로컬 무료 `jhgan/ko-sroberta-multitask`)
- **벡터 DB**: Chroma (로컬 영속)
- **생성**: OpenAI `gpt-4o-mini`, 근거 기반 프롬프트(temperature=0)
- **서빙**: FastAPI + 단일 페이지 웹 UI

## 빠른 시작

```bash
cd D:/stock_prod_rag
python -m venv .venv
.venv/Scripts/activate            # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env            # bash: cp .env.example .env
# .env 에 OPENAI_API_KEY 입력 (없으면 아래 '키 없이' 참고)

python -m app.ingest --reset      # 문서 → 벡터 DB 인덱싱
uvicorn app.main:app --reload     # http://127.0.0.1:8000 접속
```

### 키 없이 검색만 데모 (선택)
`.env` 에서:
```
EMBEDDING_PROVIDER=hf     # 로컬 임베딩 (pip install langchain-huggingface sentence-transformers)
LLM_PROVIDER=extractive   # LLM 없이 검색된 문서 발췌만 반환
```

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET  | `/health` | 상태·설정 확인 |
| POST | `/chat`   | `{"question": "..."}` → `{answer, sources[], mode}` |
| GET  | `/`       | 데모 웹 챗 UI |

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"현재 운용 모드가 뭐야?\"}"
```

## 다른 프로젝트에 재사용
`.env` 의 `KNOWLEDGE_DIRS` 만 바꾸면 어떤 마크다운 문서 폴더에도 그대로 붙는다.
```
KNOWLEDGE_DIRS=D:/other_project/docs,D:/other_project/wiki
```

---

### 이력서용 한 줄
> **LangChain·Chroma(Vector DB)를 활용한 도메인 특화 RAG 파이프라인 구축 및 FastAPI 기반 챗봇 API 서빙** —
> 마크다운 문서를 헤더 인지 청크로 분할·임베딩하고, 근거 기반 프롬프트로 환각을 억제(출처 표기)한 문서 질의응답 시스템 구현.
