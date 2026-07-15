# stock_prod RAG 챗봇 — 작업 현황 (HANDOFF)

> 목적: `D:\stock_prod\.claude\memory` 문서를 지식원으로, 프로젝트에 대해 질문하면
> **문서 근거 + 출처**와 함께 답하는 RAG 챗봇. (이력서 포트폴리오 겸용)
> 위치: `D:\stock_prod_rag` (stock_prod 와 분리된 독립 repo)
> 최종 작업일: 2026-07-15
> 정체성: 단순 '문서 챗봇'이 아니라 **운영 중인 자동매매 시스템의 문서·소스코드를 아는 어시스턴트**.
>   비개발자(대표·동료)도 쓸 수 있게 설계(결론 먼저·근거 스니펫·범위밖 거절). ※ '실사용 중'이라고 과장 금지.

## 포트폴리오 로드맵
- **Tier 1 (체감)**: LLM 연결 + 칩 + 지식패널 + 스트리밍 → **완료·실검증**
- **Tier 2 (정확도)**: 하이브리드검색+MMR, 인용 스니펫, 범위밖 처리 → **완료**
- **코드 인덱싱**: 소스 160파일 AST 청킹 → 문서+코드 통합 → **완료**
- **리랭커**: cross-encoder 구현 + 2모델 측정 → **완료(측정 후 기본 OFF 결정)**
- **Tier 3 (도장)**: 검색 평가 완료. 답변품질(faithfulness)·README·Docker·pytest → TODO

## 다음 작업 (합의 순서)
1. **git 히스토리 인덱싱** — 커밋·diff 로 "언제 왜 바뀌었나" 답변. 최신성 가중치·폐기 인지.
2. **답변 품질 평가**(LLM-as-judge, Gemini 무료) — 지금 지표는 검색만 잰다(faithfulness 미측정)
3. **비개발자 UX** — 코드 인용 접기/개발자 모드 토글, 피드백 버튼(👍/👎)→질문로그→평가셋 확장
4. Docker + pytest + README(다이어그램·측정표·비교리포트) + 면접 대비 문서
   → 사용자가 직접 써보며 RAG/벡터DB 개념 익힌 **후** 문서 작성

## 지식원 문서 정리 관련 (사용자 논의 결론)
- **챗봇용으로 문서를 다듬지 말 것**. 깨끗한 코퍼스는 오히려 경쟁력↓("정리돼야 동작하는 시스템"으로 보임).
- 난이도·경쟁력은 **코드·git 이력**(모순·폐기·시간축 존재)이 만든다. 문서 정리는 본인 개발 편의대로.
- 서류 심사가 보는 건: ①실운영 시스템 연계 ②숫자 ③최신 스택. '코퍼스가 지저분한지'는 안 봄.

## 현재 상태: 문서+코드 하이브리드 검색 + Gemini 생성 (실검증 완료)

## 실행
```powershell
cd D:\stock_prod_rag
.venv\Scripts\Activate.ps1
uvicorn app.main:app --port 8123          # 8090/8091 은 과거 좀비서버가 점유했던 이력
python -m app.ingest --reset              # 지식원 바뀌면 재인덱싱 (서버 끄고)
python -m eval.run_eval                   # 검색 평가 (recall@5 · MRR · 범위밖 거절)
```

## 아키텍처
```
지식원 ─┬ .md 문서(memory/**) → 헤더 인지 분할
        └ .py 소스(app/,scripts/) → AST 함수·클래스 분할 + '파일>심볼' 컨텍스트 헤더
      → Chroma(로컬 hf 임베딩, 무료) : 총 3342 청크(문서 1011 / 코드 2331)
   질문 ─┬→ 벡터 검색(코사인)  ─┐
         └→ BM25(어휘)        ─┴→ RRF 융합 → MMR(λ=0.8) → 범위밖 게이트(cos≥0.35)
              └(리랭커는 토글 OFF: 측정상 코드질문에서 해로움)
   → Gemini(gemini-3.1-flash-lite, 무료) 근거기반 프롬프트(문서/코드 혼합, 불일치시 코드 기준)
   → FastAPI 토큰 스트리밍 + 웹UI(출처 DOC/CODE 배지·스니펫 펼침)
```

## 파일 구조
```
app/config.py      설정. active_llm(키 유무로 provider 자동결정), 검색 파라미터
app/loader.py      .md → 헤더 인지 청크(doc_type=doc). list_sources()
app/code_loader.py ★.py → AST 함수/클래스 청크(doc_type=code) + '파일>심볼' 헤더
app/embeddings.py  hf(로컬 ko-sroberta, 무료) | openai. get_embeddings() lru_cache
app/ingest.py      문서+코드 통합 인덱싱(2000개씩 배치 add)
app/retriever.py   BM25+벡터 RRF 융합 → 리랭커(옵션) → MMR → 범위밖 게이트 → 스니펫
app/reranker.py    ★cross-encoder(bge-reranker-base). 기본 OFF(측정상 코드질문에 해로움)
app/rag.py         answer() / stream_answer(). _text_of()=Gemini 파트배열 정규화. 코드/문서 혼합 프롬프트
app/main.py        FastAPI: /health /chat /chat/stream /topics /(UI). lifespan 워밍업
web/index.html     챗UI + 자체 마크다운 렌더러 + 출처 DOC/CODE 배지·스니펫 펼침
eval/questions.json  문서 20문항 + 코드 8문항 + 범위밖 6문항 (정답=실제 구현 파일)
eval/run_eval.py     vector/bm25/RRF/MMR 분리 측정. --rerank 로 리랭커 비교행
```

## 측정 결과 (k=5, 2026-07-15) — 포트폴리오 핵심 숫자
문서+코드 통합 코퍼스(3342 청크) 기준.
| retriever | 전체 recall@5 | MRR | (문서20 / 코드8) |
|---|---:|---:|---|
| vector only | 75% | 0.57 | 85% / 50% |
| bm25 only | 75% | 0.53 | 85% / 50% |
| **hybrid(RRF)+MMR** | **93%** | **0.70** | **100% / 75%** |
| hybrid+rerank(base) | ↓ | ↓ | 95% / 50% (해로움→OFF) |

범위 밖 거절 5/6 (83%). "삼성전자 주가"(cos 0.425)만 통과 → LLM 프롬프트가 거절.

## 설계 결정과 근거 (면접 대비: 전부 실측 기반)
- **모델 = gemini-3.1-flash-lite**: 2.5-flash 는 신규 키에서 404. 3.5-flash 는 thinking 탓 첫 토큰 ~20s
  → 데모는 지연이 곧 품질이라 flash-lite(1.4s).
- **Gemini 3.x content 는 파트 배열** → `_text_of()` 정규화 필수.
- **코드 청킹은 AST**(고정길이면 함수가 허리에서 잘림). 청크마다 '파일>심볼' 헤더로 위치정보 보존
  (Contextual Retrieval 경량판). 프롬프트는 **문서/코드 불일치 시 코드를 기준**으로 답하게 지시.
- **MMR 적합도 항 = RRF 융합 점수**(코사인 아님). 코사인이면 BM25 가 올린 희귀 식별자
  문서(RC4025)가 탈락해 하이브리드가 무의미. ← 실제로 잡은 버그.
- **범위밖 게이트는 코사인 단독**(0.35). BM25 는 한글 2-gram 이 흔한 음절에 걸려 게이트 부적격.
- **mmr_lambda=0.8**: λ 스윕 결과(다양성↑, recall 유지).
- **리랭커 기본 OFF**: base/v2-m3 둘 다 측정 → 코드 질문에서 하이브리드보다 나쁨.
  cross-encoder 가 자연어-자연어를 자연어-코드보다 선호해 "코드에서" 의도를 뒤집음. v2-m3 는 CPU 4.4s 지연도.
  → 무지성 SOTA 적용이 아니라 **측정 기반 비활성화**(구현·토글은 유지). ★좋은 면접 포인트.
- **get_embeddings() lru_cache**: 없을 때 질문마다 hf 모델 재로드 → 평가 타임아웃이던 버그.

## 남은 코드질문 미스 2건 (다음 개선 후보)
"SL 은 코드에서 어떻게 계산돼?"(sl_table.py/vix_strategy.py 못 뽑음), "웹 API 라우터".
원인: 질문·문서가 둘 다 한국어 자연어라 코드보다 표면 유사도 높음("코드에서" 의도가 안 살음).
리랭커로도 안 고쳐짐 → 의도 라우팅(doc_type 부스트) 또는 git 이력 축이 필요. 93%면 현재 충분.

## 알려진 한계 (면접에서 먼저 인정할 것)
- 답변 품질(faithfulness) 미측정 / 평가셋 자가 라벨(28문항)
- Chroma 로컬 영속화 → 서비스형 벡터DB(pgvector·Qdrant) 운영 경험 아님
- 한글 토크나이저가 2-gram 근사(형태소 분석기 미사용)
- 증분 재인덱싱 없음(변경분만 갱신 X, 전체 --reset) / 테스트·Docker·CI 없음

## 메모
- Windows 콘솔 cp949 → 한글 print mojibake. `PYTHONIOENCODING=utf-8` 로 실행.
- **좀비 서버 주의**: 죽인 uvicorn 의 자식 프로세스가 포트·chroma 파일을 계속 물고 있을 수 있다.
  `Get-CimInstance Win32_Process` 로 확인 후 정리. stock_prod(8000)·stock_finder(8848) 는 실매매라 절대 건드리지 말 것.
- 임베딩 hf↔openai 전환은 항상 `--reset` 재인덱싱. LLM provider 만 바꾸는 건 불필요.
