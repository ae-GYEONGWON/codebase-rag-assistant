# stock_prod RAG 챗봇 — 작업 현황 (HANDOFF)

> 목적: `D:\stock_prod\.claude\memory` 문서를 지식원으로, 프로젝트에 대해 질문하면
> **문서 근거 + 출처**와 함께 답하는 RAG 챗봇. (이력서 포트폴리오 겸용)
> 위치: `D:\stock_prod_rag` (stock_prod 와 분리된 독립 repo)
> 최종 작업일: 2026-07-14

## 포트폴리오 로드맵
- **Tier 1 (체감)**: LLM 연결 + 예시질문 칩 + 지식 패널 + 스트리밍 → **완료·실검증**
- **Tier 2 (정확도)**: 하이브리드검색+MMR, 인용 스니펫, 범위 밖 처리 → **완료**
  (단 **리랭커 미구현** — 면접에서 "왜 안 썼냐" 나올 구멍. 다음 작업 1순위)
- **Tier 3 (도장)**: 평가 하네스 → **검색 부분 완료**. 답변품질(faithfulness)·README·Docker·pytest → TODO

## 다음 작업 (사용자와 합의한 순서)
0. **(사용자 진행 중) 지식원 문서 정리** → 끝나면 반드시 `python -m app.ingest --reset`
   ⚠ 서버가 떠 있으면 chroma_db 파일이 잠겨 재인덱싱이 실패한다. 서버 먼저 종료.
1. **리랭커(cross-encoder)** 도입 → eval 로 개선폭 측정(효과 없으면 정직하게 끄고 기록)
2. **답변 품질 평가**(LLM-as-judge, Gemini 무료) — 지금 지표는 검색만 잰다
3. Docker + pytest + README(아키텍처 다이어그램·측정표) + 면접 대비 문서
   → 사용자가 직접 써보며 RAG/벡터DB 개념 익힌 **후** 문서 작성

## 현재 상태: Gemini 생성 답변 + 하이브리드 검색 동작 (실검증 완료)

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
.md 문서(memory/**) → 헤더 인지 분할(956 청크) → Chroma(로컬 hf 임베딩, 무료)
   질문 ─┬→ 벡터 검색(코사인)  ─┐
         └→ BM25(어휘)        ─┴→ RRF 융합 → MMR(λ=0.8) → 범위밖 게이트(cos≥0.35)
   → Gemini(gemini-3.1-flash-lite, 무료) 근거기반 프롬프트 → FastAPI 토큰 스트리밍 + 웹UI
```

## 파일 구조
```
app/config.py      설정. active_llm(키 유무로 provider 자동결정), 검색 파라미터
app/loader.py      .md → 헤더 인지 청크. list_sources()
app/embeddings.py  hf(로컬 ko-sroberta, 무료) | openai
app/ingest.py      build_index(reset) / get_vectorstore()
app/retriever.py   ★신설: BM25+벡터 RRF 융합 → MMR → 범위밖 게이트 → 스니펫
app/rag.py         answer() / stream_answer(). _text_of() = Gemini 파트배열 정규화
app/main.py        FastAPI: /health /chat /chat/stream /topics /(UI). lifespan 워밍업
web/index.html     챗UI + 자체 마크다운 렌더러 + 출처 클릭시 스니펫 펼침
eval/questions.json  정답 출처 라벨 20문항 + 범위밖 6문항
eval/run_eval.py     vector / bm25 / RRF / RRF+MMR 분리 측정
```

## 측정 결과 (k=5, 2026-07-14) — 포트폴리오 핵심 숫자
| retriever | recall@5 | MRR |
|---|---:|---:|
| vector only | 90% | 0.74 |
| bm25 only | 85% | 0.68 |
| **hybrid(RRF)+MMR** | **95%** | **0.77** |

범위 밖 거절 5/6 (83%). "삼성전자 주가"(cos 0.425)만 통과 → LLM 프롬프트가 거절.

## 설계 결정과 근거 (면접 대비: 전부 실측 기반)
- **모델 = gemini-3.1-flash-lite**: 2.5-flash 는 신규 키에서 404(no longer available to new users).
  3.5-flash 는 품질↑지만 thinking 탓 첫 토큰 ~20s → 데모는 지연이 곧 품질이라 flash-lite(1.4s).
- **Gemini 3.x content 는 파트 배열**(`[{"type":"text",...}]`) → `_text_of()` 정규화 필수.
- **MMR 적합도 항 = RRF 융합 점수**(코사인 아님). 코사인을 쓰면 BM25 가 끌어올린 희귀 식별자
  문서(RC4025 등)가 최종 선택에서 탈락해 하이브리드가 무의미해짐. ← 실제로 잡은 버그.
- **범위밖 게이트는 코사인 단독**(0.35). BM25 는 한글 2-gram 이 흔한 음절에 걸려
  "고양이 키우는 법"에도 16점 → 게이트 부적격(코사인은 0.25). 범위안 0.40~0.72 / 밖 0.25~0.29.
- **mmr_lambda=0.8**: λ 스윕에서 0.5~0.7 은 recall 90%, 0.8~0.9 는 95%(RRF 단독과 동일)+다양성↑.
- **워밍업(lifespan)**: 첫 질문 18.6s → 3.5s. 원인은 LLM 이 아니라 HF 임베딩 지연 로딩이었음.
- **eval 미스 1건(m9s5)**: 본문은 `composite m=9 / s=5`, `m9s5` 는 파일명에만 존재.
  → 슬롯 보장 같은 특수 규칙은 **의도적으로 넣지 않음**(벤치마크 과적합). 라벨 현실화로 처리 예정.

## 알려진 한계 (면접에서 먼저 인정할 것)
- 리랭커 없음 / 답변 품질(faithfulness) 미측정 / 평가셋 20문항 자가 라벨
- Chroma 로컬 영속화 → 서비스형 벡터DB(pgvector·Qdrant) 운영 경험 아님
- 한글 토크나이저가 2-gram 근사(형태소 분석기 미사용)
- 테스트·Docker·CI 없음

## 메모
- Windows 콘솔 cp949 → 한글 print mojibake. `PYTHONIOENCODING=utf-8` 로 실행.
- **좀비 서버 주의**: 죽인 uvicorn 의 자식 프로세스가 포트·chroma 파일을 계속 물고 있을 수 있다.
  `Get-CimInstance Win32_Process` 로 확인 후 정리. stock_prod(8000)·stock_finder(8848) 는 실매매라 절대 건드리지 말 것.
- 임베딩 hf↔openai 전환은 항상 `--reset` 재인덱싱. LLM provider 만 바꾸는 건 불필요.
