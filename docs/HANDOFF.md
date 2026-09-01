# HANDOFF — 작업 현황과 다음 할 일

> 여러 PC 에서 이어서 개발하기 위한 **공유 핸드오프**. 저장소에 함께 커밋된다.
> (로컬 전용 메모는 `.claude/` 에 두고 git 에 올리지 않는다.)
>
> 최종 갱신: 2026-09-01

---

## 1. 이 프로젝트가 무엇인가

대상 코드베이스의 **문서 · 소스코드 · git 이력** 세 축을 지식원으로, 질문에 **근거와 출처를 붙여**
답하는 RAG 어시스턴트. 단순 문서 챗봇이 아니라 "운영 중인 시스템을 아는 어시스턴트"를 노린다.

설계·측정의 자세한 근거는 [`docs/engineering-notes.md`](engineering-notes.md) 에 16건으로 정리돼 있다.
**이 저장소에서 가장 먼저 읽을 문서다.**

## 2. 개발 환경 세팅 (새 PC 에서)

```bash
git clone https://github.com/ae-GYEONGWON/codebase-rag-assistant.git
cd codebase-rag-assistant
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt

cp .env.example .env          # GOOGLE_API_KEY 만 채우면 된다
python -m app.ingest          # demo 프로필로 인덱싱 (약 220청크, 1~2분)
python -m eval.run_eval       # 검색 평가
uvicorn app.main:app --port 8123
```

`.env` 의 기본값이 `CORPUS_PROFILE=demo` 라 **clone 직후 아무 경로 설정 없이 동작한다.**
`chroma_db/` 는 git 에 없다 — 각 PC 에서 인덱싱해 만든다(같은 코퍼스라 결과는 동일).

> Windows 콘솔은 cp949 라 한글 출력이 깨진다. `PYTHONIOENCODING=utf-8` 를 붙여 실행할 것.

## 3. 코퍼스 프로필

"무엇을 인덱싱하는가"는 `app/profiles.py` 의 프로필이 결정한다. 프로필 = (지식원 · 컬렉션 · 평가셋) 한 벌.

| 프로필 | 지식원 | 컬렉션 | 평가셋 | 용도 |
|---|---|---|---|---|
| `demo` (기본) | **이 저장소 자기 자신** (`git ls-files`) | `corpus_demo` | `eval/questions.demo.json` (추적) | 공개 데모 · CI · 다른 PC |
| `private` | `.env` 의 `KNOWLEDGE_DIRS`/`CODE_DIRS`/`GIT_REPOS` | `.env` 의 `COLLECTION_NAME` | `eval/questions.json` (git 제외) | 규모 있는 실제 코드베이스 측정 |

```bash
python -m app.ingest   --profile demo --reset
python -m eval.run_eval --profile private
```

두 프로필은 **같은 `chroma_db/` 안에서 컬렉션 이름으로 분리**되어 공존한다.
`--reset` 은 해당 프로필의 컬렉션만 지운다(디렉터리를 지우지 않는다).

demo 를 `git ls-files` 기반으로 정의한 이유는 노트 #16 참조 — **측정 재현성**이 목적이고
유출 방지는 부산물이다.

새 축(로그·티켓 등)을 붙이려면 `app/profiles.py` 에 `@register` 함수 하나만 추가하면 된다.
로더·인덱서·평가는 프로필만 보므로 손댈 곳이 없다.

## 4. 현재 측정치

### private 프로필 (N=3,619 청크 · k=5)

| retriever | recall@5 | MRR |
|---|---:|---:|
| vector only | 75% | 0.57 |
| bm25 only | 71% | 0.49 |
| **hybrid(RRF) + MMR** | **93%** | **0.68** |
| hybrid + reranker | ↓ (측정 후 기본 OFF) | ↓ |

범위 밖 거절 5/6 · 답변 groundedness(LLM-as-judge) 0.962.

### demo 프로필 (N=222 청크)

인덱싱만 검증됨. 평가셋 `questions.demo.json` 은 **아직 없다** → 아래 Phase 0-2 에서 만든다.

### 검색 지연 (노트 #15)

벡터 전수매칭 0.12 ms · BM25 4.7 ms · LLM 첫 토큰 ~1,400 ms.
ANN 전환 임계는 **N ≈ 10만** (지연이 아니라 메모리가 먼저 무너진다).

## 5. 로드맵

### Phase 0 — 평가 신뢰도 (진행 중)

지금 숫자들의 최대 약점은 **평가셋이 자가 라벨(40문항)이고 groundedness 가 자기 채점**이라는 것.
자가 채점된 자로 잰 값은 그 자체로는 근거가 약하다. 자를 먼저 고친다.

- [x] **0-4** 노트 #15 — 브루트포스 전수매칭은 의도된 선택 + ANN 전환 임계 측정
- [x] **0-5** 코퍼스 프로필 분리 (demo/private) — 노트 #16
- [ ] **0-1** CI 회귀 게이트 — eval 결과를 파일·아티팩트로 남기고 GitHub Actions 에서 실행
- [ ] **0-2** 합성 평가셋 200~500 문항 생성 + 수동 검수 서브셋 + **합성 vs 수동 라벨 불일치율**
      · RAGAS 병행 대조 (`eval/faithfulness.py` 는 RAGAS faithfulness 의 수기 재구현이다)
- [ ] **0-3** cross-judge 도입 — 생성 모델과 판정 모델을 분리해 **self vs cross 편차 수치화**
      (self-enhancement bias 의 표준 완화책)

### Phase 1 — 규모와 저장소

- [ ] 1-2 확장된 평가셋으로 재측정 → **하락분 원인 규명** (하락이 곧 개선 재료다)
- [ ] 1-3 pgvector 이전 + exact vs HNSW(`ef_search`) 스윕 → 노트 #15 의 임계 N 실증
- [ ] 1-4 증분 인덱싱 (변경분만 갱신, 현재는 전체 `--reset`)
- [ ] 1-5 한글 형태소 분석기(kiwi) 도입 전/후 BM25 recall 비교 (현재는 문자 2-gram 근사)

### Phase 2 — 제품화

- [ ] 2-1 **멀티턴 + 질의 재작성** (현재 단발 질의만 — 공개 데모의 최대 결함)
- [ ] 2-2 라우터: 단발 RAG vs 에이전트 자동 선택 (에이전트는 한 질문에 LLM 3~5회, 지연 ~10초)
- [ ] 2-3 Terraform + ECS + RDS 배포
- [ ] 2-4 관측 대시보드 + 비용 알람
- [ ] 2-5 웹 UI 개편

## 6. 알려진 한계 (먼저 인정할 것)

- 평가셋이 자가 라벨 40문항 — Phase 0-2 가 해결 대상
- groundedness 0.962 는 생성·판정이 같은 모델(self-judge) — Phase 0-3 가 해결 대상
- Chroma 를 blob 저장소로 쓰는 브루트포스 검색 — **의도된 선택이며 근거는 노트 #15**
- 한글 토크나이저가 형태소 분석기 없이 문자 2-gram 근사
- 증분 재인덱싱 없음 / 재인덱싱하려면 서버를 내려야 함(인덱싱·서빙 결합)
- 멀티턴 대화 없음

## 7. 함정 메모

- **좀비 서버**: 종료한 uvicorn 의 자식 프로세스가 포트와 `chroma_db` 파일을 계속 물고 있을 수 있다.
  포트 충돌·DB 잠금이 나면 프로세스를 먼저 확인할 것.
- 임베딩 제공자(`hf` ↔ `openai`)를 바꾸면 **반드시 `--reset` 재인덱싱** (차원·공간이 다르다).
  LLM provider 만 바꾸는 것은 재인덱싱이 필요 없다.
- 하이퍼파라미터(`mmr_lambda` 등)는 코퍼스에 종속된다. 코퍼스가 바뀌면 재측정할 것 (노트 #13).
