# HANDOFF — 작업 현황과 다음 할 일

> 여러 PC 에서 이어서 개발하기 위한 **공유 핸드오프**. 저장소에 함께 커밋된다.
> (로컬 전용 메모는 `.claude/` 에 두고 git 에 올리지 않는다.)
>
> 최종 갱신: 2026-09-02

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
| `demo` (기본) | **이 저장소 자기 자신** (`git ls-files`, 워킹트리) | `corpus_demo` | `eval/questions.demo.json` (추적) | 공개 데모 · 다른 PC |
| `eval` | 저장소 스냅샷 **@ `eval-corpus-v1`** (`git archive`) | `corpus_eval_…` | 같음 | **CI 회귀 게이트** |
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

### 합성 평가셋 측정 (258문항 · 같은 고정 코퍼스)

| 평가셋 | 문항 | 전체 recall@5 | 문서 | 코드 | 커밋 |
|---|---:|---:|---:|---:|---:|
| 수기 골든셋 | 20 | 85% | 92% | 75% | *(문항 없음)* |
| **합성셋** | **258** | **79%** | 82% | 82% | **69%** |

어휘 중복률 중앙값 0.16(베껴 쓴 질문 0건). 라벨 출처 `synthetic` 258.
**거짓 오답 감사 결과**(miss 53건 중 무작위 20건, 생성기와 다른 모델로 판정):
그중 **14건(70%)이 거짓 오답** — 검색기가 옳게 찾았는데 라벨이 좁아서 틀린 걸로 채점됐다.
전체로 외삽하면 거짓 오답률 14.4% → **보정 recall 93.8%**.

⚠️ **단일 숫자로 인용하지 말 것.** 같은 감사를 다른 판정 모델(`gemini-3.5-flash`)로 돌렸을 때는
거짓 오답 비율이 23.5% 로 나왔다(보정 ~84%). **판정기를 바꿨을 뿐인데 10%p 가 벌어진다.**

| | 값 | 뜻 |
|---|---:|---|
| 79.5% | 하한 | 라벨이 좁아 옳은 검색도 틀렸다고 침 |
| 93.8% | 상한 | 판정기가 관대할 수 있음 |

→ **구간으로 보고한다.** 동일 표본 통제 비교(같은 20건을 두 판정기에)는 Phase 0-3 에서.

### eval 프로필 — 회귀 게이트 기준 (N=324 청크 @ `eval-corpus-v1`)

| retriever | 전체 recall@5 | MRR | 문서 12 / 코드 8 |
|---|---:|---:|---|
| **운영 파이프라인** | **85%** | 0.71 | 92% / 75% |

범위 밖 거절 **6/6**. baseline = `eval/baselines/eval.json`.
코퍼스가 `eval-corpus-v1` 태그로 고정돼 있어 커밋을 해도 이 숫자는 움직이지 않는다.
**코퍼스를 갱신하려면 새 태그 + baseline 재생성**(둘 다 명시적 행위여야 한다 → 노트 #18).

### demo 프로필 — 라이브 데모용 (N=285 청크 시점 측정 · 문서 45 / 코드 211 / 커밋 29)

골든셋 `eval/questions.demo.json` — 문서 12 · 코드 8 · 멀티홉 4 · 범위밖 6 (전부 `origin: manual`).

| retriever | 전체 recall@5 | MRR | 문서 12 / 코드 8 |
|---|---:|---:|---|
| vector only | 85% | 0.61 | 92% / 75% |
| bm25 only | 90% | 0.76 | 100% / 75% |
| hybrid RRF | 90% | 0.77 | 100% / 75% |
| **운영 파이프라인** | **90%** | 0.72 | 100% / 75% |

범위 밖 거절 **6/6**. demo 는 워킹트리를 보므로 커밋마다 코퍼스가 커진다 — **게이트에는 쓰지 않는다**.

⚠️ **private 의 93% 와 직접 비교하지 말 것.** 코퍼스 크기(285 vs 3,619)도 문항 수(20 vs 28)도
다르고, demo 는 문서축이 3개 파일뿐이라 사실상 코드·커밋 축 평가에 가깝다.

**운영 파이프라인 = RRF + 심볼슬롯 + MMR(λ=1.0 → no-op) + 범위밖 게이트.**
이름이 예전엔 `hybrid+MMR` 이었는데, λ=1.0 이면 MMR 은 순위를 바꾸지 않으므로
그 행의 차이는 MMR 이 아니라 심볼 슬롯이다 → 노트 #17.

### 검색 지연 (노트 #15)

벡터 전수매칭 0.12 ms · BM25 4.7 ms · LLM 첫 토큰 ~1,400 ms.
ANN 전환 임계는 **N ≈ 10만** (지연이 아니라 메모리가 먼저 무너진다).

## 5. 로드맵

### Phase 0 — 평가 신뢰도 (진행 중)

지금 숫자들의 최대 약점은 **평가셋이 자가 라벨(40문항)이고 groundedness 가 자기 채점**이라는 것.
자가 채점된 자로 잰 값은 그 자체로는 근거가 약하다. 자를 먼저 고친다.

- [x] **0-4** 노트 #15 — 브루트포스 전수매칭은 의도된 선택 + ANN 전환 임계 측정
- [x] **0-5** 코퍼스 프로필 분리 (demo/private) — 노트 #16
- [x] **0-1** CI 회귀 게이트 — `eval/report.py`(JSON+Markdown 산출·baseline 비교) +
      `.github/workflows/eval.yml`. 골든셋 20문항, 게이트 로직은 `tests/test_report.py` 로 고정
- [x] **0-7** 평가 코퍼스를 태그로 고정(`eval` 프로필 · `GitSnapshotSource`) — 노트 #18.
      첫 CI 실행이 실패해서 드러난 결함: demo 는 저장소 자기 자신이라 커밋마다 코퍼스가 바뀐다
- [x] **0-6** 평가표 행 이름 정정 — 노트 #17 (심볼 슬롯을 MMR 의 공으로 읽고 있었다)
- [~] **0-2** 합성 평가셋 — **258문항 생성 완료**(`eval/generate.py`, 원본 청크 기반·검색기 미사용).
      전체 recall 85%(수기 20) → **79%(합성 258)**, 커밋 축 69% 로 최약점 발견 → 노트 #19.
      거짓 오답 감사 완료(`eval/audit_misses.py`) → 보정 recall 79.5%~93.8% **구간** → 노트 #20.
      남은 것: ① 49문항 **수동 검수**(`eval/verification/worksheet.md` — 조각만 읽고 ok/wrong/unclear)
      ② RAGAS 대조(격리 환경 `.venv-ragas` 구축 완료, `eval/ragas_score.py` 준비됨)
- [ ] **0-3** cross-judge — 판정 모델 분리는 구현 완료(`eval/llm.py` ModelSpec,
      `faithfulness --judge-provider/--judge-model`). 남은 것은 **동일 표본 통제 비교**:
      같은 문항·같은 답변을 두 판정기에 걸어 self vs cross 편차를 수치화.
      ⚠️ 제약: `gemini-3.5-flash` 무료 티어는 **하루 20요청** → 표본 설계 필수(노트 #20)

### 지금 사람이 해야 할 일 (Day 4 입력)

워크시트(`eval/verification/worksheet.md`)에서 **조각을 읽고** 세 값 중 하나만 적으면 된다.
저장소를 뒤질 필요 없고, "다른 파일에도 답이 있나"는 기계가 이미 감사했다(노트 #20).

| 판정 | 언제 |
|---|---|
| `ok` | 조각을 읽고 질문에 답할 수 있다 ← 기본값 |
| `wrong` | 조각에 답이 없다 |
| `unclear` | 질문이 모호하다 |

```bash
python -m eval.verify score --profile eval      # 문항 불량률 계산 + 검수본 저장
```

### Phase 0 에서 나온 다음 과제

- **의도 라우팅** — 심볼 슬롯이 코드 질문 MRR 은 올리고(0.44→0.53) 문서 질문 MRR 은 깎는다(0.88→0.78).
  질문이 어느 축을 묻는지 판별해 장치를 켜고 끄면 양쪽을 다 얻는다 → 노트 #17
- **CI baseline 의 플랫폼 종속** — baseline 을 Windows 에서 만들고 Linux 러너에서 비교하면
  부동소수점 차이로 한 문항이 흔들릴 수 있다(20문항 = 한 문항이 5%, 허용치는 1%).
  → baseline 은 CI 가 만든 것을 쓴다(`workflow_dispatch` 의 `save_baseline` 입력).
  문항이 500개가 되면 한 문항의 무게가 0.2% 로 줄어 이 문제는 자연히 완화된다.

### Phase 1 — 규모와 저장소

- [ ] 1-2 확장된 평가셋으로 재측정 → **하락분 원인 규명** (하락이 곧 개선 재료다)
- [ ] 1-3 pgvector 이전 + exact vs HNSW(`ef_search`) 스윕 → 노트 #15 의 임계 N 실증
- [ ] 1-4 증분 인덱싱 (변경분만 갱신, 현재는 전체 `--reset`)
- [ ] 1-5 한글 형태소 분석기(kiwi) 도입 전/후 BM25 recall 비교 (현재는 문자 2-gram 근사)

### Phase 2 — 제품화

- [x] 2-1 **멀티턴 + 질의 재작성** — `app/conversation.py`. 후속 질문("그건 왜?")을 독립형으로
      재작성해 검색하고, 생성에는 원문 대화를 준다(검색기와 생성기의 요구가 다르므로 입력을 나눔).
      재작성 결과를 UI 에 노출한다 — 무엇으로 검색했는지 보이지 않으면 답의 출처를 알 수 없다.
- [x] 2-5 **웹 UI 전면 개편** — 디자인 토큰·라이트/다크·반응형·접근성(aria-live·focus-visible·
      reduced-motion)·스트리밍 중단·각주[n]↔근거 카드 연동·검색 진단 패널. 외부 의존 0(단일 파일)
- [ ] 2-2 라우터: 단발 RAG vs 에이전트 자동 선택 (에이전트는 한 질문에 LLM 3~5회, 지연 ~10초)
- [ ] 2-3 Terraform + ECS + RDS 배포
- [ ] 2-4 관측 대시보드 + 비용 알람

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
