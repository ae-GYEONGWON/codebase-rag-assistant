# Codebase RAG 어시스턴트

한 코드베이스의 **문서 · 소스코드 · git 이력**을 통합 인덱싱해,
*"지금 어떤 설정으로 동작하지?"*, *"`apply_retry_policy` 함수는 뭘 해?"*, *"이 상한값은 언제 왜 없앴어?"*
같은 질문에 **근거 + 출처와 함께** 답하는 RAG 어시스턴트.

> 인덱싱 대상은 `.env`(`KNOWLEDGE_DIRS` / `CODE_DIRS` / `GIT_REPOS`)로 지정한다.
> 아래 측정 수치는 **운영 중인 실시간 시계열 처리 시스템(비공개)** 을 대상으로 얻은 값이다.

단순 "문서 넣고 질문하는 챗봇"이 아니라, **검색 품질과 답변 충실도를 자체 평가 하네스로 수치화**하고
그 수치로 설계를 결정한 것이 핵심이다. 전 구간 **무료 스택**(Gemini 무료 티어 + 로컬 임베딩)으로 운영비 0원.

---

## 5분이면 확인할 수 있는 것

> 서버를 띄우지 않아도 아래 표만 읽으면 무엇이 되는 물건인지 알 수 있게 썼다.
> 띄운다면 `python -m uvicorn app.main:app --port 8123` 뒤 **http://127.0.0.1:8123** —
> 첫 화면에 아래 질문들이 **기능 지도**로 놓여 있고, 각 질문마다 "무엇을 보게 되는지"가 붙어 있다.

| 무엇을 확인하나 | 물어볼 질문 | 화면에서 볼 것 | 실측 |
|---|---|---|---:|
| **세 축을 함께 본다** | `리랭커를 왜 기본으로 껐어?` | 근거 카드의 `DOC` 배지 + 답변의 각주 `[n]` | 2.2s |
| | `RRF 융합은 코드에서 어떻게 구현돼 있어?` | 진단 패널 `질문 의도 = code`, `CODE` 배지 | 1.8s |
| | `평가 코퍼스를 태그로 고정한 이유가 뭐야?` | `COMMIT` 배지(`git:해시`)가 섞여 나온다 | 2.5s |
| **함수 이름 정확매칭** | `_symbol_hits 함수는 뭘 해?` | 진단 패널의 `심볼 슬롯` 줄 | 1.8s |
| **근거 없으면 거절** | `김치찌개 맛있게 끓이는 법 알려줘` | 근거 0건 — **LLM 을 아예 호출하지 않는다** | **0.0s** |
| **멀티홉 → 에이전트 자동 전환** | `리랭커를 왜 껐고 지금 코드는 어떻게 돼 있어?` | 보라색 배지가 답변보다 **먼저** 뜨고 근거가 12건까지 | 6.7s |
| **후속 질문 이해** | (위 답변 뒤) `그건 어떻게 측정했어?` | "후속 질문을 이렇게 이해했습니다" 배지 | 3.0s |

두 가지가 특히 눈에 띌 것이다.

- **범위 밖 거절이 0.0초** — 코사인 임계로 검색 단계에서 자르므로 LLM 호출이 없다.
  "환각하지 말라고 프롬프트로 시켰다"가 아니라 구조로 막았고, 그래서 비용도 0이다.
- **에이전트가 근거를 12건 모은다** — 단발 경로는 5건이다. 축마다 따로 검색했다는 것이 숫자로 보인다.

### 그리고 `/eval` — 이 프로젝트의 실제 내용

챗봇 화면은 이 저장소에서 **가장 덜 중요한 부분**이다. 시간을 쓴 곳은 평가 체계이고,
그 결과는 **http://127.0.0.1:8123/eval** 에 있다(스냅샷이라 서버만 띄우면 바로 보인다).

- 리트리버별 recall·MRR — 코퍼스·문항 수가 다른 네 벌을 **섞지 않고** 따로
- LLM 판정기 5종을 같은 답변에 걸어 본 결과 — 평균은 모였는데 **합의는 우연 이하**(κ ≈ -0.1)
- 환각 유도 문항 29개와 **실제로 지어낸 사례**
- 브루트포스 → ANN 전환 임계 실측
- **측정해서 기각한 판단 10건** ← 이 페이지의 결론부

---

## 무엇이 다른가 (측정으로 증명)

| 지표 | 벡터 단독 | **하이브리드(본 시스템)** |
|---|---:|---:|
| 검색 recall@5 (전체 28문항) | 75% | **93%** |
| MRR | 0.53 | **0.68** |
| — 문서 질문 20 | 85% | **100%** |
| — 코드 질문 8 | 50% | **75%** |
| 답변 groundedness (환각 없음, LLM-judge) | — | 0.962 † |
| 범위 밖 질문 거절률 | — | **83%** (5/6) |

> 지식원: 문서 1,022 + 코드 2,331 + git 커밋 266 = **3,619 청크**.
> 측정 재현: `python -m eval.run_eval` (검색), `python -m eval.faithfulness` (답변 충실도).

**† 이 숫자는 단독으로 쓰지 않는다.** 판정기를 4종으로 바꿔 같은 답변을 채점해 봤더니
평균은 0.950~1.000 안에 몰렸지만 **어느 답이 환각인지에 대한 합의는 0 이었다**
(Cohen's κ ≈ -0.1, 우연 이하 → [노트 #21](docs/engineering-notes.md)).

그래서 **환각을 유도하는 적대적 문항 29개를 만들어 다시 쟀다.** 그런데도 값이 안 움직였고,
원인은 평가셋이 아니라 **지표의 정의**였다 — 거절은 근거에 없는 주장을 하지 않으므로
자동으로 만점이라, groundedness 는 '말하지 않아서 얻는 만점'을 구분하지 못한다
([노트 #22](docs/engineering-notes.md)).

→ 환각 여부는 **행동 기반 채점**으로 잰다. 덫 종류별 통과 조건을 두고 채점하면 지표가
비로소 변별한다: **통과율 97%**(29문항 중 환각 1건 검출).

| 덫 | 문항 | 통과율 | 옳은 행동 |
|---|---:|---:|---|
| `absent` (없는 사실) | 10 | 100% | 근거 없음을 밝힌다 |
| `partial` (절반만 있음) | 7 | 100% | 있는 갈래만 답한다 |
| `superseded` (옛 값이 튀어나옴) | 12 | 92% | 현재 값을 답한다 |

지표를 자랑하기 전에 그 지표가 변별력이 있는지를 먼저 쟀고, 없다는 결과를 그대로 적는다.

구현하며 마주친 문제와 해결 과정 27건은 **[엔지니어링 노트](docs/engineering-notes.md)** 에 정리.

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
- **BM25(어휘) + 벡터(의미)를 RRF 로 융합** — 임베딩은 `ERR7742` 같은 희귀 식별자에 약하고,
  BM25 는 표현이 다른 질문에 약하다. 서로의 약점을 메운다.
- **심볼 정확 매칭** — 질문에 코드 심볼명(`apply_retry_policy`)이 있으면 그 함수 본문을 top-k 에 보장.
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
- **근거 → 원문** — 근거 카드에서 파일 전문을 열고 그 근거가 나온 줄을 강조해 보여준다.
  발췌만으로는 근거를 검증할 수 없다(앞뒤가 잘려 있으면 반대 뜻이어도 모른다). 저장소
  링크도 함께 걸되 **브랜치가 아니라 커밋 해시**로 건다 — 브랜치는 움직인 뒤 다른 줄을
  가리킨다 → [노트 #27](docs/engineering-notes.md).
- **검색 범위 지정**(자동/문서/코드/커밋 이력) — 축 판별이 틀렸을 때 사람이 못 박을 수 있게.
  자연어로 "코드에서만"이라고 해도 그 문장이 다시 같은 판별기를 통과하므로 스위치가 필요하다.
- **대화 보관·공유** — 대화는 브라우저에 남아 새로고침해도 이어지고, 링크로 공유하면
  받는 사람이 답변뿐 아니라 **근거와 검색 진단까지** 그대로 본다.
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

python -m app.ingest                        # 지식원(문서+코드+git) → 벡터 DB 인덱싱
uvicorn app.main:app --port 8123           # http://127.0.0.1:8123
```

기본 프로필이 `demo`(이 저장소 자기 자신)라 **clone 직후 경로 설정 없이 바로 동작한다.**
다른 코드베이스를 붙이려면 아래 [코퍼스 프로필](#코퍼스-프로필) 참조.

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

## 코퍼스 프로필

"무엇을 인덱싱하는가"는 `app/profiles.py` 의 **프로필**이 결정한다. 프로필 = (지식원 · 컬렉션 · 평가셋) 한 벌.

| 프로필 | 지식원 | 컬렉션 | 평가셋 |
|---|---|---|---|
| `demo` (기본) | **이 저장소 자기 자신** — `git ls-files` 추적 파일(워킹트리) | `corpus_demo` | `eval/questions.demo.json` |
| `eval` | 저장소 스냅샷 **@ 태그 `eval-corpus-v1`** — `git archive` 로 고정 | `corpus_eval_…` | 같음 |
| `private` | `.env` 의 `KNOWLEDGE_DIRS`/`CODE_DIRS`/`GIT_REPOS` | `.env` 의 `COLLECTION_NAME` | `eval/questions.json` |

```bash
python -m app.ingest    --profile demo --reset   # 프로필 지정 인덱싱
python -m eval.run_eval --profile private        # 인덱스와 평가셋이 함께 전환된다
```

두 프로필은 같은 `chroma_db/` 안에서 **컬렉션 이름으로 분리**되어 공존한다(전환에 재인덱싱 불필요).
`--reset` 은 해당 프로필의 컬렉션만 지운다.

demo 코퍼스를 폴더 walk 가 아니라 **git 추적 파일**로 정의한 것은 측정 재현성 때문이다 —
로컬에만 있는 파일이 섞이면 같은 코드인데 PC·CI 마다 청크 수와 recall 이 달라진다
(→ [engineering-notes #16](docs/engineering-notes.md)).

**CI 회귀 게이트는 `demo` 가 아니라 `eval` 프로필을 쓴다.** demo 는 워킹트리를 보므로
커밋할 때마다 코퍼스가 커져, 점수 변화가 코드 탓인지 문서 탓인지 구분되지 않는다
(→ [engineering-notes #18](docs/engineering-notes.md)).

새 축(로그·티켓 등)은 `app/profiles.py` 에 `@register` 함수 하나만 추가하면 된다.

## 설정 (`.env`)

| 키 | 기본값 | 설명 |
|---|---|---|
| `CORPUS_PROFILE` | `demo` | 인덱싱 대상 프로필 — `demo` \| `private` |
| `GOOGLE_API_KEY` | (필수) | Gemini 무료 키. 없으면 extractive 폴백 |
| `GEMINI_CHAT_MODEL` | `gemini-3.1-flash-lite` | 생성 모델 |
| `EMBEDDING_PROVIDER` | `hf` | `hf`(로컬 무료) \| `openai` |
| `KNOWLEDGE_DIRS` | `/path/to/your/repo/docs` | 문서 지식원(쉼표 구분) |
| `CODE_DIRS` | `/path/to/your/repo/app,...` | 코드 지식원 (`INDEX_CODE=false` 로 끄기) |
| `GIT_REPOS` | `/path/to/your/repo` | git 이력 지식원 (`INDEX_GIT=false` 로 끄기) |
| `USE_RERANKER` | `false` | 리랭커(측정상 비활성 권장) |

> **다른 프로젝트 재사용**: `CORPUS_PROFILE=private` 로 두고 `KNOWLEDGE_DIRS`/`CODE_DIRS`/`GIT_REPOS` 만
> 바꾸면 어떤 코드베이스에도 붙는다.
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
  profiles.py     ★코퍼스 프로필(demo|private) — 지식원·컬렉션·평가셋 한 벌
  fs_utils.py     지식원 파일 열거 전략(폴더 walk | git 추적 파일 | ref 스냅샷)
  loader.py       .md → 헤더 인지 청크
  code_loader.py  .py → AST 함수/클래스 청크 + 컨텍스트 헤더
  git_loader.py   git log → 커밋 청크(날짜 메타)
  embeddings.py   로컬 hf | openai 임베딩
  ingest.py       문서+코드+git 통합 인덱싱 → Chroma
  retriever.py    BM25+벡터 RRF → 심볼매칭 → MMR → 범위밖 게이트
  reranker.py     cross-encoder(기본 off, 측정 근거)
  rag.py          근거기반 생성 + 스트리밍, 문서/코드 혼합 프롬프트
  intent.py       질문 의도 판별(문서/코드/커밋) — 규칙 기반, LLM 미사용
  router.py       단발 RAG vs 에이전트 자동 선택(축을 둘 이상 물으면 에이전트)
  tokenizer.py    BM25 토크나이저 — 문자 2-gram 근사 ↔ 형태소 분석기(kiwi) 교체 가능
  index_state.py  인덱스 버전 도장 — 서빙 프로세스가 재기동 없이 갱신을 알아챈다
  feedback.py     👍/👎 로그
  source_view.py  근거 원문 조회 — 인덱싱된 파일만 여는 허용목록 + 근거 줄 계산
  share.py        대화 공유 — 서버가 스키마를 강제해 아는 필드만 저장한다
  main.py         FastAPI 엔드포인트 + 기동 워밍업
eval/
  datasets.py     평가셋 로딩(프로필별) + 라벨 출처(manual|synthetic) 집계
  report.py       ★평가 결과 JSON/Markdown 산출 + baseline 대비 회귀 판정(CI 게이트)
  questions.demo.json     골든셋 20문항(수기 라벨) — demo·eval 프로필 공용
  baselines/      회귀 게이트 기준선(demo·eval 만 추적)
  questions.example.json  평가셋 템플릿(→ questions.json 으로 복사해 대상에 맞게 작성)
  run_eval.py     검색 평가(recall@k·MRR·거절률, 리트리버별 비교)
  faithfulness.py 답변 groundedness(LLM-as-judge)
  generate.py     합성 평가셋 생성(원본 청크 기반, 검색기 미사용) + 어휘 중복률 검사
  generate_hard.py ★적대적 평가셋 생성 — 환각이 나오는 문항을 의도적으로 만들고
                  생성기와 다른 모델로 "정말 근거가 없는지" 검증해 걸러낸다
  hard_eval.py    ★행동 기반 채점 — 덫 종류별 통과 조건(거절/부분답변/현재값)
  audit_misses.py 거짓 오답 감사 — miss 를 독립 판정기로 재확인(라벨 편향 보정)
  judge_panel.py  ★판정기 패널 — 답변을 동결하고 판정기만 교체해 통제 비교
                  (부트스트랩 CI·Cohen's κ·Spearman ρ, 쿼터 중단 안전)
  ann_threshold.py ANN 전환 임계 실증 — 지연·메모리는 합성으로, recall 은 실제 임베딩으로
  ragas_score.py  RAGAS Faithfulness 대조(격리 환경 .venv-ragas 에서 실행)
tests/            pytest (토크나이저·정규화·config·AST·검색통합)
docs/
  engineering-notes.md  구현 중 마주친 문제 27건
  HANDOFF.md            작업 현황·로드맵(여러 PC 에서 이어서 개발)
```

---

### 이력서용 한 줄
> **운영 중인 실시간 시스템의 문서·소스코드·git 이력(3,600+ 청크)을 대상으로 한 RAG 어시스턴트 구축** —
> BM25+벡터 하이브리드 검색(RRF)과 코드 심볼 매칭으로 검색 recall 을 벡터 단독 75%→93% 개선,
> 자체 평가 하네스로 검색 정확도와 답변 충실도를 수치화하고 **그 평가자 자체를 검증** —
> 답변을 고정한 채 판정 모델만 교체하는 통제 비교로 LLM-judge 의 문항 단위 합의도가
> 우연 수준임을 발견(Cohen's κ ≈ -0.1)하고, **환각 유도 문항 29개를 만들어 재측정한 뒤
> 원인이 평가셋이 아니라 지표의 정의임을 규명**해 행동 기반 채점으로 교체(통과율 97%).
> 증분 인덱싱(내용 해시)·질의 의도 라우팅·단발/에이전트 라우터, AST 코드 청킹,
> 범위 밖 질문 거절, 문서·코드 불일치 판별, 토큰 스트리밍 서빙까지 무료 스택(Gemini·Chroma·FastAPI)으로 구현.
