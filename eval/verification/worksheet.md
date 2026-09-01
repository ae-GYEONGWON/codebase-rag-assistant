# 합성 평가셋 수동 검수 워크시트

표본 49문항 (전체 258문항에서 축 비율 유지 추출, seed=20260903)

**판정에 필요한 건 화면에 보이는 조각뿐이다.** 저장소를 뒤질 필요 없다.
각 항목의 `verdict:` 줄에 아래 하나를 적고, 다 채운 뒤 `python -m eval.verify score`.

| 판정 | 언제 |
|---|---|
| `ok` | 아래 조각을 읽고 질문에 답할 수 있다 ← **기본값. 헷갈리면 이걸로** |
| `wrong` | 조각에 질문의 답이 없다(엉뚱한 조각이 붙었다) |
| `unclear` | 질문 자체가 무슨 말인지 모르겠다 |

한 문항 30초. 오래 고민되면 `ok` 로 두고 넘어갈 것. 빈칸은 집계에서 자동 제외된다.

> **'다른 파일에도 답이 있나'는 판정하지 않아도 된다.** 그건 저장소 전체를 알아야 하는
> 일이라 사람보다 기계가 낫다 — `eval/audit_misses.py` 가 독립 판정기로 전수 확인한다.
> (아는 경우에만 `elsewhere` + `also:` 로 적으면 그 값도 함께 반영된다.)
>
> 이 워크시트에는 **검색기가 무엇을 찾았는지 넣지 않았다.** 그걸 보고 판정하면
> 검수가 검색기를 정당화하는 절차가 되어 버린다.

---

## 1. faithfulness.py의 main 함수에서 평가 점수를 계산할 때 어떤 과정을 거쳐서 최종 평균값을 도출해?

- 라벨: `eval/faithfulness.py` · 축 `code` · 어휘중복 0.138

```
# 파일: eval/faithfulness.py
# 심볼: main
scored: List[float] = []
    for c in cases + [{"q": oos, "expected": []}]:
        q = c["q"]
        result = _call(lambda: answer(q))
        docs, _ = search(q)
        # 생성기가 본 것과 동일한 근거(파일명·섹션 헤더 포함)를 judge 에도 준다.
        context = _format_context(docs) if docs else ""
        verdict = _judge(q, context, result["answer"])
        s = verdict.get("score")
        if s is not None:
            scored.append(s)
        n_unsup = len(verdict.get("unsupported", []))
        flag = verdict.get("note", "")
        print(f"{(f'{s:.2f}' if s is not None else '  -- '):>6}  {n_unsup:>4}  {q[:48]}  {flag}")
        for u in verdict.get("unsupported", [])[:2]:
            print(f"{'':>14}↳ {u[:70]}")

    if scored:
        avg = sum(scored) / len(scored)
        print(f"\n평균 groundedness = {avg:.3f}  (n={len(scored)}, 1.0=환각 0)")
    print()
```

verdict: 
also: 

---

## 2. 사용자가 특정 함수 이름을 언급하면서 검색하면, 결과에 해당 코드 블록이 우선적으로 노출되도록 설계되어 있나요?

- 라벨: `tests/test_retrieval.py` · 축 `code` · 어휘중복 0.048

```
# 파일: tests/test_retrieval.py
# 심볼: test_symbol_exact_match_boosts_code
def test_symbol_exact_match_boosts_code():
    """질문에 코드 심볼명이 있으면 그 함수 본문(코드 청크)이 top-k 에 포함."""
    from app.retriever import search

    sym = _a_code_symbol()
    docs, _ = search(f"{sym} 함수 코드 보여줘")
    types = {d.metadata.get("doc_type") for d in docs}
    assert "code" in types
    assert any(sym in (d.metadata.get("section") or "").lower() for d in docs)
```

verdict: 
also: 

---

## 3. 부동소수점 오차 때문에 리포트 검증이 실패하는 문제를 어떻게 방지하고 있어?

- 라벨: `tests/test_report.py` · 축 `code` · 어휘중복 0.2

```
# 파일: tests/test_report.py
# 심볼: test_허용치_안의_흔들림은_통과시킨다
def test_허용치_안의_흔들림은_통과시킨다():
    """부동소수점·동점 순서 차이로 한 문항이 흔들리는 것까지 실패로 보면 게이트가 못 쓰게 된다."""
    assert rp.compare(_report(recall=0.90), _report(recall=0.895)) == []
```

verdict: 
also: 

---

## 4. app/agent_graph.py를 구현하면서 왜 기본 제공되는 ToolNode를 사용하지 않고 별도로 노드를 구성했나요?

- 라벨: `git:b2515af` · 축 `commit` · 어휘중복 0.281

```
# 커밋 b2515af (2026-08-05)
제목: feat(agent): LangGraph 판 에이전트 추가 — 수동 루프와 동일 평가셋 비교

app/agent.py 의 while 루프를 LangGraph 상태 그래프로 옮겼다. 교체가 아니라
나란히 둔다 — 프레임워크 도입이 무엇을 바꾸고 무엇을 그대로 두는지
같은 평가셋으로 말할 수 있어야 하기 때문.

  START → agent ─(tool_calls 없음)→ END
            ↑   ─(있음·상한 이내)→ tools ─┘
                ─(있음·상한 초과)→ finalize → END

측정(멀티홉 12문항, 홉 커버리지 / 전체정답):
  단발 RAG k=5(운영)      48% / 17%   0.5s
  단발 RAG k=12(예산맞춤)  70% / 42%   0.0s
  에이전트(수동 루프)      78% / 50%  10.0s  LLM 2.5회
  에이전트(LangGraph)     78% / 50%  10.4s  LLM 2.6회
→ 정답률·지연·호출 수는 동등. 달라지는 것은 구조다.

★포팅 중 잡은 버그(기록 가치 있음): 1차 포팅은 라운드 상한을
GraphRecursionError 예외에 맡겼는데, 그 폴백이 모아둔 근거를 통째로
버렸다(수동 루프는 같은 자리에서 '강제 마무리'를 한다). 결과가 70%/42% 로
낮게 나왔고 2회 재현까지 동일해 편차가 아니었다. 한 질문을 추적해
llm_calls=0 / sources=0 을 확인하고 원인을 규명 → finalize 노드를 추가해
상한 처리를 수동 루프와 대칭으로 맞추자 수치가 일치했다.
"프레임워크가 더 나빴다" 로 끝냈으면 틀린 결론이 될 뻔했다.

prebuilt ToolNode 를 쓰지 않은 이유는 전역 인용 번호 채번과 청크 중복
제거가 노드 바깥 상태를 필요로 해서다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

변경 파일:
  - app/agent_graph.py
  - eval/run_eval.py
  - requirements.txt
```

verdict: 
also: 

---

## 5. 검색 기능 테스트를 돌리려면 로컬 환경에 어떤 데이터가 미리 준비되어 있어야 해?

- 라벨: `tests/test_retrieval.py` · 축 `code` · 어휘중복 0.296

```
# 파일: tests/test_retrieval.py
# 심볼: (module)
"""검색 파이프라인 통합 테스트.

인덱싱된 chroma_db + 로컬 임베딩 모델이 있어야 의미가 있으므로, 없으면 skip.
(CI 에서는 인덱스를 만든 뒤 실행하거나, 이 파일만 마커로 제외)

※ 질의를 코퍼스에서 동적으로 뽑는다 — 특정 프로젝트의 심볼·용어를 하드코딩하지 않으므로
   어떤 코드베이스를 인덱싱했든 그대로 통과한다.
"""
from pathlib import Path

import pytest

from app.profiles import active_profile

_HAS_INDEX = Path(active_profile().chroma_dir).exists()
pytestmark = pytest.mark.skipif(not _HAS_INDEX, reason="chroma_db 인덱스 없음")
```

verdict: 
also: 

---

## 6. 현재 시스템에서 브루트포스 방식 대신 ANN으로 검색 전략을 바꿔야 하는 데이터 규모 기준이 어떻게 돼?

- 라벨: `docs/engineering-notes.md` · 축 `doc` · 어휘중복 0.242

```
**전환 임계**: 지연으로는 N=100,000 에서도 5 ms 라 문제가 안 된다. 먼저 무너지는 건 **메모리**다.
코퍼스 전체를 프로세스에 상주시키므로 **인스턴스 사양과 콜드스타트가 데이터 크기에 비례**한다.
Fargate 1 GB 구성 기준으로 임베딩 행렬 300 MB(N≈10만)가 임베딩 모델·런타임과 함께 올라갈 수 있는 한계다.
→ **N ≈ 10만이 ANN(또는 pgvector 서버사이드 검색) 전환 임계**이고, 그 전까지는 exact 가 더 정확하고 더 빠르다.  
**교훈**: 리랭커를 측정해서 껐던 것과 같은 종류의 판단인데, 이건 기록이 없어서 **판단이 아니라 사고처럼
보였다.** 고른 것과 그냥 그렇게 된 것의 차이는 코드가 아니라 **숫자와 기록**에 있다.
(파생 결함 2개 — 메모리가 코퍼스에 비례, 인덱싱·서빙 결합으로 재인덱싱에 서버 정지 필요 —
는 실재하며, 이것이 pgvector 이전의 명분이다. "벡터DB 경험이 필요해서"가 아니라.)
```

verdict: 
also: 

---

## 7. docker-compose.yml을 활용해서 쿼리 서빙 환경을 구성할 때 인덱싱 데이터는 어떻게 관리해야 해?

- 라벨: `git:ad07a2f` · 축 `commit` · 어휘중복 0.276

```
# 커밋 ad07a2f (2026-07-15)
제목: test+docker: pytest 21개(순수 로직 + 검색 통합) + Dockerfile/compose

- tests/: 토크나이저(식별자·snake·한글2gram), Gemini 멀티파트 정규화, config 폴백,
  AST 청킹, 검색 통합(범위밖 거절·심볼매칭·출처정렬). 통합 테스트는 chroma 없으면 skip
- Docker: 쿼리 서빙 컨테이너(CPU torch). 인덱싱은 호스트, 컨테이너는 chroma_db 마운트.
  compose 로 .env·볼륨·HF 캐시 배선. ※ 이 환경엔 docker 미설치로 빌드 자체는 미검증

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

변경 파일:
  - .dockerignore
  - Dockerfile
  - docker-compose.yml
  - pytest.ini
  - requirements.txt
  - tests/__init__.py
  - tests/test_code_loader.py
  - tests/test_config.py
  - tests/test_retrieval.py
  - tests/test_text_normalize.py
  - tests/test_tokenizer.py
```

verdict: 
also: 

---

## 8. 최근에 첫 질문 응답 속도가 너무 느렸는데, main.py에서 어떤 조치를 취해서 개선한 거야?

- 라벨: `git:96c61e1` · 축 `commit` · 어휘중복 0.111

```
# 커밋 96c61e1 (2026-07-14)
제목: fix(llm): Gemini 실키 검증 — 신형 모델 content 파트배열 정규화 + 모델 교체 + 기동 워밍업

- rag.py: _text_of() 추가. Gemini 3.x 는 content 를 [{"type":"text",...}] 파트 배열로
  반환해 기존 isinstance(str) 경로가 str(list) 로 깨졌음 → text 파트만 결합
- 모델: gemini-2.5-flash 는 신규 발급 키에 404(no longer available to new users)
  → gemini-3.1-flash-lite 로 교체(첫 토큰 1.4s. 3.5-flash 는 thinking 탓에 ~20s)
- main.py: lifespan 워밍업으로 첫 질문 18.6s → 3.5s (HF 임베딩 지연로딩 제거)
- 검증: /health llm_provider=gemini, /chat/stream 토큰 스트리밍·[n]인용·범위밖 거절 정상

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

변경 파일:
  - .env.example
  - app/config.py
  - app/main.py
  - app/rag.py
```

verdict: 
also: 

---

## 9. 사용자가 질문했을 때 검색 결과가 하나도 없으면 응답이 어떤 식으로 처리돼?

- 라벨: `app/rag.py` · 축 `code` · 어휘중복 0.036

```
# 파일: app/rag.py
# 심볼: stream_answer
def stream_answer(question: str, dev_mode: bool = False) -> Iterator[Dict[str, Any]]:
    """질문 → 이벤트 스트림. 각 이벤트: {type: sources|token|done|error, ...}."""
    docs, debug = search(question)

    if not docs:
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": OUT_OF_SCOPE}
        yield {"type": "done", "mode": "no_hit", "retrieval": debug}
        return

    yield {"type": "sources", "sources": snippets_for(docs, question)}

    provider = settings.active_llm
    if provider == "extractive":
        yield {"type": "token", "text": _extractive_text(docs)}
        yield {"type": "done", "mode": "extractive", "retrieval": debug}
        return
```

verdict: 
also: 

---

## 10. 866a012 커밋에서 적용된 변경 사항은 구체적으로 어떤 목적을 가지고 있어?

- 라벨: `git:866a012` · 축 `commit` · 어휘중복 0.08

```
# 커밋 866a012 (2026-09-01)
제목: merge: v3-eval-ci — CI 회귀 게이트(Phase 0-1)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019nZhqfMvSn5y45f35gJSS4

```

verdict: 
also: 

---

## 11. 회귀 테스트에서 게이트가 제대로 작동하는지 확인하는 이유는 뭐야?

- 라벨: `tests/test_report.py` · 축 `code` · 어휘중복 0.36

```
# 파일: tests/test_report.py
# 심볼: (module)
"""평가 리포트 · 회귀 게이트 로직 테스트.

게이트가 조용히 망가지면 회귀를 놓치고도 초록불이 뜬다 — 그래서 게이트 자체를 고정한다.
인덱스·임베딩이 필요 없는 순수 로직이라 CI 에서 그대로 돈다.
"""
import json

import pytest

from eval import report as rp
```

verdict: 
also: 

---

## 12. 기존의 단발성 RAG 방식이 복합적인 질문을 처리할 때 겪던 한계가 뭐야?

- 라벨: `git:d870c7e` · 축 `commit` · 어휘중복 0.16

```
# 커밋 d870c7e (2026-08-03)
제목: feat(agent): 툴콜링 에이전트 레이어 추가 — 축별 검색 툴 4종 + 루프

단발 RAG(검색 1회 → 답변)는 멀티홉 질문에 구조적으로 약하다.
"재시도 정책이 왜 바뀌었고 지금 코드는 어떻게 동작해?" 를 던지면
상위 3개가 전부 문서로 채워져 코드·커밋 축이 표면 유사도에 밀린다
(기존 '코드질문 미스 2건'과 같은 원인).

축을 나눈 툴을 주고 LLM 이 호출 순서를 정하게 하면 문서4+커밋4+코드4 를
확보한다. 멀티홉 1질문 = LLM 2회 / 12.3s (gemini-3.1-flash-lite).

- app/agent.py: search_docs/search_code/search_commits/read_symbol +
  루프. answer() 가 trace·steps·llm_calls 를 함께 반환 — '에이전트 vs
  단발 RAG' 를 eval 하네스가 숫자로 비교하기 위한 것.
- app/retriever.py: search(doc_types=…) 축 필터(게이트도 필터 후 기준),
  get_symbol() — 심볼 본문 직접 조회(랭킹 우회).
- app/config.py: use_agent / agent_max_steps / agent_throttle_sec.

기본 USE_AGENT=false. 리랭커와 같은 원칙으로 측정 전에는 켜지 않는다.
프롬프트 1차본은 search_code 를 부르지 않고 문서만으로 현재 구현을
단정해, "'지금 어떻게 동작하는가'는 코드가 근거" 를 명시해 교정했다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

변경 파일:
  - app/agent.py
  - app/config.py
  - app/retriever.py
```

verdict: 
also: 

---

## 13. 임베딩 모델을 허깅페이스로 설정했는데, 필요한 라이브러리가 설치되지 않았을 때 어떤 에러가 발생해?

- 라벨: `app/embeddings.py` · 축 `code` · 어휘중복 0.05

```
# 파일: app/embeddings.py
# 심볼: get_embeddings
def get_embeddings():
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        if not settings.has_openai:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai 인데 OPENAI_API_KEY 가 없습니다. "
                ".env 에 키를 넣거나 EMBEDDING_PROVIDER=hf 로 바꾸세요."
            )
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    if provider == "hf":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as e:
            raise RuntimeError(
                "hf 임베딩을 쓰려면: pip install langchain-huggingface sentence-transformers"
            ) from e

        return HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
```

verdict: 
also: 

---

## 14. 데이터셋에 포함된 문항들이 어떤 경로로 생성되었는지 비율을 확인하려면 어떤 메서드를 써야 해?

- 라벨: `eval/datasets.py` · 축 `code` · 어휘중복 0.029

```
# 파일: eval/datasets.py
# 심볼: QuestionSet.origin_counts
def origin_counts(self) -> Dict[str, int]:
        """라벨 출처별 문항 수 — 자가 라벨 비중을 리포트에 남기기 위한 것."""
        counts: Dict[str, int] = {}
        for case in [*self.in_scope, *self.in_scope_code, *self.multihop]:
            key = case.get("origin", ORIGIN_MANUAL)
            counts[key] = counts.get(key, 0) + 1
        if self.out_of_scope:
            counts[ORIGIN_MANUAL] = counts.get(ORIGIN_MANUAL, 0) + len(self.out_of_scope)
        return dict(sorted(counts.items()))
```

verdict: 
also: 

---

## 15. test_function_is_own_chunk 함수에서는 소스 코드가 의도한 대로 쪼개졌는지 어떻게 판단하고 있어?

- 라벨: `tests/test_code_loader.py` · 축 `code` · 어휘중복 0.188

```
# 파일: tests/test_code_loader.py
# 심볼: test_function_is_own_chunk
def test_function_is_own_chunk():
    segs = _seg_map(SRC)
    assert "def foo" in segs["foo"]
    assert "return a + 1" in segs["foo"]
```

verdict: 
also: 

---

## 16. 토크나이저가 언더바가 포함된 단어를 처리할 때 어떤 방식으로 색인을 생성하는지 궁금해.

- 라벨: `tests/test_tokenizer.py` · 축 `code` · 어휘중복 0.057

```
# 파일: tests/test_tokenizer.py
# 심볼: test_snake_case_split
def test_snake_case_split():
    """snake_case 식별자는 전체 + 조각으로 색인 → 부분 질의도 매칭."""
    toks = _tokenize("apply_retry_policy")
    assert "apply_retry_policy" in toks   # 전체
    assert "retry" in toks
```

verdict: 
also: 

---

## 17. 우리 시스템에서 마크다운 문서를 불러올 때 섹션별로 출처를 명확하게 구분하는 방식이 뭐야?

- 라벨: `app/loader.py` · 축 `code` · 어휘중복 0.057

```
# 파일: app/loader.py
# 심볼: (module)
"""지식원(.md) 파일을 읽어 헤더 인지 청크로 분할.

- Markdown 헤더(#, ##, ###)를 메타데이터로 보존 → 답변 출처가 "파일 > 섹션" 으로 정확히 찍힘
- 큰 섹션은 RecursiveCharacterTextSplitter 로 chunk_size 단위 재분할
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import settings
from app.profiles import active_profile

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
```

verdict: 
also: 

---

## 18. QuestionSet 객체의 요약 정보를 출력하면 어떤 항목들이 표시되는지 알려줄 수 있어?

- 라벨: `eval/datasets.py` · 축 `code` · 어휘중복 0.038

```
# 파일: eval/datasets.py
# 심볼: QuestionSet.summary
def summary(self) -> str:
        origins = ", ".join(f"{k} {v}" for k, v in self.origin_counts().items())
        return (
            f"[{self.profile}] {self.path.name}: 문서 {len(self.in_scope)} · "
            f"코드 {len(self.in_scope_code)} · 멀티홉 {len(self.multihop)} · "
            f"범위밖 {len(self.out_of_scope)} (라벨: {origins})"
        )
```

verdict: 
also: 

---

## 19. eval/faithfulness.py에서 LLM이 답변의 정확성을 판단할 때 어떤 전략을 써서 편향을 줄이고 있어?

- 라벨: `git:c94d9c8` · 축 `commit` · 어휘중복 0.37

```
# 커밋 c94d9c8 (2026-07-15)
제목: feat(eval): 답변 groundedness 평가(LLM-as-judge) — 검색뿐 아니라 환각까지 측정

검색 평가는 '맞는 문서를 찾았나'만 본다. RAG 의 최종 실패는 근거를 두고도 딴소리하는
것(환각)이라, 답변의 각 주장이 근거에서 지지되는지 Gemini judge 로 채점한다.

- eval/faithfulness.py: (근거, 답변) → groundedness 0~1 + 근거없는 주장 목록
  · self-judge 편향 완화: judge 에 생성 맥락 없이 '근거에 없는 주장을 찾아라'는 역과제
  · 생성기와 동일한 근거 포맷(_format_context)을 judge 에 제공(파일명 오판 방지)
  · 무료 티어 분당 15요청 → throttle + 429 재시도로 완주
- 결과: 평균 0.962 (28문항+범위밖1). 범위밖 거절도 1.00(충실)
  · judge 가 실제 과일반화 환각을 검출: "옵션 스로틀은 월물 전환 churn 방지"(0.50)
    → 문서 실제 원인은 OCX 큐 포화+strike flip-flop. 근거 밖 추론을 정확히 지적
- 한계(정직): self-judge 라 판정 편차 있음(프로젝트명·엄격도). 절대점수보다 회귀 감지용

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

변경 파일:
  - eval/faithfulness.py
```

verdict: 
also: 

---

## 20. 우리 프로젝트에서 환경 변수를 불러올 때 어떤 라이브러리를 사용하고 있어?

- 라벨: `app/config.py` · 축 `code` · 어휘중복 0.038

```
# 파일: app/config.py
# 심볼: (module)
"""환경설정 로드 (.env → pydantic-settings)."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict
```

verdict: 
also: 

---

## 21. app/reranker.py에서 설명하는 리랭킹 방식이 기존의 임베딩 기반 검색과 비교했을 때 어떤 차이점이 있어?

- 라벨: `app/reranker.py` · 축 `code` · 어휘중복 0.294

```
# 파일: app/reranker.py
# 심볼: (module)
"""Cross-encoder 리랭커.

임베딩(bi-encoder)은 질문과 문서를 **따로** 벡터로 만들어 비교한다. 빠르지만
"이 문서가 이 질문에 답이 되는가"를 직접 보지는 못한다.
cross-encoder 는 (질문, 문서)를 **한 입력으로 함께** 넣어 관련도를 점수화한다.
느려서 전체 코퍼스에는 못 쓰고, 하이브리드 검색이 좁혀준 후보 20개에만 적용한다.

  전체 3342청크 ──BM25+벡터 RRF──▶ 후보 20 ──cross-encoder──▶ 최종 5
      (싸고 넓게)                        (비싸고 정확하게)

코드 인덱싱으로 코퍼스가 3.5배가 되면서 top-k 노이즈가 늘었고("SL 은 코드에서 어떻게
계산돼?" 가 실제 구현 파일을 못 집었다), 그 지점을 메우는 것이 이 단계의 목적이다.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

from langchain_core.documents import Document

from app.config import settings


@lru_cache(maxsize=1)
```

verdict: 
also: 

---

## 22. RAG 시스템에서 검색 단계가 성공했음에도 불구하고 답변이 부정확해지는 문제를 어떻게 진단하고 있어?

- 라벨: `eval/faithfulness.py` · 축 `code` · 어휘중복 0.179

```
# 파일: eval/faithfulness.py
# 심볼: (module)
"""답변 품질 평가 — LLM-as-judge 로 groundedness(근거 충실도) 채점.

검색 평가(run_eval.py)는 "맞는 문서를 찾았나"만 본다. 하지만 RAG 의 최종 실패는
**찾은 근거를 두고도 답변이 딴소리(환각)를 하는 것**이다. 그걸 잡는 지표가 groundedness:
답변의 각 주장이 제공된 근거에서 실제로 지지되는가.

judge 는 답변을 만든 것과 같은 Gemini(무료)를 쓴다. self-judge 편향을 줄이려 judge 에는
**답변 생성 맥락을 주지 않고**, 오직 (근거, 답변)만 주고 "근거에 없는 주장을 찾아라"라는
반대 방향 과제를 준다. 이렇게 하면 생성기가 놓친 환각을 판별기가 잡아낼 여지가 생긴다.

실행:
    python -m eval.faithfulness            # 문서+코드 질문 전체
    python -m eval.faithfulness --n 6      # 앞 6개만(빠른 점검)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from app.config import settings
from app.profiles import available_profiles, use_profile
from app.rag import _format_context, _llm, _text_of, answer
from app.retriever import search
from eval.datasets import load_questions
```

verdict: 
also: 

---

## 23. 검색된 문서들을 LLM에 전달하기 전에 어떤 형식으로 묶어서 보여주는 거야?

- 라벨: `app/rag.py` · 축 `code` · 어휘중복 0.038

```
# 파일: app/rag.py
# 심볼: _format_context
def _format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "?")
        sec = d.metadata.get("section", "")
        kind = "코드" if d.metadata.get("doc_type") == "code" else "문서"
        head = f"[근거 {i}] ({kind}) {src}" + (f" > {sec}" if sec else "")
        blocks.append(f"{head}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)
```

verdict: 
also: 

---

## 24. 검색 결과로 반환되는 문서 리스트는 어떤 기준으로 정렬되어서 나오게 되는 거야?

- 라벨: `eval/run_eval.py` · 축 `code` · 어휘중복 0.0

```
# 파일: eval/run_eval.py
# 심볼: retrieve_bm25
def retrieve_bm25(question: str, k: int) -> List[str]:
    docs, _, bm25 = _corpus()
    scores = np.asarray(bm25.get_scores(_tokenize(question)), dtype=np.float32)
    return [docs[i].metadata.get("source", "?") for i in np.argsort(-scores)[:k]]
```

verdict: 
also: 

---

## 25. 특정 모델을 사용할 수 있는지 확인하려면 문서만 믿지 말고 어떻게 검증하는 게 좋아?

- 라벨: `docs/engineering-notes.md` · 축 `doc` · 어휘중복 0.214

```
## 2. 모델이 "사라졌다" — 404 no longer available  
**증상**: `gemini-2.5-flash` 호출 시 404 NOT_FOUND, "no longer available to **new users**".  
**원인**: 기존 사용자에겐 남아 있어 문서엔 보이지만, 신규 발급 키에는 제공되지 않는 모델이었다.  
**해결**: API 로 실제 사용 가능한 모델을 나열 → 후보를 직접 호출해 통과하는 것만 채택.
품질·지연을 재서 `gemini-3.1-flash-lite` 선택(첫 토큰 1.4s, 3.5-flash 는 thinking 탓 ~20s).  
**교훈**: 모델 가용성은 키/계정마다 다르다. 문서보다 런타임 probe 가 진실.
```

verdict: 
also: 

---

## 26. 이번 업데이트에서 Gemini 무료 버전을 연동하면서 채팅 응답을 실시간으로 보여주기 위해 어떤 기술적 변화가 있었나요?

- 라벨: `git:477ecb5` · 축 `commit` · 어휘중복 0.095

```
# 커밋 477ecb5 (2026-07-14)
제목: feat(tier1): Gemini(무료) 연결 + 토큰 스트리밍 + 예시질문 칩/지식패널(/topics)

- config: gemini/openai/extractive provider 스위치 + active_llm 자동 폴백
- rag: stream_answer() 이벤트 제너레이터, 가독성·[n]인용 프롬프트
- main: POST /chat/stream(NDJSON), GET /topics
- web: 예시질문 칩, 지식패널, 실시간 스트리밍 렌더

변경 파일:
  - .claude/memory/progress.md
  - .env.example
  - README.md
  - _uv.txt
  - app/config.py
  - app/loader.py
  - app/main.py
  - app/rag.py
  - requirements.txt
  - web/index.html
```

verdict: 
also: 

---

## 27. test_module_head_captured 테스트 케이스는 어떤 의도로 작성된 거야?

- 라벨: `tests/test_code_loader.py` · 축 `code` · 어휘중복 0.25

```
# 파일: tests/test_code_loader.py
# 심볼: test_module_head_captured
def test_module_head_captured():
    """첫 def 이전(import·상수)은 (module) 청크로 보존 — 상수 검색 가치."""
    segs = _seg_map(SRC)
    assert "X = 1" in segs["(module)"]
```

verdict: 
also: 

---

## 28. 현재 데모 버전에서 대화 맥락을 기억하지 못하는 문제를 해결하려면 어떤 기능을 구현해야 해?

- 라벨: `docs/HANDOFF.md` · 축 `doc` · 어휘중복 0.061

```
### Phase 2 — 제품화  
- [ ] 2-1 **멀티턴 + 질의 재작성** (현재 단발 질의만 — 공개 데모의 최대 결함)
- [ ] 2-2 라우터: 단발 RAG vs 에이전트 자동 선택 (에이전트는 한 질문에 LLM 3~5회, 지연 ~10초)
- [ ] 2-3 Terraform + ECS + RDS 배포
- [ ] 2-4 관측 대시보드 + 비용 알람
- [ ] 2-5 웹 UI 개편
```

verdict: 
also: 

---

## 29. 애플리케이션이 처음 실행될 때 모델을 미리 불러오는 이유가 뭐야?

- 라벨: `app/main.py` · 축 `code` · 어휘중복 0.087

```
# 파일: app/main.py
# 심볼: lifespan
async def lifespan(app: FastAPI):
    """기동 시 임베딩 모델·Chroma 를 미리 적재(워밍업).

    지연 로딩이면 첫 질문이 ~18초 걸린다(HF 모델 로드). 미리 데워두면 1~2초.
    BM25 인덱스 구축도 여기서 함께 끝난다.
    """
    from app.retriever import search

    search("warmup")
    if settings.use_reranker:
        from app.reranker import warmup as rr_warmup

        rr_warmup()
    yield
```

verdict: 
also: 

---

## 30. eval/report.py에서 수행하는 비교 로직은 구체적으로 어떤 지표들의 변화를 감시하고 있어?

- 라벨: `git:ec842a5` · 축 `commit` · 어휘중복 0.233

```
# 커밋 ec842a5 (2026-09-01)
제목: feat(eval): CI 회귀 게이트 — 결과를 파일로 남기고 baseline 대비 하락을 자동 차단

측정치가 콘솔에만 찍히고 커밋 메시지로 옮겨 적히던 구조라 ①회귀를 자동으로 못 잡고
②숫자가 사람 손을 거쳐 틀려도 아무도 모른다. 결과를 기계용(JSON)·사람용(Markdown)으로
떨구고, 운영 파이프라인 행과 범위밖 거절률만 baseline 과 비교해 게이트한다.

- eval/report.py            EvalReport 직렬화 + compare() 회귀 판정(허용치 recall 1%·MRR 0.03)
- eval/questions.demo.json  demo 골든셋 20문항 + 범위밖 6 (전부 origin=manual, 원본에서 수기 라벨)
- eval/baselines/demo.json  게이트 기준선(demo 만 추적 — private 은 미스 문항 텍스트가 유출)
- .github/workflows/eval.yml  pytest → demo 인덱싱 → 게이트 → 리포트 아티팩트·잡 요약
  fetch-depth: 0 필수(git 이력이 지식원 축이라 얕은 클론이면 코퍼스가 달라진다)
- tests/test_report.py      게이트 로직 10건 고정(게이트가 조용히 망가지면 초록불로 회귀를 놓친다)

★ 측정 중 발견 — 평가표 행 이름이 잘못된 결론을 만들고 있었다(노트 #17).
  mmr_lambda=1.0 이면 MMR 은 no-op 인데 'hybrid+MMR' 행이 하이브리드보다 MRR 이 낮았다.
  실제 차이는 MMR 이 아니라 심볼 슬롯이었고, 이 장치는 축에 따라 부호가 반대다
  (문서질문 MRR 0.88→0.78 해로움 / 코드질문 0.44→0.53 이로움) → 의도 라우팅의 근거.
  행 이름을 '운영 파이프라인'으로 고치고 구성 요소를 범례로 명시. recall 은 동일하므로
  채택 판단은 안 바뀐다 — 바뀌는 건 그 공을 무엇에 돌리느냐다.

demo 기준선: 285청크 / 전체 recall@5 90% · 문서 100% · 코드 75% · 범위밖 거절 6/6
private 재확인: 93% 유지(회귀 없음). pytest 41개 통과.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019nZhqfMvSn5y45f35gJSS4

변경 파일:
  - .github/workflows/eval.yml
  - .gitignore
  - docs/HANDOFF.md
  - docs/engineering-notes.md
  - eval/baselines/demo.json
  - eval/datasets.py
  - eval/questions.demo.json
  - eval/report.py
  - eval/run_eval.py
  - tests/test_report.py
```

verdict: 
also: 

---

## 31. 이번에 작성된 엔지니어링 노트에는 어떤 기술적 고민들이 담겨 있어?

- 라벨: `git:9055ba9` · 축 `commit` · 어휘중복 0.24

```
# 커밋 9055ba9 (2026-07-15)
제목: docs: 엔지니어링 노트 — 구현 중 마주친 문제 14건 정리(증상/원인/해결/교훈)

포폴·면접 자산. 라이브러리를 붙였다가 아니라 재보고 판단한 과정 기록:
Gemini 3.x 멀티파트 응답, MMR 이 하이브리드 무효화, 리랭커 측정 후 비활성화,
BM25 게이트 부적격, 코드 심볼 검색, MMR λ 재측정, LLM-judge 공정성 등.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

변경 파일:
  - docs/engineering-notes.md
```

verdict: 
also: 

---

## 32. RATE_CAP 설정처럼 문서와 코드가 일치하지 않는 상황을 마주하면 어떻게 대응하는 게 좋아?

- 라벨: `docs/engineering-notes.md` · 축 `doc` · 어휘중복 0.333

```
## 11. 문서와 코드가 어긋날 때 — 무엇을 진실로?  
**상황**: `RATE_CAP` 이 문서엔 남아 있으나 코드에선 제거됨(설정 체계 개편 때).  
**해결**: 프롬프트에 "문서와 코드가 어긋나면 명시하고, **실제 동작은 코드를 기준**으로" 지시.
지식원에 코드·git 이력을 함께 넣었기에 가능한 판단.  
**교훈**: 다중 소스 RAG 는 소스 간 신뢰 우선순위를 정해야 한다. 실행 진실은 코드에 있다.  
---
```

verdict: 
also: 

---

## 33. 현재 설정된 프로필에서 평가용 질문 데이터가 저장된 위치를 가져오려면 어떤 함수를 호출해야 해?

- 라벨: `eval/datasets.py` · 축 `code` · 어휘중복 0.083

```
# 파일: eval/datasets.py
# 심볼: questions_path
def questions_path() -> Path:
    """활성 프로필의 평가셋 경로. 없으면 형식만 담은 템플릿으로 폴백."""
    p = active_profile().eval_questions
    return p if p.exists() else TEMPLATE
```

verdict: 
also: 

---

## 34. 현재 데이터셋에 포함된 문서나 코드, 멀티홉 질문이 각각 몇 개씩 있는지 확인하려면 어떤 메서드를 호출해야 해?

- 라벨: `eval/datasets.py` · 축 `code` · 어휘중복 0.132

```
# 파일: eval/datasets.py
# 심볼: QuestionSet.summary
def summary(self) -> str:
        origins = ", ".join(f"{k} {v}" for k, v in self.origin_counts().items())
        return (
            f"[{self.profile}] {self.path.name}: 문서 {len(self.in_scope)} · "
            f"코드 {len(self.in_scope_code)} · 멀티홉 {len(self.multihop)} · "
            f"범위밖 {len(self.out_of_scope)} (라벨: {origins})"
        )
```

verdict: 
also: 

---

## 35. 에이전트가 작업을 수행하면서 참고한 문서 조각들을 어떤 방식으로 관리하고 있어?

- 라벨: `app/agent_graph.py` · 축 `code` · 어휘중복 0.032

```
# 파일: app/agent_graph.py
# 심볼: AgentState
class AgentState(TypedDict):
    """그래프가 들고 다니는 상태.

    messages 만 add_messages 리듀서로 누적하고, 나머지는 노드가 통째로 새 값을 돌려준다.
    """

    messages: Annotated[list, add_messages]
    collected: List[Document]  # 인용 순서대로 모인 근거 청크
    seen: List[tuple]          # (source, section) — 같은 청크 재채번 방지
    trace: List[Dict[str, Any]]
    llm_calls: int
    rounds: int
```

verdict: 
also: 

---

## 36. app/agent_graph.py의 answer 함수에서 예외가 발생했을 때 에러 메시지는 어떻게 구성돼?

- 라벨: `app/agent_graph.py` · 축 `code` · 어휘중복 0.214

```
# 파일: app/agent_graph.py
# 심볼: answer
try:
        final = _graph().invoke(init, config={"recursion_limit": limit})
        text = _text_of(final["messages"][-1].content) or OUT_OF_SCOPE
        mode = "agent_graph"
    except GraphRecursionError:
        return {
            "answer": "재귀 상한에 도달해 답변을 마무리하지 못했습니다.",
            "sources": [],
            "mode": "agent_graph_recursion",
            "trace": [],
            "steps": settings.agent_max_steps,
            "llm_calls": 0,
        }
    except Exception as e:  # noqa: BLE001 — 호출 실패는 그대로 알린다
        return {
            "answer": f"에이전트(LangGraph) 호출 오류: {e}",
            "sources": [],
            "mode": "agent_graph_error",
            "trace": [],
            "steps": 0,
            "llm_calls": 0,
        }

    return {
        "answer": text,
        "sources": snippets_for(final["collected"], question),
        "mode": mode,
        "trace": final["trace"],
        "steps": len(final["trace"]),
        "llm_calls": final["llm_calls"],
    }
```

verdict: 
also: 

---

## 37. 현재 활성화된 프로필 설정에 따라 파일 경로를 보여주는 방식이 어떻게 달라져?

- 라벨: `app/code_loader.py` · 축 `code` · 어휘중복 0.138

```
# 파일: app/code_loader.py
# 심볼: _display_name
def _display_name(path: Path) -> str:
    """'app/worker/worker_main.py' 처럼 최상위 패키지명을 남긴 상대경로(프로필이 결정)."""
    prof = active_profile()
    return prof.code.display_name(path) if prof.index_code else path.name
```

verdict: 
also: 

---

## 38. 평가 결과를 마크다운 형식으로 변환할 때 어떤 정보들이 상단에 요약되어 표시돼?

- 라벨: `eval/report.py` · 축 `code` · 어휘중복 0.033

```
# 파일: eval/report.py
# 심볼: EvalReport.to_markdown
def to_markdown(self) -> str:
        L: List[str] = []
        L.append(f"# 검색 평가 리포트 — `{self.profile}` 프로필")
        L.append("")
        L.append(f"- 커밋 `{self.git_sha}` · {self.created_at} · k={self.k}")
        corpus = " / ".join(f"{k} {v}" for k, v in self.corpus.items())
        L.append(f"- 코퍼스: **{sum(self.corpus.values())}청크** ({corpus}) · 컬렉션 `{self.collection}`")
        origins = self.dataset.get("origins") or {}
        L.append(f"- 평가셋: {self.dataset.get('path')} — 라벨 출처 {origins or '미기재'}")
        L.append("")
        for s in self.suites:
            L.append(f"## {s.title} — {s.n}문항")
            L.append("")
            L.append("| retriever | recall@k | MRR | miss |")
            L.append("|---|---:|---:|---:|")
            for r in s.rows:
                mark = "**" if r.name == PRIMARY_ROW else ""
                L.append(f"| {mark}{r.name}{mark} | {r.recall:.0%} | {r.mrr:.2f} | {len(r.misses)} |")
            L.append("")
```

verdict: 
also: 

---

## 39. _display_name 함수는 어떤 기준으로 파일의 이름을 반환하는 거야?

- 라벨: `app/code_loader.py` · 축 `code` · 어휘중복 0.182

```
# 파일: app/code_loader.py
# 심볼: _display_name
def _display_name(path: Path) -> str:
    """'app/worker/worker_main.py' 처럼 최상위 패키지명을 남긴 상대경로(프로필이 결정)."""
    prof = active_profile()
    return prof.code.display_name(path) if prof.index_code else path.name
```

verdict: 
also: 

---

## 40. 이번에 메인 브랜치로 에이전트 관련 코드를 합친 이유가 뭐야?

- 라벨: `git:8d40e5a` · 축 `commit` · 어휘중복 0.333

```
# 커밋 8d40e5a (2026-08-05)
제목: merge: v2-agent — 툴콜링 에이전트 레이어 + 멀티홉 평가셋

공개 레포 링크를 지원서에 제출하므로 기본 브랜치에 에이전트 코드가
보여야 한다. 이력서에 기재한 '멀티홉 정답률 42%→50%' 의 근거(app/agent.py,
eval 멀티홉 AND 채점)가 main 에 없으면 링크를 타고 온 사람이 확인할 수 없다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

```

verdict: 
also: 

---

## 41. reranker에서 사용하는 모델의 최대 토큰 제한은 어느 정도로 설정되어 있나요?

- 라벨: `app/reranker.py` · 축 `code` · 어휘중복 0.08

```
# 파일: app/reranker.py
# 심볼: _model
def _model():
    from sentence_transformers import CrossEncoder

    # max_length: 코드 청크는 토큰이 길다. 512 를 넘으면 뒤가 잘린다.
    return CrossEncoder(settings.reranker_model, max_length=512)
```

verdict: 
also: 

---

## 42. test_out_of_scope_rejected 테스트 케이스는 검색 엔진이 관련 없는 질의를 받았을 때 어떤 상태를 반환하는지 확인하는 거야?

- 라벨: `tests/test_retrieval.py` · 축 `code` · 어휘중복 0.184

```
# 파일: tests/test_retrieval.py
# 심볼: test_out_of_scope_rejected
def test_out_of_scope_rejected():
    """지식원과 무관한 질문은 검색 단계에서 거절(빈 결과)."""
    from app.retriever import search

    docs, debug = search("고양이 키우는 법 알려줘")
    assert docs == []
    assert debug["reason"] == "out_of_scope"
```

verdict: 
also: 

---

## 43. 코드 로더가 특정 함수를 올바르게 분할했는지 확인하는 테스트는 어떤 로직으로 검증해?

- 라벨: `tests/test_code_loader.py` · 축 `code` · 어휘중복 0.0

```
# 파일: tests/test_code_loader.py
# 심볼: test_function_is_own_chunk
def test_function_is_own_chunk():
    segs = _seg_map(SRC)
    assert "def foo" in segs["foo"]
    assert "return a + 1" in segs["foo"]
```

verdict: 
also: 

---

## 44. _mmr 함수에서 적합도 점수를 계산할 때 왜 코사인 유사도 대신 RRF 융합 점수를 사용하는 거야?

- 라벨: `app/retriever.py` · 축 `code` · 어휘중복 0.517

```
# 파일: app/retriever.py
# 심볼: _mmr
def _mmr(cands: List[int], embs: np.ndarray, rel: Dict[int, float], k: int, lam: float) -> List[int]:
    """Maximal Marginal Relevance — 적합도와 '이미 고른 것과의 차별성'을 lam 으로 절충.

    적합도 항은 **RRF 융합 점수**를 쓴다. 여기서 코사인을 쓰면 BM25 가 끌어올린
    희귀 식별자 문서가 최종 선택에서 다시 탈락해 하이브리드가 무의미해진다.
    임베딩은 중복 penalty(청크 간 유사도) 계산에만 쓴다.
    """
    # rel 에 없는 후보(심볼 매칭으로 뒤늦게 합류한 청크)는 적합도 0 으로 본다.
    top = (max(rel.values()) if rel else 0.0) or 1.0
    selected: List[int] = []
    pool = list(cands)
    while pool and len(selected) < k:
        if not selected:
            best = max(pool, key=lambda i: rel.get(i, 0.0))
        else:
            chosen = embs[selected]  # (n_sel, dim)
            best = max(
                pool,
                key=lambda i: lam * (rel.get(i, 0.0) / top)
                - (1 - lam) * float(np.max(chosen @ embs[i])),
            )
        selected.append(best)
        pool.remove(best)
    return selected
```

verdict: 
also: 

---

## 45. 리포트를 마크다운으로 변환했을 때, 운영 데이터와 코퍼스 정보가 제대로 포함되는지 확인하려면 어떤 테스트 코드를 보면 돼?

- 라벨: `tests/test_report.py` · 축 `code` · 어휘중복 0.065

```
# 파일: tests/test_report.py
# 심볼: test_markdown_에_운영행과_코퍼스가_찍힌다
def test_markdown_에_운영행과_코퍼스가_찍힌다():
    md = _report().to_markdown()
    assert rp.PRIMARY_ROW in md and "285청크" in md and "manual" in md
```

verdict: 
also: 

---

## 46. 데이터를 인덱싱할 때 코드 파일이랑 커밋 로그가 각각 몇 개씩 포함됐는지 어떻게 확인할 수 있어?

- 라벨: `app/ingest.py` · 축 `code` · 어휘중복 0.161

```
# 파일: app/ingest.py
# 심볼: build_index
print(f"[ingest] 임베딩 제공자: {settings.embedding_provider} — 임베딩 계산 중...")

    # 청크가 많아 한 번에 넣으면 Chroma 배치 상한(약 5461)에 걸린다 → 나눠서 add.
    batch = 2000
    for i in range(0, len(docs), batch):
        store.add_documents(docs[i : i + batch])
        print(f"[ingest]   … {min(i + batch, len(docs))}/{len(docs)}")

    n_code = sum(1 for d in docs if d.metadata.get("doc_type") == "code")
    n_commit = sum(1 for d in docs if d.metadata.get("doc_type") == "commit")
    n_doc = len(docs) - n_code - n_commit
    print(
        f"[ingest] 완료 — 청크 {len(docs)}개(문서 {n_doc} / 코드 {n_code} / 커밋 {n_commit})를 "
        f"'{prof.collection_name}' 에 인덱싱"
    )
    return len(docs)
```

verdict: 
also: 

---

## 47. 무료 사용자를 위한 요청 제한 정책이 어떻게 구현되어 있는지 궁금해.

- 라벨: `app/agent.py` · 축 `code` · 어휘중복 0.083

```
# 파일: app/agent.py
# 심볼: _throttle
def _throttle() -> None:
    """무료 티어(분당 15요청) 보호. 에이전트는 한 질문에 3~5회 호출한다."""
    wait = settings.agent_throttle_sec - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()
```

verdict: 
also: 

---

## 48. test_snake_case_split 테스트 케이스는 검색 성능을 확인하기 위해 어떤 검증 과정을 거치고 있어?

- 라벨: `tests/test_tokenizer.py` · 축 `code` · 어휘중복 0.167

```
# 파일: tests/test_tokenizer.py
# 심볼: test_snake_case_split
def test_snake_case_split():
    """snake_case 식별자는 전체 + 조각으로 색인 → 부분 질의도 매칭."""
    toks = _tokenize("apply_retry_policy")
    assert "apply_retry_policy" in toks   # 전체
    assert "retry" in toks
```

verdict: 
also: 

---

## 49. app/profiles.py의 register 데코레이터는 내부적으로 빌더를 어떻게 관리하고 있어?

- 라벨: `app/profiles.py` · 축 `code` · 어휘중복 0.346

```
# 파일: app/profiles.py
# 심볼: register
def register(name: str) -> Callable[[Callable[[Settings], CorpusProfile]], Callable]:
    """프로필 빌더 등록 데코레이터. 새 코퍼스 축은 여기 하나만 추가하면 된다."""

    def deco(fn: Callable[[Settings], CorpusProfile]):
        _BUILDERS[name] = fn
        return fn

    return deco
```

verdict: 
also: 

---
