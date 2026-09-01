"""환경설정 로드 (.env → pydantic-settings)."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API 키
    google_api_key: str = ""     # Gemini (무료 티어) — https://aistudio.google.com/apikey
    openai_api_key: str = ""     # (선택) OpenAI 를 쓸 경우에만

    # --- 코퍼스 프로필 ---
    # 어떤 지식원을 인덱싱할지 한 덩어리로 고른다(app/profiles.py).
    #   demo    = 이 저장소 자기 자신(git 추적 파일). 어느 PC·CI 에서든 동일 → 기본값.
    #   private = 아래 KNOWLEDGE_DIRS/CODE_DIRS/GIT_REPOS 로 지정한 외부 코드베이스.
    # ※ 기본을 demo 로 둬야 clone 직후 아무 설정 없이 인덱싱·평가가 돌아간다.
    corpus_profile: str = "demo"

    # 지식원 — 문서 (private 프로필 전용 입력)
    knowledge_dirs: str = "/path/to/your/repo/docs"
    file_globs: str = "*.md"

    # 지식원 — 소스코드(AST 청킹). 문서가 말하지 않는 '실제 구현'을 답하게 하는 축.
    index_code: bool = True
    code_dirs: str = "/path/to/your/repo/app,/path/to/your/repo/scripts"
    code_globs: str = "*.py"

    # 지식원 — git 이력. "언제 왜 바뀌었나"를 답하는 축(최신성·폐기 인지).
    index_git: bool = True
    git_repos: str = "/path/to/your/repo"
    git_max_commits: int = 300

    # 임베딩 (기본: 로컬 무료 hf)
    embedding_provider: str = "hf"             # hf(로컬 무료) | openai
    openai_embedding_model: str = "text-embedding-3-small"
    hf_embedding_model: str = "jhgan/ko-sroberta-multitask"

    # LLM (생성) — 기본: Gemini 무료 티어
    llm_provider: str = "gemini"               # gemini | openai | extractive
    # flash-lite: 첫 토큰 ~1.4s (thinking 기본 off). 3.5-flash 는 품질 ↑ 지만 첫 토큰 ~20s.
    gemini_chat_model: str = "gemini-3.1-flash-lite"
    openai_chat_model: str = "gpt-4o-mini"

    # 분할/검색
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 5

    # 하이브리드 검색(BM25+벡터 RRF 융합 → MMR)
    fetch_k: int = 20            # 융합 전에 각 검색기가 가져올 후보 수
    # 1.0=적합도만(순수 RRF 순위), 0.0=다양성만. λ 스윕으로 결정.
    #   문서만(956청크)일 땐 0.8 이 최적이었으나, 코드+git 확장(3619청크) 후 재측정 결과
    #   0.9→86% / 0.95→89% / 1.0→93%. 확장 코퍼스에선 다양성이 정답을 밀어내 recall 저하.
    #   → 1.0 채택(다양성 off). 같은 파일 청크 독점은 심볼슬롯·출처 dedup 으로 별도 관리.
    mmr_lambda: float = 1.0

    # 리랭커(cross-encoder) — 하이브리드가 좁힌 rerank_candidates 개 후보만 재채점.
    # ★ 기본 OFF: eval 로 측정한 결과 이 코퍼스에선 하이브리드보다 나빴다.
    #   - bge-reranker-base: 코드 질문 recall 75%→50% (코드 청크 점수 ~0, 자연어 문서 선호)
    #   - bge-reranker-v2-m3: 분별력은 낫지만 "…코드에서?" 질문도 문서를 1위로 올리고
    #     CPU 4.4s/질문으로 데모 지연 과다. 둘 다 RRF 가 이미 뽑은 정답 코드파일을 문서로 뒤집음.
    #   원인: cross-encoder 가 자연어-자연어 매칭을 자연어-코드보다 선호 → "코드" 의도 유실.
    #   → 구현·토글은 유지(USE_RERANKER=true 로 실험 가능), 기본은 하이브리드 단독.
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_candidates: int = 20

    # 범위 밖 판정 임계(코사인). 실측 분포로 보정:
    #   범위 안 질문 0.40~0.72 / 범위 밖(날씨·요리·상식) 0.25~0.29
    #   → 0.35 로 두면 잡담은 검색 단계에서 컷. BM25 는 한글 2-gram 특성상
    #     범위 밖에도 16+ 가 나와 게이트로 못 쓰고 순위 매기기에만 쓴다.
    min_similarity: float = 0.35

    # --- 에이전트(툴콜링) 레이어 ---
    # 단발 RAG 는 "검색 1회 → 답변". 에이전트는 축(문서/코드/커밋)을 골라 여러 번 검색하고
    # 필요하면 심볼 본문을 다시 읽는다 → 멀티홉 질문("왜 바뀌었고 지금 코드는 어떻게 돼?")에 강함.
    # 대가는 LLM 호출 수 증가(지연·토큰·무료티어 RPM). 그 트레이드오프를 측정하는 게 v2 목표.
    use_agent: bool = False       # 기본 OFF — 단발 RAG 가 기존 검증된 경로
    agent_max_steps: int = 5      # 툴 호출 라운드 상한(무한루프·토큰폭주 방지)
    # Gemini 무료 티어는 분당 15요청. 에이전트는 한 질문에 3~5회 호출하므로 간격을 둔다.
    agent_throttle_sec: float = 4.0

    # 벡터 DB
    chroma_dir: str = "./chroma_db"
    collection_name: str = "codebase_memory"

    # 사용자 피드백 로그(👍/👎). 개인 로그라 git 제외(data/).
    feedback_log: str = "./data/feedback.jsonl"

    # --- 파생 헬퍼 ---
    @property
    def knowledge_dir_list(self) -> List[Path]:
        return [Path(p.strip()) for p in self.knowledge_dirs.split(",") if p.strip()]

    @property
    def glob_list(self) -> List[str]:
        return [g.strip() for g in self.file_globs.split(",") if g.strip()]

    @property
    def code_dir_list(self) -> List[Path]:
        return [Path(p.strip()) for p in self.code_dirs.split(",") if p.strip()]

    @property
    def code_glob_list(self) -> List[str]:
        return [g.strip() for g in self.code_globs.split(",") if g.strip()]

    @property
    def git_repo_list(self) -> List[Path]:
        return [Path(p.strip()) for p in self.git_repos.split(",") if p.strip()]

    @property
    def has_gemini(self) -> bool:
        return bool(self.google_api_key.strip())

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.startswith("sk-"))

    @property
    def active_llm(self) -> str:
        """설정된 provider 에 필요한 키가 없으면 extractive 로 자동 폴백."""
        p = self.llm_provider.lower()
        if p == "gemini" and self.has_gemini:
            return "gemini"
        if p == "openai" and self.has_openai:
            return "openai"
        return "extractive"


settings = Settings()
