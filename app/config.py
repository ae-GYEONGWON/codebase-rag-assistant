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

    # 지식원 — 문서
    knowledge_dirs: str = "D:/stock_prod/.claude/memory"
    file_globs: str = "*.md"

    # 지식원 — 소스코드(AST 청킹). 문서가 말하지 않는 '실제 구현'을 답하게 하는 축.
    index_code: bool = True
    code_dirs: str = "D:/stock_prod/app,D:/stock_prod/scripts"
    code_globs: str = "*.py"

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
    # 1.0=적합도만, 0.0=다양성만. λ 스윕(eval/run_eval.py) 결과 0.8 채택:
    #   0.5~0.7 → recall 90% / 0.8~0.9 → recall 95%(RRF 단독과 동일)면서 출처 다양성 ↑
    mmr_lambda: float = 0.8

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

    # 벡터 DB
    chroma_dir: str = "./chroma_db"
    collection_name: str = "stock_prod_memory"

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
