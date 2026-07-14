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

    # 지식원
    knowledge_dirs: str = "D:/stock_prod/.claude/memory"
    file_globs: str = "*.md"

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
