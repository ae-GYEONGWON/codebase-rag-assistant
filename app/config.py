"""환경설정 로드 (.env → pydantic-settings)."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: str = ""

    # 지식원
    knowledge_dirs: str = "D:/stock_prod/.claude/memory"
    file_globs: str = "*.md"

    # 임베딩
    embedding_provider: str = "openai"          # openai | hf
    openai_embedding_model: str = "text-embedding-3-small"
    hf_embedding_model: str = "jhgan/ko-sroberta-multitask"

    # LLM
    llm_provider: str = "openai"                # openai | extractive
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
    def has_openai(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.startswith("sk-"))


settings = Settings()
