"""임베딩 제공자 선택 (openai | hf 로컬)."""
from __future__ import annotations

from app.config import settings


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

    raise ValueError(f"알 수 없는 EMBEDDING_PROVIDER: {provider}")
