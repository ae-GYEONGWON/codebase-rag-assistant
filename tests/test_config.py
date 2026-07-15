"""설정 파생 로직 — 특히 키 유무에 따른 LLM provider 자동 폴백."""
from app.config import Settings


def test_active_llm_gemini_with_key():
    s = Settings(google_api_key="AQ.xxxx", llm_provider="gemini")
    assert s.active_llm == "gemini"


def test_active_llm_falls_back_without_key():
    """키가 없으면 gemini 로 설정돼 있어도 extractive 로 폴백(크래시 대신 발췌)."""
    s = Settings(google_api_key="", llm_provider="gemini")
    assert s.active_llm == "extractive"


def test_openai_requires_sk_prefix():
    assert Settings(openai_api_key="sk-abc", llm_provider="openai").active_llm == "openai"
    assert Settings(openai_api_key="invalid", llm_provider="openai").active_llm == "extractive"


def test_dir_list_parsing():
    s = Settings(knowledge_dirs="a, b ,c")
    assert [p.name for p in s.knowledge_dir_list] == ["a", "b", "c"]


def test_glob_list_parsing():
    assert Settings(file_globs="*.md, *.txt").glob_list == ["*.md", "*.txt"]
