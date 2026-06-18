"""
LLM factory — returns the configured LLM.
Priority: Groq (free) → Ollama (local) → OpenAI (fallback)
Single place to swap models without changing agent/tool code.
"""
from functools import lru_cache
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm():
    """Returns the configured LLM. Cached — only loaded once."""

    # 1. Groq (free, fast — preferred)
    if settings.groq_api_key:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0.1,
        )
        logger.info(f"Using Groq: {settings.groq_model}")
        return llm

    # 2. Local Ollama
    if settings.use_local_llm:
        try:
            from langchain_community.llms import Ollama
            llm = Ollama(
                base_url=settings.ollama_base_url,
                model=settings.local_model_name,
                temperature=0.1,
            )
            logger.info(f"Using local LLM: {settings.local_model_name} via Ollama")
            return llm
        except Exception as e:
            logger.warning(f"Ollama not available ({e}) — falling back to OpenAI")

    # 3. OpenAI fallback
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.1,
        api_key=settings.openai_api_key,
    )
    logger.info(f"Using OpenAI: {settings.openai_model}")
    return llm
