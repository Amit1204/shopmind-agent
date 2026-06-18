"""
LangSmith tracing configuration.
Automatically traces every LangGraph agent run when LANGCHAIN_TRACING_V2=true.
View traces at: https://smith.langchain.com
"""
import os
from config.settings import settings


def setup_langsmith():
    """Call this once at app startup to enable LangSmith tracing."""
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        print(f"✅ LangSmith tracing enabled → project: {settings.langchain_project}")
    else:
        print("⚠️  LangSmith tracing disabled (set LANGCHAIN_API_KEY in .env)")
