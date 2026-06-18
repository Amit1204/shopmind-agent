"""
Central configuration — all settings loaded from .env via pydantic-settings.
Import: from config.settings import settings
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM — Groq (free) takes priority when key is set
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Ollama (local fallback)
    ollama_base_url: str = "http://localhost:11434"
    local_model_name: str = "mistral:7b-instruct-q4_0"
    use_local_llm: bool = False         # set True to use Ollama instead

    # OpenAI (last resort fallback)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Observability
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "shopmind-agent"
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Vector DB
    chroma_persist_dir: str = "./outputs/chroma_db"
    faiss_index_path: str = "./outputs/faiss_index"

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Model paths (set after training)
    absa_model_path: str = "./outputs/models/absa_model"
    fake_review_model_path: str = "./outputs/models/fake_review_model.joblib"

    # Data
    products_csv: str = "./data/datasets/products.csv"
    reviews_csv: str = "./data/datasets/reviews.csv"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
