"""
FastAPI application entrypoint.
Run: uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="ShopMind Agent API",
    description="Multimodal agentic AI for e-commerce",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Allow Streamlit frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routes under /api prefix
app.include_router(router, prefix="/api")


@app.on_event("startup")
async def startup():
    logger.info("ShopMind API starting up...")
    # Pre-load models on startup so first request isn't slow
    from services.embedding_service import embedding_service
    embedding_service.initialize()
    logger.info("ShopMind API ready ✓")


@app.on_event("shutdown")
async def shutdown():
    logger.info("ShopMind API shutting down")
