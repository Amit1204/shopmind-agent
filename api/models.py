"""Pydantic request/response models for FastAPI."""
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's query")
    user_id: str = Field(default="anonymous")
    conversation_history: list = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    tools_used: list[str]
    evidence: list[str]
    safety_passed: bool


class ProductFilter(BaseModel):
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    min_rating: Optional[float] = None


class ProductSearchRequest(BaseModel):
    query: str
    filters: Optional[ProductFilter] = None
    top_k: int = Field(default=10, ge=1, le=50)


class VisualSearchResponse(BaseModel):
    products: list[dict]
    confidence: float


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    llm_available: bool
    index_loaded: bool
