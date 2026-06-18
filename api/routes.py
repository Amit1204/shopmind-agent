"""All FastAPI route definitions."""
import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.models import (
    ChatRequest, ChatResponse,
    ProductSearchRequest, VisualSearchResponse, HealthResponse
)
from agents.orchestrator import ShopMindAgent
from tools.product_lookup import product_lookup
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Agent singleton (initialised once on startup)
_agent: ShopMindAgent | None = None


def get_agent() -> ShopMindAgent:
    global _agent
    if _agent is None:
        _agent = ShopMindAgent()
    return _agent


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        llm_available=True,
        index_loaded=os.path.exists("./outputs/faiss_index/product_images.index")
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main conversational endpoint — runs the full agent."""
    try:
        agent = get_agent()
        result = await agent.run(
            query=req.message,
            user_id=req.user_id,
            conversation_history=req.conversation_history
        )
        return ChatResponse(
            answer=result.answer,
            confidence=result.confidence,
            tools_used=result.tools_used,
            evidence=result.evidence,
            safety_passed=result.safety_passed
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/visual-search", response_model=VisualSearchResponse)
async def visual_search_endpoint(image: UploadFile = File(...)):
    """Upload a product image → find visually similar products."""
    try:
        # Save uploaded image to temp file
        suffix = os.path.splitext(image.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(image.file, tmp)
            tmp_path = tmp.name

        agent = get_agent()
        result = await agent.run(
            query=f"Find products visually similar to this image",
            image_path=tmp_path
        )
        os.unlink(tmp_path)

        return VisualSearchResponse(
            products=result.evidence,
            confidence=result.confidence
        )
    except Exception as e:
        logger.error(f"Visual search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
def product_search(req: ProductSearchRequest):
    """Direct product catalog search (bypasses agent — faster)."""
    filters = req.filters.model_dump(exclude_none=True) if req.filters else {}
    result = product_lookup.invoke({"query": req.query, "filters": filters})
    return result
