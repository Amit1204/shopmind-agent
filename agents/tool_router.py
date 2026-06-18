"""
Tool Router — lightweight intent classifier that pre-selects relevant tools
before the ReAct agent runs. Avoids unnecessary LLM tool-call cycles.
Uses zero-shot classification (no training needed to start).
"""
from transformers import pipeline
from utils.logger import get_logger

logger = get_logger(__name__)

# Intent → tools mapping
INTENT_TOOL_MAP = {
    "visual_search":      ["visual_search"],
    "product_search":     ["product_lookup", "fake_review_filter"],
    "review_question":    ["review_qa", "fake_review_filter"],
    "sentiment_check":    ["aspect_sentiment", "fake_review_filter"],
    "price_comparison":   ["price_compare"],
    "recommendation":     ["recommend", "fake_review_filter"],
    "multi_intent":       None,  # None = use all tools (let agent decide)
}

INTENT_LABELS = list(INTENT_TOOL_MAP.keys())

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-MiniLM2-L6-H768",
            device=-1  # CPU
        )
        logger.info("Tool router classifier loaded")
    return _classifier


def route(query: str, has_image: bool = False) -> list[str]:
    """
    Given a user query, return the list of tool names the agent should consider.
    Returns None if all tools should be available (complex/multi-intent query).
    """
    if has_image:
        return ["visual_search", "product_lookup", "fake_review_filter", "recommend"]

    try:
        clf = _get_classifier()
        result = clf(query, candidate_labels=INTENT_LABELS)
        top_intent = result["labels"][0]
        top_score = result["scores"][0]

        # Low confidence → don't restrict, let agent use all tools
        if top_score < 0.45:
            logger.info(f"Router: low confidence ({top_score:.2f}) → all tools")
            return None

        tools = INTENT_TOOL_MAP.get(top_intent)
        logger.info(f"Router: intent={top_intent} ({top_score:.2f}) → tools={tools}")
        return tools

    except Exception as e:
        logger.warning(f"Tool router error: {e} — defaulting to all tools")
        return None
