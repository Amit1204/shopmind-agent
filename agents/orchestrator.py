"""
ShopMind Orchestrator Agent
LangGraph ReAct agent that orchestrates all 7 tools.
Features:
  - Tool routing (pre-filters tools per query)
  - Parallel tool execution (price + sentiment + Q&A simultaneously)
  - Confidence aggregation
  - Short-term memory (conversation buffer)
  - Safety validation before responding
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from tools.visual_search import visual_search
from tools.review_qa import review_qa
from tools.product_lookup import product_lookup
from tools.aspect_sentiment import aspect_sentiment
from tools.fake_review_filter import fake_review_filter
from tools.price_compare import price_compare
from tools.recommend import recommend
from agents.tool_router import route
from agents.llm_factory import get_llm
from services.safety_validator import safety_validator
from utils.logger import get_logger

logger = get_logger(__name__)

ALL_TOOLS = [
    visual_search,
    review_qa,
    product_lookup,
    aspect_sentiment,
    fake_review_filter,
    price_compare,
    recommend,
]

SYSTEM_PROMPT = """You are ShopMind, an intelligent e-commerce assistant.
You MUST use your tools to answer every user request. Never respond without calling at least one tool first.

Available tools:
- product_lookup: search for products by name, category, or description — USE THIS for any product search
- review_qa: answer questions using real product reviews
- aspect_sentiment: get sentiment scores for product aspects (battery, screen, sound, etc.)
- fake_review_filter: check if product reviews are trustworthy
- price_compare: compare product prices
- recommend: get personalised product recommendations
- visual_search: find products similar to an uploaded image

Rules:
1. ALWAYS call product_lookup first when a user asks to find, search, or compare products.
2. ALWAYS call at least one tool — never answer from memory alone.
3. After getting tool results, summarise them clearly for the user.
4. If a tool returns no results, say so and suggest alternatives.
"""


@dataclass
class AgentState:
    messages: list = field(default_factory=list)
    user_id: str = "anonymous"
    image_path: str | None = None
    tool_results: dict = field(default_factory=dict)
    final_response: str = ""
    confidence: float = 0.0


@dataclass
class AgentResponse:
    answer: str
    confidence: float
    tools_used: list[str]
    evidence: list[str]
    safety_passed: bool


class ShopMindAgent:

    def __init__(self):
        self.llm = get_llm()
        self._graph = create_react_agent(
            self.llm,
            ALL_TOOLS,
            prompt=SYSTEM_PROMPT
        )
        logger.info("ShopMindAgent initialized")

    async def run(self, query: str, user_id: str = "anonymous",
                  image_path: str = None, conversation_history: list = None) -> AgentResponse:
        """Main entry point — run the agent on a user query."""
        logger.info(f"Agent run: user={user_id}, query={query[:60]}")

        # Step 1: Tool routing
        allowed_tools = route(query, has_image=bool(image_path))
        if allowed_tools:
            logger.info(f"Router restricted tools to: {allowed_tools}")

        # Step 2: Build message history
        messages = conversation_history or []
        if image_path:
            query = f"[Image provided: {image_path}]\n{query}"
        messages.append(HumanMessage(content=query))

        # Step 3: Run LangGraph ReAct agent
        config = RunnableConfig(recursion_limit=50)
        result = await self._graph.ainvoke({"messages": messages}, config=config)

        final_message = result["messages"][-1]
        answer = final_message.content if hasattr(final_message, "content") else str(final_message)

        # Step 4: Extract tool usage from message history
        tools_used = []
        evidence = []
        confidence_scores = []
        products_in_response = []

        for msg in result["messages"]:
            if hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls:
                    tools_used.append(tc["name"])
            if hasattr(msg, "content") and isinstance(msg.content, dict):
                if "confidence" in msg.content:
                    confidence_scores.append(msg.content["confidence"])
                if "evidence" in msg.content:
                    evidence.extend(msg.content["evidence"])

        # Step 5: Aggregate confidence
        overall_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores else 0.7
        )

        # Step 6: Safety validation
        safety_result = safety_validator.validate(products_in_response, answer)

        if not safety_result.passed:
            answer += f"\n\n⚠️ Note: {safety_result.message}"

        return AgentResponse(
            answer=answer,
            confidence=round(overall_confidence, 3),
            tools_used=list(set(tools_used)),
            evidence=evidence[:10],
            safety_passed=safety_result.passed
        )

    def run_sync(self, query: str, **kwargs) -> AgentResponse:
        """Synchronous wrapper for non-async contexts."""
        return asyncio.get_event_loop().run_until_complete(self.run(query, **kwargs))
