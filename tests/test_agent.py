"""Integration tests for the orchestrator agent."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_agent_returns_response_structure():
    """Agent run() must return AgentResponse with required fields."""
    with patch("agents.orchestrator.get_llm") as mock_llm:
        mock_llm.return_value = MagicMock()
        from agents.orchestrator import ShopMindAgent, AgentResponse
        agent = ShopMindAgent()

        # Mock the graph invocation
        with patch.object(agent, "_graph") as mock_graph:
            from langchain_core.messages import AIMessage
            mock_graph.ainvoke = AsyncMock(return_value={
                "messages": [AIMessage(content="Test answer")]
            })
            result = await agent.run("Find me earphones", user_id="test_user")

        assert isinstance(result, AgentResponse)
        assert isinstance(result.answer, str)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.tools_used, list)
        assert isinstance(result.safety_passed, bool)
