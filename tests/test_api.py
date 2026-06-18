"""FastAPI endpoint tests using TestClient."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.fixture
def client():
    with patch("agents.orchestrator.get_llm", return_value=MagicMock()):
        from api.app import app
        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_chat_endpoint_structure(client):
    with patch("api.routes.get_agent") as mock_agent_factory:
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MagicMock(
            answer="Test answer",
            confidence=0.85,
            tools_used=["product_lookup"],
            evidence=[],
            safety_passed=True
        ))
        mock_agent_factory.return_value = mock_agent

        response = client.post("/api/chat", json={
            "message": "Find earphones under 2000",
            "user_id": "test_user"
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert "tools_used" in data
        assert "safety_passed" in data
