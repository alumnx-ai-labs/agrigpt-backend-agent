
import sys
from unittest.mock import MagicMock, patch

# Mock dependencies before importing app
mock_mongo = MagicMock()
mock_client = MagicMock()
mock_mongo.return_value = mock_client

sys.modules["pymongo"] = MagicMock()
sys.modules["pymongo"].MongoClient = mock_mongo
sys.modules["pymongo"].ASCENDING = 1

sys.modules["langchain_google_genai"] = MagicMock()
sys.modules["langgraph"] = MagicMock()
sys.modules["langgraph.graph"] = MagicMock()
sys.modules["langgraph.prebuilt"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["google.genai.types"] = MagicMock()

import unittest.mock
with unittest.mock.patch("app.build_agent", return_value=MagicMock()):
    import app
    from app import app as fastapi_app

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    return TestClient(fastapi_app)


# ── /hello ────────────────────────────────────────────────────────────────────

def test_hello_endpoint(client):
    response = client.get("/hello")
    assert response.status_code == 200
    assert response.json() == {"message": "hello claude!!"}


# ── GET /webhook — WhatsApp verification ─────────────────────────────────────

def test_webhook_verification_success(client):
    params = {
        "hub.mode": "subscribe",
        "hub.verify_token": "test_verify_token_123",
        "hub.challenge": "123456",
    }
    response = client.get("/webhook", params=params)
    assert response.status_code == 200
    assert response.text == "123456"


def test_webhook_verification_wrong_token(client):
    params = {
        "hub.mode": "subscribe",
        "hub.verify_token": "wrong_token",
        "hub.challenge": "123456",
    }
    response = client.get("/webhook", params=params)
    assert response.status_code == 403


def test_webhook_verification_wrong_mode(client):
    params = {
        "hub.mode": "unsubscribe",
        "hub.verify_token": "test_verify_token_123",
        "hub.challenge": "abc",
    }
    response = client.get("/webhook", params=params)
    assert response.status_code == 403


def test_webhook_verification_missing_params(client):
    response = client.get("/webhook")
    assert response.status_code == 403


# ── POST /webhook — WhatsApp message handler ─────────────────────────────────

def test_webhook_post_no_messages(client):
    payload = {"entry": [{"changes": [{"value": {"messages": []}}]}]}
    response = client.post("/webhook", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_webhook_post_non_text_message(client):
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {"type": "image", "from": "1234567890"}
                            ]
                        }
                    }
                ]
            }
        ]
    }
    response = client.post("/webhook", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_webhook_post_text_message_enqueues(client):
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "type": "text",
                                    "from": "919876543210",
                                    "text": {"body": "What is the best fertilizer for wheat?"},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
    response = client.post("/webhook", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_webhook_post_empty_payload(client):
    response = client.post("/webhook", json={})
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_webhook_post_malformed_payload(client):
    response = client.post("/webhook", json={"entry": []})
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── POST /test/chat ───────────────────────────────────────────────────────────

def test_chat_endpoint_success(client):
    with (
        patch("app.load_history", return_value=[]),
        patch("app.app_agent") as mock_agent,
        patch("app.save_history"),
        patch("app.has_meaningful_tool_results", return_value=True),
        patch("app.extract_sources_from_tool_results", return_value=["crops.pdf"]),
        patch("app.clean_response_text", return_value="Wheat needs NPK fertilizer."),
    ):
        from langchain_core.messages import AIMessage
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Wheat needs NPK fertilizer.")],
            "tool_results": [{"tool": "sme_divesh", "result": {"sources": [{"filename": "crops.pdf"}]}}],
        }
        response = client.post(
            "/test/chat",
            json={"chatId": "abc-123", "phone_number": "919000000000", "message": "best fertilizer for wheat"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["chatId"] == "abc-123"
    assert data["phone_number"] == "919000000000"
    assert "response" in data
    assert "sources" in data


def test_chat_endpoint_missing_fields(client):
    response = client.post("/test/chat", json={"chatId": "x"})
    assert response.status_code == 422


def test_chat_endpoint_empty_message(client):
    with (
        patch("app.load_history", return_value=[]),
        patch("app.app_agent") as mock_agent,
        patch("app.save_history"),
        patch("app.has_meaningful_tool_results", return_value=False),
        patch("app.get_gemini_fallback", return_value=("General answer.", "success")),
        patch("app.clean_response_text", return_value="General answer."),
    ):
        from langchain_core.messages import AIMessage
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="")],
            "tool_results": [],
        }
        response = client.post(
            "/test/chat",
            json={"chatId": "xyz", "phone_number": "910000000000", "message": ""},
        )
    assert response.status_code == 200
