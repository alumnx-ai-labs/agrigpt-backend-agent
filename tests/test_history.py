
import sys
from unittest.mock import MagicMock, patch, call

# Mock heavy dependencies before importing app
mock_mongo_module = MagicMock()
sys.modules["pymongo"] = mock_mongo_module
sys.modules["pymongo.collection"] = MagicMock()
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
    from app import load_history, save_history

import pytest
from langchain_core.messages import HumanMessage, AIMessage


# ── load_history ──────────────────────────────────────────────────────────────

def test_load_history_no_doc():
    with patch.object(app.chat_sessions, "find_one", return_value=None):
        result = load_history("nonexistent-chat")
    assert result == []


def test_load_history_doc_without_messages():
    with patch.object(app.chat_sessions, "find_one", return_value={"chat_id": "x"}):
        result = load_history("x")
    assert result == []


def test_load_history_reconstructs_human_and_ai():
    doc = {
        "chat_id": "abc",
        "messages": [
            {"role": "human", "content": "What crop grows in sandy soil?"},
            {"role": "ai",    "content": "Groundnut grows well in sandy soil."},
        ],
    }
    with patch.object(app.chat_sessions, "find_one", return_value=doc):
        result = load_history("abc")

    assert len(result) == 2
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "What crop grows in sandy soil?"
    assert isinstance(result[1], AIMessage)
    assert result[1].content == "Groundnut grows well in sandy soil."


def test_load_history_skips_unknown_roles():
    doc = {
        "chat_id": "z",
        "messages": [
            {"role": "human",   "content": "Hi"},
            {"role": "unknown", "content": "???"},
            {"role": "ai",      "content": "Hello"},
        ],
    }
    with patch.object(app.chat_sessions, "find_one", return_value=doc):
        result = load_history("z")

    roles = [type(m).__name__ for m in result]
    assert "HumanMessage" in roles
    assert "AIMessage" in roles
    assert len(result) == 2  # unknown role skipped


# ── save_history ──────────────────────────────────────────────────────────────

def test_save_history_upserts_with_correct_chat_id():
    messages = [
        HumanMessage(content="When to sow wheat?"),
        AIMessage(content="Sow wheat in October–November."),
    ]
    with patch.object(app.chat_sessions, "update_one") as mock_update:
        save_history("chat-99", messages, phone_number="919876543210")

    mock_update.assert_called_once()
    filter_arg, update_arg = mock_update.call_args[0]
    assert filter_arg == {"chat_id": "chat-99"}
    assert update_arg["$set"]["messages"][0]["role"] == "human"
    assert update_arg["$set"]["messages"][1]["role"] == "ai"
    assert update_arg["$set"]["phone_number"] == "919876543210"
    assert mock_update.call_args[1]["upsert"] is True


def test_save_history_omits_phone_when_none():
    messages = [HumanMessage(content="Hello"), AIMessage(content="Hi")]
    with patch.object(app.chat_sessions, "update_one") as mock_update:
        save_history("chat-1", messages, phone_number=None)

    _, update_arg = mock_update.call_args[0]
    assert "phone_number" not in update_arg["$set"]


def test_save_history_sliding_window_trims_to_max():
    # Build more messages than MAX_MESSAGES
    messages = []
    for i in range(15):
        messages.append(HumanMessage(content=f"Question {i}"))
        messages.append(AIMessage(content=f"Answer {i}"))

    with patch.object(app.chat_sessions, "update_one") as mock_update:
        save_history("chat-trim", messages)

    _, update_arg = mock_update.call_args[0]
    stored = update_arg["$set"]["messages"]
    assert len(stored) <= app.MAX_MESSAGES


def test_save_history_skips_tool_messages():
    from langchain_core.messages import ToolMessage
    messages = [
        HumanMessage(content="question"),
        ToolMessage(content="raw tool output", tool_call_id="t1"),
        AIMessage(content="real answer"),
    ]
    with patch.object(app.chat_sessions, "update_one") as mock_update:
        save_history("chat-tool", messages)

    _, update_arg = mock_update.call_args[0]
    stored = update_arg["$set"]["messages"]
    roles = [m["role"] for m in stored]
    assert "tool" not in roles
    assert len(stored) == 2
