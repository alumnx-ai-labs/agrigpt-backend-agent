
import sys
from unittest.mock import MagicMock

# Mock heavy dependencies before importing app
sys.modules.setdefault("pymongo", MagicMock())
sys.modules.setdefault("langchain_google_genai", MagicMock())
sys.modules.setdefault("langgraph", MagicMock())
sys.modules.setdefault("langgraph.graph", MagicMock())
sys.modules.setdefault("langgraph.prebuilt", MagicMock())
sys.modules.setdefault("google", MagicMock())
sys.modules.setdefault("google.genai", MagicMock())
sys.modules.setdefault("google.genai.types", MagicMock())

import unittest.mock
with unittest.mock.patch("app.build_agent", return_value=MagicMock()):
    import app
    from app import (
        extract_final_answer,
        extract_sources_from_result,
        clean_response_text,
        is_agriculture_query,
        has_meaningful_tool_results,
        extract_sources_from_tool_results,
    )

import pytest
from langchain_core.messages import AIMessage, HumanMessage

def test_extract_final_answer_string():
    result = {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hello there!")
        ]
    }
    assert extract_final_answer(result) == "Hello there!"

def test_extract_final_answer_list():
    result = {
        "messages": [
            AIMessage(content=[{"text": "Hello block"}])
        ]
    }
    assert extract_final_answer(result) == "Hello block"

def test_extract_final_answer_empty():
    result = {"messages": []}
    assert extract_final_answer(result) == "No response generated."

def test_extract_sources_from_result_pdf():
    # We need to simulate the AIMessage structure that the function expects
    msg = AIMessage(content="Based on the information in crop_disease.pdf, you should use fertilizer.")
    result = {"messages": [msg]}
    sources = extract_sources_from_result(result)
    assert "crop_disease.pdf" in sources

def test_extract_sources_from_result_multiple():
    msg = AIMessage(content="Check pestinfo.pdf and schemes.pdf for more details.")
    result = {"messages": [msg]}
    sources = extract_sources_from_result(result)
    assert "pestinfo.pdf" in sources
    assert "schemes.pdf" in sources

def test_extract_sources_from_result_labeled():
    msg = AIMessage(content="Source: monsoon_guide.pdf")
    result = {"messages": [msg]}
    sources = extract_sources_from_result(result)
    assert "monsoon_guide.pdf" in sources

def test_chat_request_validation():
    from app import ChatRequest
    # Valid request
    req = ChatRequest(chatId="123", phone_number="987", message="hello")
    assert req.chatId == "123"

    # Missing field
    with pytest.raises(ValueError):
        ChatRequest(chatId="123")


# ── clean_response_text ───────────────────────────────────────────────────────

def test_clean_response_text_strips_headers():
    assert clean_response_text("# Title\nsome text") == "Title\nsome text"
    assert clean_response_text("## Section\nbody") == "Section\nbody"


def test_clean_response_text_strips_bold():
    assert clean_response_text("Use **NPK** fertilizer") == "Use NPK fertilizer"


def test_clean_response_text_strips_inline_code():
    result = clean_response_text("Run `pip install` first")
    assert "`" not in result
    assert "pip install" in result


def test_clean_response_text_strips_code_block():
    text = "Example:\n```python\nprint('hi')\n```\nDone."
    result = clean_response_text(text)
    assert "```" not in result
    assert "Done." in result


def test_clean_response_text_strips_sources_section():
    text = "Use urea.\nSources: soil_guide.pdf"
    result = clean_response_text(text)
    assert "Sources:" not in result
    assert "Use urea." in result


def test_clean_response_text_empty_string():
    assert clean_response_text("") == ""


# ── is_agriculture_query ──────────────────────────────────────────────────────

def test_is_agriculture_query_true_for_crop():
    assert is_agriculture_query("How do I grow wheat?") is True


def test_is_agriculture_query_true_for_pest():
    assert is_agriculture_query("My mango tree has a pest problem") is True


def test_is_agriculture_query_true_for_fertilizer():
    assert is_agriculture_query("Which fertilizer is best for paddy?") is True


def test_is_agriculture_query_false_for_non_ag():
    assert is_agriculture_query("What is the capital of France?") is False


def test_is_agriculture_query_false_for_tech():
    assert is_agriculture_query("How do I fix a Python bug?") is False


def test_is_agriculture_query_case_insensitive():
    assert is_agriculture_query("IRRIGATION techniques") is True


# ── has_meaningful_tool_results ───────────────────────────────────────────────

def test_has_meaningful_tool_results_empty_list():
    assert has_meaningful_tool_results([]) is False


def test_has_meaningful_tool_results_error_status():
    results = [{"tool": "t", "result": {"status": "error", "message": "fail"}}]
    assert has_meaningful_tool_results(results) is False


def test_has_meaningful_tool_results_with_sources():
    results = [
        {
            "tool": "pests_and_diseases",
            "result": {
                "sources": [{"filename": "crop.pdf", "score": 0.9}]
            },
        }
    ]
    assert has_meaningful_tool_results(results) is True


def test_has_meaningful_tool_results_with_results_list():
    results = [{"tool": "sme", "result": {"results": [{"source": "guide.pdf"}]}}]
    assert has_meaningful_tool_results(results) is True


def test_has_meaningful_tool_results_with_direct_list():
    results = [{"tool": "vignan", "result": [{"answer": "Use NPK"}]}]
    assert has_meaningful_tool_results(results) is True


def test_has_meaningful_tool_results_no_useful_data():
    results = [{"tool": "t", "result": {"status": "ok", "sources": []}}]
    assert has_meaningful_tool_results(results) is False


# ── extract_sources_from_tool_results ─────────────────────────────────────────

def test_extract_sources_from_tool_results_empty():
    assert extract_sources_from_tool_results([]) == []


def test_extract_sources_from_tool_results_dict_filename():
    results = [
        {
            "tool": "pests",
            "result": {
                "sources": [
                    {"filename": "crop_disease.pdf", "score": 0.9},
                    {"filename": "pesticide_guide.pdf", "score": 0.8},
                ]
            },
        }
    ]
    sources = extract_sources_from_tool_results(results)
    assert "crop_disease.pdf" in sources
    assert "pesticide_guide.pdf" in sources


def test_extract_sources_from_tool_results_plain_string_sources():
    results = [{"tool": "t", "result": {"sources": ["manual.pdf", "guide.pdf"]}}]
    sources = extract_sources_from_tool_results(results)
    assert "manual.pdf" in sources
    assert "guide.pdf" in sources


def test_extract_sources_from_tool_results_results_field():
    results = [
        {"tool": "sme", "result": {"results": [{"source": "soil_health.pdf"}]}}
    ]
    sources = extract_sources_from_tool_results(results)
    assert "soil_health.pdf" in sources


def test_extract_sources_from_tool_results_stringified_json():
    import json
    inner = {"sources": [{"filename": "wheat_care.pdf"}]}
    results = [{"tool": "t", "result": json.dumps(inner)}]
    sources = extract_sources_from_tool_results(results)
    assert "wheat_care.pdf" in sources


def test_extract_sources_deduplicates():
    results = [
        {"tool": "t1", "result": {"sources": [{"filename": "dup.pdf"}]}},
        {"tool": "t2", "result": {"sources": [{"filename": "dup.pdf"}]}},
    ]
    sources = extract_sources_from_tool_results(results)
    assert sources.count("dup.pdf") == 1
