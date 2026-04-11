import pytest

from app import (
    AppValidationError,
    build_conversation_markdown,
    build_turn_record,
    get_help_content,
    get_response_type_label,
    get_user_facing_error_message,
    validate_query,
)
from src.schemas import AnswerResult


def test_validate_query_trims_whitespace() -> None:
    assert validate_query("  hello world  ", max_length=20) == "hello world"


def test_validate_query_rejects_empty_input() -> None:
    with pytest.raises(AppValidationError, match="Enter a question"):
        validate_query("   ", max_length=20)


def test_validate_query_rejects_overly_long_input() -> None:
    with pytest.raises(AppValidationError, match="characters or fewer"):
        validate_query("x" * 21, max_length=20)


def test_get_user_facing_error_message_maps_known_errors() -> None:
    assert get_user_facing_error_message(ValueError("OPENAI_API_KEY missing")) == (
        "OpenAI is not configured yet. Add OPENAI_API_KEY and try again."
    )
    assert get_user_facing_error_message(
        ValueError("Chroma vector store is empty. Build the local index before retrieval.")
    ) == (
        "The local knowledge base is not ready. Build the Chroma index before asking questions."
    )
    assert get_user_facing_error_message(RuntimeError("Connection error.")) == (
        "The AI backend could not be reached. Please try again in a moment."
    )


def test_build_turn_record_keeps_only_ui_fields() -> None:
    result = AnswerResult(
        answer="Grounded answer",
        used_context=True,
        retrieval=None,
        answer_sources=["Source A"],
        tool_result=None,
    )

    turn = build_turn_record("How should I persist Chroma locally?", result)

    assert turn == {
        "query": "How should I persist Chroma locally?",
        "answer": "Grounded answer",
        "used_context": True,
        "sources": ["Source A"],
        "tool_result": None,
    }


def test_get_response_type_label_maps_turn_variants() -> None:
    assert get_response_type_label(
        {
            "query": "q",
            "answer": "a",
            "used_context": True,
            "sources": ["s"],
            "tool_result": None,
        }
    ) == "Grounded answer"
    assert get_response_type_label(
        {
            "query": "q",
            "answer": "a",
            "used_context": False,
            "sources": [],
            "tool_result": {"tool_name": "estimate_openai_cost"},
        }
    ) == "Tool result"
    assert get_response_type_label(
        {
            "query": "q",
            "answer": "a",
            "used_context": False,
            "sources": [],
            "tool_result": None,
        }
    ) == "No-context fallback"


def test_get_help_content_includes_practical_sections() -> None:
    content = get_help_content()

    assert "LangChain-based RAG application design" in content["helps_with"]
    assert "Out of scope" not in content["out_of_scope"]
    assert len(content["example_questions"]) >= 3
    assert any("Grounded answer" in item for item in content["response_types"])


def test_build_conversation_markdown_handles_empty_history() -> None:
    markdown = build_conversation_markdown([])

    assert markdown == "# Conversation Export\n\n_No conversation history available._"


def test_build_conversation_markdown_formats_grounded_tool_and_fallback_turns() -> None:
    conversation_history = [
        {
            "query": "How should I persist Chroma locally?",
            "answer": "Persist the collection in a stable directory.",
            "used_context": True,
            "sources": ["Chroma Persistence and Reindexing Guide"],
            "tool_result": None,
        },
        {
            "query": "Estimate OpenAI cost",
            "answer": "Estimated total OpenAI cost: $0.002400 for model gpt-4.1-mini.",
            "used_context": False,
            "sources": [],
            "tool_result": {
                "tool_name": "estimate_openai_cost",
                "tool_error": None,
            },
        },
        {
            "query": "What is the capital of France?",
            "answer": "I could not find enough relevant context in the knowledge base to answer that safely.",
            "used_context": False,
            "sources": [],
            "tool_result": None,
        },
    ]

    markdown = build_conversation_markdown(conversation_history)

    assert "# Conversation Export" in markdown
    assert "**User question:** How should I persist Chroma locally?" in markdown
    assert "**Response type:** Grounded answer" in markdown
    assert "- Chroma Persistence and Reindexing Guide" in markdown
    assert "**Response type:** Tool result" in markdown
    assert "- tool_name: estimate_openai_cost" in markdown
    assert "**Response type:** No-context fallback" in markdown
