import pytest

from app import AppValidationError, get_user_facing_error_message, validate_query


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
