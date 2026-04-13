import pytest

from src.official_docs_service import (
    OFFICIAL_DOCS_LOOKUP_UNAVAILABLE_ANSWER,
    answer_official_docs_query,
)
from src.schemas import (
    OfficialDocsDocument,
    OfficialDocsLookupRequest,
    OfficialDocsLookupResult,
    OfficialDocsSnippet,
)


def build_lookup_result() -> OfficialDocsLookupResult:
    return OfficialDocsLookupResult(
        library="langchain",
        documents=[
            OfficialDocsDocument(
                title="Build a RAG agent with LangChain",
                url="https://docs.langchain.com/guides/rag",
                provider_mode="official_mcp",
                snippets=[
                    OfficialDocsSnippet(
                        text="Start with a simple retrieval pipeline.",
                        rank=1,
                    )
                ],
            )
        ],
    )


def test_answer_official_docs_query_orchestrates_lookup_then_summary() -> None:
    request = OfficialDocsLookupRequest(
        query="According to LangChain docs, how should I start a small RAG app?",
        library="langchain",
    )
    calls: list[tuple[str, object]] = []

    def lookup_impl(*, request, adapters=None):
        calls.append(("lookup", request.query))
        return build_lookup_result()

    def summary_impl(*, request, lookup_result, chat_model):
        calls.append(("summary", chat_model))
        assert lookup_result.library == "langchain"
        return ("According to the docs, start with a simple retrieval pipeline.", None)

    result = answer_official_docs_query(
        request=request,
        chat_model="stub-chat-model",
        lookup_impl=lookup_impl,
        summary_impl=summary_impl,
    )

    assert result.answer == "According to the docs, start with a simple retrieval pipeline."
    assert calls == [
        ("lookup", "According to LangChain docs, how should I start a small RAG app?"),
        ("summary", "stub-chat-model"),
    ]


def test_answer_official_docs_query_returns_graceful_result_when_lookup_not_implemented() -> None:
    request = OfficialDocsLookupRequest(
        query="According to LangChain docs, how should I start a small RAG app?",
        library="langchain",
    )

    def lookup_impl(*, request, adapters=None):
        raise NotImplementedError("Remote MCP not available")

    def fail_summary_impl(*, request, lookup_result, chat_model):
        raise AssertionError("summary should not run when MCP is unavailable")

    result = answer_official_docs_query(
        request=request,
        chat_model="stub-chat-model",
        lookup_impl=lookup_impl,
        summary_impl=fail_summary_impl,
    )

    assert result.library == "langchain"
    assert result.answer == OFFICIAL_DOCS_LOOKUP_UNAVAILABLE_ANSWER
    assert result.lookup_result.library == "langchain"
    assert result.lookup_result.documents == []
    assert result.usage is None


def test_answer_official_docs_query_wraps_lookup_failures() -> None:
    request = OfficialDocsLookupRequest(
        query="According to LangChain docs, how should I start a small RAG app?",
        library="langchain",
    )

    def lookup_impl(*, request, adapters=None):
        raise RuntimeError("connection failed")

    with pytest.raises(RuntimeError, match="Official docs lookup failed: connection failed"):
        answer_official_docs_query(
            request=request,
            chat_model="stub-chat-model",
            lookup_impl=lookup_impl,
            summary_impl=lambda **kwargs: ("unused", None),
        )


def test_answer_official_docs_query_wraps_summary_failures() -> None:
    request = OfficialDocsLookupRequest(
        query="According to LangChain docs, how should I start a small RAG app?",
        library="langchain",
    )

    def summary_impl(*, request, lookup_result, chat_model):
        raise ValueError("blank answer")

    with pytest.raises(RuntimeError, match="Official docs summary failed: blank answer"):
        answer_official_docs_query(
            request=request,
            chat_model="stub-chat-model",
            lookup_impl=lambda **kwargs: build_lookup_result(),
            summary_impl=summary_impl,
        )
