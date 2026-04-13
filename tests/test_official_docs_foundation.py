import pytest

from official_docs_mcp_server import (
    LOOKUP_OFFICIAL_DOCS_TOOL_NAME,
    handle_mcp_jsonrpc_request,
)
from src.official_docs_service import answer_official_docs_query
from src.official_docs_sources import lookup_official_docs_documents
from src.official_docs_summary import summarize_official_docs_answer
from src.schemas import (
    OfficialDocsAnswerResult,
    OfficialDocsDocument,
    OfficialDocsLookupRequest,
    OfficialDocsLookupResult,
    OfficialDocsSnippet,
    RequestUsage,
)


def build_lookup_result(library: str = "langchain") -> OfficialDocsLookupResult:
    return OfficialDocsLookupResult(
        library=library,
        documents=[
            OfficialDocsDocument(
                title="Build a RAG agent with LangChain",
                url="https://docs.example.com/rag",
                provider_mode="official_mcp",
                snippets=[
                    OfficialDocsSnippet(
                        text="Start with a small retrieval pipeline.",
                        rank=1,
                    )
                ],
            )
        ],
    )


def test_official_docs_lookup_request_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        OfficialDocsLookupRequest(query="   ", library="langchain")


def test_official_docs_answer_result_requires_matching_lookup_library() -> None:
    with pytest.raises(ValueError, match="must match the lookup result library"):
        OfficialDocsAnswerResult(
            library="openai",
            answer="Use the official docs.",
            lookup_result=build_lookup_result("langchain"),
        )


def test_lookup_official_docs_documents_selects_requested_adapter() -> None:
    request = OfficialDocsLookupRequest(
        query="How should I start a small RAG app?",
        library="langchain",
    )

    def langchain_adapter(*, request: OfficialDocsLookupRequest) -> OfficialDocsLookupResult:
        assert request.library == "langchain"
        return build_lookup_result("langchain")

    result = lookup_official_docs_documents(
        request=request,
        adapters={"langchain": langchain_adapter},
    )

    assert result.library == "langchain"
    assert result.documents[0].snippets[0].text == "Start with a small retrieval pipeline."


def test_summarize_official_docs_answer_wraps_summary_output() -> None:
    request = OfficialDocsLookupRequest(
        query="What do the docs recommend?",
        library="langchain",
    )
    lookup_result = build_lookup_result("langchain")
    chat_model = object()

    def summary_impl(*, request, lookup_result, chat_model):
        assert request.query == "What do the docs recommend?"
        assert lookup_result.library == "langchain"
        assert chat_model is not None
        return (
            "According to the official LangChain docs, start with a small retrieval pipeline.",
            RequestUsage(
                model_name="gpt-4.1-mini",
                input_tokens=10,
                output_tokens=4,
                total_tokens=14,
                estimated_cost_usd=0.00001,
            ),
        )

    result = summarize_official_docs_answer(
        request=request,
        lookup_result=lookup_result,
        chat_model=chat_model,
        summary_impl=summary_impl,
    )

    assert result.library == "langchain"
    assert result.lookup_result == lookup_result
    assert result.usage is not None
    assert result.usage.total_tokens == 14


def test_answer_official_docs_query_orchestrates_lookup_and_summary() -> None:
    request = OfficialDocsLookupRequest(
        query="How should I approach a small RAG app?",
        library="langchain",
    )
    calls: list[tuple[str, object]] = []
    adapter_sentinel = object()

    def lookup_impl(*, request, adapters=None):
        calls.append(("lookup", request.query))
        assert adapters == {"langchain": adapter_sentinel}
        return build_lookup_result("langchain")

    def summary_impl(*, request, lookup_result, chat_model):
        calls.append(("summary", chat_model))
        assert lookup_result.library == "langchain"
        return ("Use the smallest viable retrieval pipeline first.", None)

    adapters = {"langchain": adapter_sentinel}
    result = answer_official_docs_query(
        request=request,
        chat_model="stub-chat-model",
        adapters=adapters,
        lookup_impl=lookup_impl,
        summary_impl=summary_impl,
    )

    assert result.answer == "Use the smallest viable retrieval pipeline first."
    assert calls == [
        ("lookup", "How should I approach a small RAG app?"),
        ("summary", "stub-chat-model"),
    ]


def test_handle_mcp_jsonrpc_request_returns_tool_result_payload() -> None:
    request_payload = {
        "jsonrpc": "2.0",
        "id": "request-1",
        "method": "tools/call",
        "params": {
            "name": LOOKUP_OFFICIAL_DOCS_TOOL_NAME,
            "arguments": {
                "query": "How should I start with LangChain RAG?",
                "library": "langchain",
            },
        },
    }

    def service_fn(*, request: OfficialDocsLookupRequest) -> OfficialDocsAnswerResult:
        assert request.library == "langchain"
        return OfficialDocsAnswerResult(
            library="langchain",
            answer="According to the official LangChain docs, start with a small retrieval pipeline.",
            lookup_result=build_lookup_result("langchain"),
            usage=None,
        )

    response = handle_mcp_jsonrpc_request(
        request_payload,
        service_fn=service_fn,
    )

    assert response["result"]["content"] == [
        {
            "type": "text",
            "text": "According to the official LangChain docs, start with a small retrieval pipeline.",
        }
    ]
    assert response["result"]["structuredContent"]["library"] == "langchain"


def test_handle_mcp_jsonrpc_request_returns_tools_list() -> None:
    response = handle_mcp_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": "request-2",
            "method": "tools/list",
        },
        service_fn=lambda *, request: OfficialDocsAnswerResult(
            library=request.library,
            answer="unused",
            lookup_result=build_lookup_result(request.library),
            usage=None,
        ),
    )

    assert response["result"]["tools"] == [
        {
            "name": LOOKUP_OFFICIAL_DOCS_TOOL_NAME,
            "description": "Look up official documentation and return a normalized answer payload.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "library": {
                        "type": "string",
                        "enum": ["langchain", "openai", "streamlit", "chroma"],
                    },
                },
                "required": ["query", "library"],
            },
        }
    ]
