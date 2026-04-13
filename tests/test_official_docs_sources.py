import json

import pytest

from src.official_docs_fallback_adapters import (
    lookup_chroma_official_docs,
    lookup_streamlit_official_docs,
)
from src.official_docs_mcp_adapters import (
    REMOTE_MCP_UNAVAILABLE_MESSAGE,
    lookup_langchain_official_docs,
    lookup_openai_official_docs,
)
from src.official_docs_sources import (
    lookup_official_docs_documents,
    select_official_docs_source_adapter,
)
from src.schemas import OfficialDocsLookupRequest, OfficialDocsLookupResult


def test_select_official_docs_source_adapter_returns_requested_adapter() -> None:
    adapter = lambda *, request: OfficialDocsLookupResult(library="langchain", documents=[])

    selected_adapter = select_official_docs_source_adapter(
        library="langchain",
        adapters={"langchain": adapter},
    )

    assert selected_adapter is adapter


def test_lookup_langchain_official_docs_raises_when_remote_mcp_is_unavailable() -> None:
    request = OfficialDocsLookupRequest(
        query="How should I start a RAG app?",
        library="langchain",
    )
    called = False

    def mcp_call_fn(*, server_url, tool_name, arguments, timeout_seconds):
        nonlocal called
        called = True
        return {}

    with pytest.raises(NotImplementedError, match=REMOTE_MCP_UNAVAILABLE_MESSAGE):
        lookup_langchain_official_docs(
            request=request,
            mcp_call_fn=mcp_call_fn,
        )

    assert called is False


def test_lookup_openai_official_docs_raises_when_remote_mcp_is_unavailable() -> None:
    request = OfficialDocsLookupRequest(
        query="How do streaming responses work?",
        library="openai",
    )
    called = False

    def mcp_call_fn(*, server_url, tool_name, arguments, timeout_seconds):
        nonlocal called
        called = True
        return {}

    with pytest.raises(NotImplementedError, match=REMOTE_MCP_UNAVAILABLE_MESSAGE):
        lookup_openai_official_docs(
            request=request,
            mcp_call_fn=mcp_call_fn,
        )

    assert called is False


def test_lookup_streamlit_official_docs_uses_deterministic_fallback_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "library": "streamlit",
                        "title": "st.session_state",
                        "url": "https://docs.streamlit.io/session-state",
                        "snippets": [
                            "st.session_state stores values across reruns."
                        ],
                        "keywords": ["streamlit", "session", "state", "reruns", "chat"],
                    },
                    {
                        "library": "streamlit",
                        "title": "st.chat_message",
                        "url": "https://docs.streamlit.io/chat-message",
                        "snippets": [
                            "st.chat_message renders chat containers."
                        ],
                        "keywords": ["streamlit", "chat", "message", "ui"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    request = OfficialDocsLookupRequest(
        query="How do I keep chat history across reruns in Streamlit?",
        library="streamlit",
    )

    result = lookup_streamlit_official_docs(
        request=request,
        manifest_path=manifest_path,
    )

    assert result.library == "streamlit"
    assert result.documents[0].provider_mode == "official_fallback"
    assert result.documents[0].title == "st.session_state"
    assert "reruns" in result.documents[0].snippets[0].text


def test_lookup_chroma_official_docs_uses_deterministic_fallback_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "library": "chroma",
                        "title": "Persistent Client",
                        "url": "https://docs.trychroma.com/persistent-client",
                        "snippets": [
                            "Use a persistent client to keep collections on disk."
                        ],
                        "keywords": ["chroma", "persistent", "client", "disk", "restarts"],
                    },
                    {
                        "library": "chroma",
                        "title": "Metadata Filtering",
                        "url": "https://docs.trychroma.com/metadata-filtering",
                        "snippets": [
                            "Metadata filters narrow retrieval by attributes."
                        ],
                        "keywords": ["chroma", "metadata", "filtering", "retrieval"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    request = OfficialDocsLookupRequest(
        query="How do I persist Chroma collections across restarts?",
        library="chroma",
    )

    result = lookup_chroma_official_docs(
        request=request,
        manifest_path=manifest_path,
    )

    assert result.library == "chroma"
    assert result.documents[0].provider_mode == "official_fallback"
    assert result.documents[0].title == "Persistent Client"
    assert "collections on disk" in result.documents[0].snippets[0].text


def test_lookup_official_docs_documents_rejects_empty_adapter_output() -> None:
    request = OfficialDocsLookupRequest(
        query="How do I start?",
        library="langchain",
    )

    def empty_adapter(*, request):
        return OfficialDocsLookupResult(
            library="langchain",
            documents=[],
        )

    with pytest.raises(ValueError, match="returned no documents"):
        lookup_official_docs_documents(
            request=request,
            adapters={"langchain": empty_adapter},
        )


def test_lookup_openai_official_docs_still_raises_remote_unavailable_with_custom_transport() -> None:
    request = OfficialDocsLookupRequest(
        query="How do streaming responses work?",
        library="openai",
    )
    called = False

    def mcp_call_fn(*, server_url, tool_name, arguments, timeout_seconds):
        nonlocal called
        called = True
        return {"content": [{"type": "text", "text": "{\"unexpected\": true}"}]}

    with pytest.raises(NotImplementedError, match=REMOTE_MCP_UNAVAILABLE_MESSAGE):
        lookup_openai_official_docs(
            request=request,
            mcp_call_fn=mcp_call_fn,
        )

    assert called is False
