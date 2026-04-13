from types import SimpleNamespace

import pytest

from src import official_docs_summary
from src.schemas import (
    OfficialDocsDocument,
    OfficialDocsLookupRequest,
    OfficialDocsLookupResult,
    OfficialDocsSnippet,
)


class StubChatModel:
    def __init__(self, response_text: str, *, model_name: str = "gpt-4.1-mini") -> None:
        self.response_text = response_text
        self.model_name = model_name
        self.prompts: list[str] = []
        self.response_metadata: dict[str, object] | None = None
        self.usage_metadata: dict[str, int] | None = None

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(
            content=self.response_text,
            response_metadata=self.response_metadata,
            usage_metadata=self.usage_metadata,
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
                    ),
                    OfficialDocsSnippet(
                        text="Keep the first version small enough to inspect end to end.",
                        rank=2,
                    ),
                ],
            )
        ],
    )


def test_build_official_docs_summary_prompt_includes_grounding_and_evidence() -> None:
    prompt = official_docs_summary.build_official_docs_summary_prompt(
        request=OfficialDocsLookupRequest(
            query="According to LangChain docs, how should I start a small RAG app?",
            library="langchain",
        ),
        lookup_result=build_lookup_result(),
    )

    assert official_docs_summary.OFFICIAL_DOCS_SUMMARY_SYSTEM_PROMPT in prompt
    assert "Use only the provided official-docs evidence." in prompt
    assert "Do not use outside knowledge." in prompt
    assert "Do not invent facts, titles, URLs, methods, APIs, or capabilities." in prompt
    assert "If the evidence is insufficient, say so clearly." in prompt
    assert "User query: According to LangChain docs, how should I start a small RAG app?" in prompt
    assert "Library: langchain" in prompt
    assert "Title: Build a RAG agent with LangChain" in prompt
    assert "URL: https://docs.langchain.com/guides/rag" in prompt
    assert "Snippet 1: Start with a simple retrieval pipeline." in prompt
    assert "Snippet 2: Keep the first version small enough to inspect end to end." in prompt


def test_summarize_official_docs_answer_returns_grounded_result_and_usage() -> None:
    model = StubChatModel(
        "According to the official LangChain docs, start with a simple retrieval pipeline."
    )
    model.usage_metadata = {
        "input_tokens": 18,
        "output_tokens": 7,
        "total_tokens": 25,
    }

    result = official_docs_summary.summarize_official_docs_answer(
        request=OfficialDocsLookupRequest(
            query="According to LangChain docs, how should I start a small RAG app?",
            library="langchain",
        ),
        lookup_result=build_lookup_result(),
        chat_model=model,
    )

    assert result.library == "langchain"
    assert result.lookup_result == build_lookup_result()
    assert result.answer.startswith("According to the official LangChain docs")
    assert result.usage is not None
    assert result.usage.model_name == "gpt-4.1-mini"
    assert result.usage.total_tokens == 25
    assert result.usage.estimated_cost_usd == 0.000018
    assert len(model.prompts) == 1
    assert "Build a RAG agent with LangChain" in model.prompts[0]


def test_summarize_official_docs_answer_rejects_blank_model_output() -> None:
    model = StubChatModel("   ")

    with pytest.raises(ValueError, match="returned empty output"):
        official_docs_summary.summarize_official_docs_answer(
            request=OfficialDocsLookupRequest(
                query="According to LangChain docs, how should I start a small RAG app?",
                library="langchain",
            ),
            lookup_result=build_lookup_result(),
            chat_model=model,
        )


def test_summarize_official_docs_answer_returns_none_usage_when_metadata_is_missing() -> None:
    model = StubChatModel("According to the official LangChain docs, start small.")

    result = official_docs_summary.summarize_official_docs_answer(
        request=OfficialDocsLookupRequest(
            query="According to LangChain docs, how should I start a small RAG app?",
            library="langchain",
        ),
        lookup_result=build_lookup_result(),
        chat_model=model,
    )

    assert result.answer == "According to the official LangChain docs, start small."
    assert result.usage is None
