from types import SimpleNamespace

from langchain_core.embeddings.fake import FakeEmbeddings

from src import chains
from src.config import Settings
from src.knowledge_base import build_index
from src.schemas import (
    AnswerResult,
    RetrievalFilters,
    RetrievalResult,
    RetrievedChunk,
)


class StubChatModel:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.response_text)


def test_answer_query_returns_fallback_without_model_call(monkeypatch) -> None:
    retrieval_result = RetrievalResult(
        rewritten_query="streamlit chat source display",
        applied_filters=RetrievalFilters(topic="streamlit"),
        used_fallback=False,
        chunks=[],
        sources=[],
    )

    def fake_retrieve_chunks(*, vector_store, request):
        return retrieval_result

    monkeypatch.setattr(chains, "retrieve_chunks", fake_retrieve_chunks)
    model = StubChatModel("unused")

    result = chains.answer_query(
        query="How should I show retrieved sources in Streamlit?",
        vector_store=object(),
        chat_model=model,
    )

    assert result.answer == chains.NO_CONTEXT_FALLBACK
    assert result.used_context is False
    assert result.answer_sources == []
    assert model.prompts == []
    assert result.tool_result is None


def test_build_grounded_prompt_includes_query_context_and_sources() -> None:
    retrieval_result = RetrievalResult(
        rewritten_query="chroma persist local directory",
        applied_filters=RetrievalFilters(topic="chroma", library="chroma"),
        used_fallback=False,
        chunks=[
            RetrievedChunk.model_validate(
                {
                    "content": "Use a stable local path for Chroma persistence.",
                    "metadata": {
                        "doc_id": "chroma-persistence-guide",
                        "source_path": "data/raw/chroma_persistence_guide.md",
                        "title": "Chroma Persistence Guide",
                        "topic": "chroma",
                        "library": "chroma",
                        "doc_type": "how_to",
                        "difficulty": "intro",
                        "error_family": "persistence",
                        "chunk_index": 0,
                    },
                }
            )
        ],
        sources=[
            "Chroma Persistence Guide | topic=chroma | library=chroma | "
            "doc_type=how_to | difficulty=intro | "
            "source=data/raw/chroma_persistence_guide.md | chunk=0 | "
            "error_family=persistence"
        ],
    )

    prompt = chains.build_grounded_prompt(
        original_query="How do I persist Chroma locally?",
        retrieval=retrieval_result,
    )

    assert "Answer the question using only the provided context." in prompt
    assert "User query: How do I persist Chroma locally?" in prompt
    assert "Retrieval query: chroma persist local directory" in prompt
    assert "Use a stable local path for Chroma persistence." in prompt
    assert "Chroma Persistence Guide | topic=chroma | library=chroma" in prompt


def test_answer_query_returns_structured_output(monkeypatch) -> None:
    retrieval_result = RetrievalResult(
        rewritten_query="streamlit source metadata",
        applied_filters=RetrievalFilters(topic="streamlit", library="streamlit"),
        used_fallback=False,
        chunks=[
            RetrievedChunk.model_validate(
                {
                    "content": "Show source titles next to the answer in Streamlit.",
                    "metadata": {
                        "doc_id": "streamlit-chat-patterns",
                        "source_path": "data/raw/streamlit_chat_patterns.md",
                        "title": "Streamlit Chat Patterns",
                        "topic": "streamlit",
                        "library": "streamlit",
                        "doc_type": "example",
                        "difficulty": "intermediate",
                        "error_family": "ui",
                        "chunk_index": 0,
                    },
                }
            )
        ],
        sources=[
            "Streamlit Chat Patterns | topic=streamlit | library=streamlit | "
            "doc_type=example | difficulty=intermediate | "
            "source=data/raw/streamlit_chat_patterns.md | chunk=0 | error_family=ui"
        ],
    )

    def fake_retrieve_chunks(*, vector_store, request):
        return retrieval_result

    monkeypatch.setattr(chains, "retrieve_chunks", fake_retrieve_chunks)
    model = StubChatModel(
        "Use the retrieved source title and metadata next to the answer in Streamlit."
    )

    result = chains.answer_query(
        query="How should I show sources in Streamlit?",
        vector_store=object(),
        chat_model=model,
        top_k=2,
    )

    assert result.answer == (
        "Use the retrieved source title and metadata next to the answer in Streamlit."
    )
    assert result.used_context is True
    assert result.retrieval == retrieval_result
    assert result.answer_sources == retrieval_result.sources
    assert len(model.prompts) == 1
    assert result.tool_result is None


def test_run_backend_query_routes_tool_request_without_answer_call(monkeypatch) -> None:
    def fail_answer_query(**kwargs):
        raise AssertionError("answer_query should not run for a matched tool request")

    monkeypatch.setattr(chains, "answer_query", fail_answer_query)

    result = chains.run_backend_query(
        query=(
            "Estimate OpenAI cost for openai model gpt-4.1-mini "
            "input_tokens=1000 output_tokens=500 calls=2"
        ),
        vector_store=object(),
        chat_model=StubChatModel("unused"),
    )

    assert result.tool_result is not None
    assert result.tool_result.tool_name == "estimate_openai_cost"
    assert "Estimated total OpenAI cost" in result.answer
    assert result.retrieval is None
    assert result.answer_sources == []


def test_run_backend_query_returns_tool_validation_error_without_rag_fallback(monkeypatch) -> None:
    def fail_answer_query(**kwargs):
        raise AssertionError("answer_query should not run when a tool route is selected")

    monkeypatch.setattr(chains, "answer_query", fail_answer_query)

    result = chains.run_backend_query(
        query="Estimate OpenAI cost",
        vector_store=object(),
        chat_model=StubChatModel("unused"),
    )

    assert result.tool_result is not None
    assert result.tool_result.tool_name == "estimate_openai_cost"
    assert result.tool_result.tool_output is None
    assert "supported model name" in result.answer
    assert result.retrieval is None


def test_run_backend_query_routes_retrieval_config_request(monkeypatch) -> None:
    def fail_answer_query(**kwargs):
        raise AssertionError("answer_query should not run for retrieval config tool requests")

    monkeypatch.setattr(chains, "answer_query", fail_answer_query)

    result = chains.run_backend_query(
        query="Recommend retrieval config for short markdown docs used for question answering",
        vector_store=object(),
        chat_model=StubChatModel("unused"),
    )

    assert result.tool_result is not None
    assert result.tool_result.tool_name == "recommend_retrieval_config"
    assert result.tool_result.tool_error is None
    assert "Recommended retrieval settings" in result.answer


def test_run_backend_query_keeps_normal_answer_flow(monkeypatch) -> None:
    expected = AnswerResult(
        answer="Grounded answer",
        used_context=True,
        retrieval=RetrievalResult(
            rewritten_query="langchain retrieval source display",
            applied_filters=RetrievalFilters(topic="langchain"),
            used_fallback=False,
            chunks=[],
            sources=[],
        ),
        answer_sources=["source-1"],
        tool_result=None,
    )

    def fake_answer_query(**kwargs):
        return expected

    monkeypatch.setattr(chains, "answer_query", fake_answer_query)

    result = chains.run_backend_query(
        query="How should I format sources in LangChain retrieval results?",
        vector_store=object(),
        chat_model=StubChatModel("unused"),
    )

    assert result == expected


def test_answer_query_uses_no_context_fallback_for_weak_retrieval(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    chroma_dir = tmp_path / "chroma_db"

    (raw_dir / "chroma_persistence.md").write_text(
        """---
title: Chroma Persistence Guide
topic: chroma
library: chroma
doc_type: how_to
difficulty: intro
error_family: persistence
---
Use a persistent Chroma directory when you want local retrieval data to survive between runs.
Rebuild the collection if you need a clean index without duplicate chunks.
""",
        encoding="utf-8",
    )
    (raw_dir / "streamlit_debug.md").write_text(
        """---
title: Streamlit Debugging Example
topic: streamlit
library: streamlit
doc_type: example
difficulty: intermediate
error_family: ui
---
Streamlit chat interfaces should show source metadata next to the answer.
Use session state carefully when debugging reruns in a retrieval app.
""",
        encoding="utf-8",
    )

    settings = Settings(
        RAW_DATA_DIR=raw_dir,
        CHROMA_PERSIST_DIR=chroma_dir,
        CHROMA_COLLECTION_NAME="weak_retrieval_chain_test",
        CHUNK_SIZE=250,
        CHUNK_OVERLAP=20,
    )
    vector_store = build_index(
        settings=settings,
        embeddings=FakeEmbeddings(size=32),
    )
    model = StubChatModel("This should not be used.")

    result = chains.answer_query(
        query="What is the capital of France?",
        vector_store=vector_store,
        chat_model=model,
    )

    assert result.answer == chains.NO_CONTEXT_FALLBACK
    assert result.used_context is False
    assert result.answer_sources == []
    assert result.retrieval is not None
    assert result.retrieval.chunks == []
    assert model.prompts == []
