from __future__ import annotations

import time

import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.chains import run_backend_query
from src.config import get_settings
from src.logger import configure_logging, get_logger
from src.rate_limit import apply_rate_limit


DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
PREVIEW_LENGTH = 80


class AppValidationError(ValueError):
    pass


LOGGER = get_logger(__name__)


@st.cache_resource
def get_vector_store() -> Chroma:
    settings = get_settings()
    api_key = settings.ensure_openai_api_key()
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model=settings.embedding_model,
    )
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_persist_dir),
    )


@st.cache_resource
def get_chat_model() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        api_key=settings.ensure_openai_api_key(),
        model=DEFAULT_CHAT_MODEL,
        temperature=0,
    )


def render_latest_turn() -> None:
    latest_turn = st.session_state.get("latest_turn")
    if not latest_turn:
        return

    with st.chat_message("user"):
        st.write(latest_turn["query"])

    with st.chat_message("assistant"):
        st.write(latest_turn["answer"])
        if latest_turn["tool_result"]:
            with st.expander("Tool Result"):
                st.json(latest_turn["tool_result"])
        if not latest_turn["used_context"] and not latest_turn["tool_result"]:
            st.caption("This response used the no-context fallback because no relevant chunks were retrieved.")
        if latest_turn["sources"]:
            with st.expander("Sources"):
                for source in latest_turn["sources"]:
                    st.write(f"- {source}")


def validate_query(raw_query: str, *, max_length: int) -> str:
    cleaned = raw_query.strip()
    if not cleaned:
        raise AppValidationError("Enter a question before sending a request.")
    if len(cleaned) > max_length:
        raise AppValidationError(
            f"Questions must be {max_length} characters or fewer."
        )
    return cleaned


def build_safe_log_metadata(query: str) -> dict[str, object]:
    preview = query[:PREVIEW_LENGTH].replace("\n", " ").strip()
    return {
        "query_length": len(query),
        "query_preview": preview,
    }


def get_user_facing_error_message(exc: Exception) -> str:
    message = str(exc)
    if isinstance(exc, AppValidationError):
        return message
    if "OPENAI_API_KEY" in message:
        return "OpenAI is not configured yet. Add OPENAI_API_KEY and try again."
    if "Chroma vector store is unavailable" in message or "Chroma vector store is empty" in message:
        return "The local knowledge base is not ready. Build the Chroma index before asking questions."
    if "Connection error" in message:
        return "The AI backend could not be reached. Please try again in a moment."
    return "Something went wrong while processing your request. Please try again."


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    st.set_page_config(page_title="RAG Assistant", page_icon=":speech_balloon:")
    st.title("RAG Assistant")
    st.write(
        "Ask about LangChain-based RAG application development with Chroma and Streamlit."
    )

    if "latest_turn" not in st.session_state:
        st.session_state["latest_turn"] = None
    if "request_timestamps" not in st.session_state:
        st.session_state["request_timestamps"] = []

    render_latest_turn()

    question = st.chat_input("Ask a question about the knowledge base")
    if not question:
        return

    with st.status("Handling request...", expanded=True) as status:
        try:
            status.write("Validating request")
            validated_query = validate_query(
                question,
                max_length=settings.max_query_length,
            )
            safe_metadata = build_safe_log_metadata(validated_query)
            LOGGER.info("request_received %s", safe_metadata)
            status.write("Checking rate limit")
            rate_limit_result = apply_rate_limit(
                st.session_state["request_timestamps"],
                now=time.time(),
                max_requests=settings.rate_limit_request_count,
                window_seconds=settings.rate_limit_window_seconds,
            )
            st.session_state["request_timestamps"] = rate_limit_result.updated_timestamps
            if not rate_limit_result.allowed:
                LOGGER.warning("request_rate_limited %s", safe_metadata)
                status.update(label="Rate limit exceeded", state="error")
                st.warning(
                    "Too many requests in a short period. "
                    f"Wait about {rate_limit_result.retry_after_seconds} seconds and try again."
                )
                return

            status.write("Loading resources")
            vector_store = get_vector_store()
            chat_model = get_chat_model()
            status.write("Processing request")
            result = run_backend_query(
                query=validated_query,
                vector_store=vector_store,
                chat_model=chat_model,
            )
            status.update(label="Request completed", state="complete")
        except AppValidationError as exc:
            LOGGER.info(
                "request_rejected %s",
                build_safe_log_metadata(question.strip()),
            )
            status.update(label="Request rejected", state="error")
            st.warning(get_user_facing_error_message(exc))
            return
        except Exception as exc:
            LOGGER.exception("backend_error %s", safe_metadata if "safe_metadata" in locals() else {})
            status.update(label="Request failed", state="error")
            st.error(get_user_facing_error_message(exc))
            return

    path = "tool" if result.tool_result is not None else "rag"
    if result.used_context is False and result.tool_result is None:
        path = "fallback"
    LOGGER.info(
        "request_completed %s",
        {
            **safe_metadata,
            "path": path,
            "source_count": len(result.answer_sources),
        },
    )

    st.session_state["latest_turn"] = {
        "query": validated_query,
        "answer": result.answer,
        "used_context": result.used_context,
        "sources": result.answer_sources,
        "tool_result": (
            result.tool_result.model_dump()
            if result.tool_result is not None
            else None
        ),
    }
    st.rerun()


if __name__ == "__main__":
    main()
