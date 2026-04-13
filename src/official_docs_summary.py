from __future__ import annotations

from typing import Protocol

from src.llm_response_utils import extract_request_usage, extract_text
from src.schemas import (
    OfficialDocsAnswerResult,
    OfficialDocsDocument,
    OfficialDocsLookupRequest,
    OfficialDocsLookupResult,
    RequestUsage,
)


OFFICIAL_DOCS_SUMMARY_SYSTEM_PROMPT = (
    "You are answering a question using only official documentation evidence.\n"
    "Use only the provided official-docs evidence.\n"
    "Do not use outside knowledge.\n"
    "Do not invent facts, titles, URLs, methods, APIs, or capabilities.\n"
    "If the evidence is insufficient, say so clearly.\n"
    "Keep the answer short and practical."
)


class OfficialDocsSummaryImplementation(Protocol):
    def __call__(
        self,
        *,
        request: OfficialDocsLookupRequest,
        lookup_result: OfficialDocsLookupResult,
        chat_model: object | None,
    ) -> tuple[str, RequestUsage | None]:
        ...


class OfficialDocsChatModelLike(Protocol):
    def invoke(self, prompt: str):
        ...


def summarize_official_docs_answer(
    *,
    request: OfficialDocsLookupRequest,
    lookup_result: OfficialDocsLookupResult,
    chat_model: OfficialDocsChatModelLike | None = None,
    summary_impl: OfficialDocsSummaryImplementation | None = None,
) -> OfficialDocsAnswerResult:
    if lookup_result.library != request.library:
        raise ValueError(
            "Official docs lookup result library must match the request library."
        )

    implementation = summary_impl or _summarize_with_chat_model
    answer_text, usage = implementation(
        request=request,
        lookup_result=lookup_result,
        chat_model=chat_model,
    )
    return OfficialDocsAnswerResult(
        library=request.library,
        answer=answer_text,
        lookup_result=lookup_result,
        usage=usage,
    )


def build_official_docs_summary_prompt(
    *,
    request: OfficialDocsLookupRequest,
    lookup_result: OfficialDocsLookupResult,
) -> str:
    document_blocks = [
        _format_document_block(index=index, document=document)
        for index, document in enumerate(lookup_result.documents, start=1)
    ]
    return (
        f"{OFFICIAL_DOCS_SUMMARY_SYSTEM_PROMPT}\n\n"
        f"User query: {request.query}\n"
        f"Library: {request.library}\n\n"
        f"Official documentation evidence:\n\n"
        + "\n\n".join(document_blocks)
    )


def _summarize_with_chat_model(
    *,
    request: OfficialDocsLookupRequest,
    lookup_result: OfficialDocsLookupResult,
    chat_model: OfficialDocsChatModelLike | None,
) -> tuple[str, RequestUsage | None]:
    if chat_model is None:
        raise ValueError("Official docs summary requires a chat model.")

    prompt = build_official_docs_summary_prompt(
        request=request,
        lookup_result=lookup_result,
    )
    model_response = chat_model.invoke(prompt)
    answer_text = extract_text(model_response)
    if not answer_text:
        raise ValueError("Official docs summary model returned empty output.")

    usage = extract_request_usage(model_response, chat_model=chat_model)
    return answer_text, usage


def _format_document_block(*, index: int, document: OfficialDocsDocument) -> str:
    snippet_lines = [
        f"Snippet {snippet_index}: {snippet.text}"
        for snippet_index, snippet in enumerate(document.snippets, start=1)
    ]
    return (
        f"[Document {index}]\n"
        f"Title: {document.title}\n"
        f"URL: {document.url}\n"
        f"Provider mode: {document.provider_mode}\n"
        + "\n".join(snippet_lines)
    )
