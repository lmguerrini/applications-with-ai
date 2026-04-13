from __future__ import annotations

from typing import Protocol

from src.schemas import (
    OfficialDocsAnswerResult,
    OfficialDocsLookupRequest,
    OfficialDocsLookupResult,
    RequestUsage,
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


def summarize_official_docs_answer(
    *,
    request: OfficialDocsLookupRequest,
    lookup_result: OfficialDocsLookupResult,
    chat_model: object | None = None,
    summary_impl: OfficialDocsSummaryImplementation | None = None,
) -> OfficialDocsAnswerResult:
    if lookup_result.library != request.library:
        raise ValueError(
            "Official docs lookup result library must match the request library."
        )

    if summary_impl is None:
        raise NotImplementedError("Official docs summary implementation is not wired yet.")

    answer_text, usage = summary_impl(
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
