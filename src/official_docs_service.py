from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from src.official_docs_sources import (
    OfficialDocsSourceAdapter,
    lookup_official_docs_documents,
)
from src.official_docs_summary import (
    OfficialDocsSummaryImplementation,
    summarize_official_docs_answer,
)
from src.schemas import OfficialDocsAnswerResult, OfficialDocsLookupRequest, OfficialDocsLookupResult


class OfficialDocsLookupImplementation(Protocol):
    def __call__(
        self,
        *,
        request: OfficialDocsLookupRequest,
        adapters: Mapping[str, OfficialDocsSourceAdapter] | None = None,
    ) -> OfficialDocsLookupResult:
        ...


def answer_official_docs_query(
    *,
    request: OfficialDocsLookupRequest,
    chat_model: object | None = None,
    adapters: Mapping[str, OfficialDocsSourceAdapter] | None = None,
    lookup_impl: OfficialDocsLookupImplementation = lookup_official_docs_documents,
    summary_impl: OfficialDocsSummaryImplementation | None = None,
) -> OfficialDocsAnswerResult:
    lookup_result = lookup_impl(
        request=request,
        adapters=adapters,
    )
    return summarize_official_docs_answer(
        request=request,
        lookup_result=lookup_result,
        chat_model=chat_model,
        summary_impl=summary_impl,
    )
