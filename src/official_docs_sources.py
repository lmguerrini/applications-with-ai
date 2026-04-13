from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from src.schemas import OfficialDocsLookupRequest, OfficialDocsLookupResult, SupportedToolLibrary


class OfficialDocsSourceAdapter(Protocol):
    def __call__(
        self,
        *,
        request: OfficialDocsLookupRequest,
    ) -> OfficialDocsLookupResult:
        ...


def lookup_langchain_official_docs(
    *,
    request: OfficialDocsLookupRequest,
) -> OfficialDocsLookupResult:
    raise NotImplementedError("LangChain official-docs adapter is not implemented yet.")


def lookup_openai_official_docs(
    *,
    request: OfficialDocsLookupRequest,
) -> OfficialDocsLookupResult:
    raise NotImplementedError("OpenAI official-docs adapter is not implemented yet.")


def lookup_streamlit_official_docs(
    *,
    request: OfficialDocsLookupRequest,
) -> OfficialDocsLookupResult:
    raise NotImplementedError("Streamlit official-docs adapter is not implemented yet.")


def lookup_chroma_official_docs(
    *,
    request: OfficialDocsLookupRequest,
) -> OfficialDocsLookupResult:
    raise NotImplementedError("Chroma official-docs adapter is not implemented yet.")


DEFAULT_OFFICIAL_DOCS_SOURCE_ADAPTERS: dict[
    SupportedToolLibrary,
    OfficialDocsSourceAdapter,
] = {
    "langchain": lookup_langchain_official_docs,
    "openai": lookup_openai_official_docs,
    "streamlit": lookup_streamlit_official_docs,
    "chroma": lookup_chroma_official_docs,
}


def select_official_docs_source_adapter(
    *,
    library: SupportedToolLibrary,
    adapters: Mapping[str, OfficialDocsSourceAdapter],
) -> OfficialDocsSourceAdapter:
    adapter = adapters.get(library)
    if adapter is None:
        raise ValueError(f"No official docs adapter configured for library: {library}")
    return adapter


def lookup_official_docs_documents(
    *,
    request: OfficialDocsLookupRequest,
    adapters: Mapping[str, OfficialDocsSourceAdapter] | None = None,
) -> OfficialDocsLookupResult:
    adapter_map = adapters or DEFAULT_OFFICIAL_DOCS_SOURCE_ADAPTERS
    adapter = select_official_docs_source_adapter(
        library=request.library,
        adapters=adapter_map,
    )
    result = adapter(request=request)
    if result.library != request.library:
        raise ValueError(
            "Official docs adapter returned a lookup result for the wrong library."
        )
    return result
