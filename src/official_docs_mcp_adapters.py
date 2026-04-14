from __future__ import annotations

from typing import Protocol

from src.config import Settings, get_settings
from src.official_docs_langchain_adapter import run_langchain_official_docs_lookup
from src.official_docs_openai_adapter import run_openai_official_docs_lookup
from src.official_docs_mcp_transport import send_mcp_jsonrpc_request
from src.schemas import OfficialDocsLookupRequest, OfficialDocsLookupResult


class MCPRequestFn(Protocol):
    def __call__(
        self,
        *,
        server_url: str,
        method: str,
        params: dict[str, object] | None,
        timeout_seconds: float,
    ) -> dict[str, object]:
        ...


def lookup_langchain_official_docs(
    *,
    request: OfficialDocsLookupRequest,
    settings: Settings | None = None,
    mcp_call_fn: MCPRequestFn = send_mcp_jsonrpc_request,
) -> OfficialDocsLookupResult:
    resolved_settings = settings or get_settings()
    return run_langchain_official_docs_lookup(
        request=request,
        settings=resolved_settings,
        mcp_call_fn=mcp_call_fn,
    )


def lookup_openai_official_docs(
    *,
    request: OfficialDocsLookupRequest,
    settings: Settings | None = None,
    mcp_call_fn: MCPRequestFn = send_mcp_jsonrpc_request,
) -> OfficialDocsLookupResult:
    resolved_settings = settings or get_settings()
    return run_openai_official_docs_lookup(
        request=request,
        settings=resolved_settings,
        mcp_call_fn=mcp_call_fn,
    )
