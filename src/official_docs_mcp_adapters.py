from __future__ import annotations

from typing import Protocol

from src.config import Settings
from src.schemas import OfficialDocsLookupRequest, OfficialDocsLookupResult


LANGCHAIN_OFFICIAL_MCP_TOOL_NAME = "search_docs_by_lang_chain"
OPENAI_OFFICIAL_MCP_TOOL_NAME = "search_openai_docs"
REMOTE_MCP_UNAVAILABLE_MESSAGE = "Remote MCP not available"


class MCPToolCallFn(Protocol):
    def __call__(
        self,
        *,
        server_url: str,
        tool_name: str,
        arguments: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        ...


def send_mcp_tools_call_request(
    *,
    server_url: str,
    tool_name: str,
    arguments: dict[str, object],
    timeout_seconds: float,
) -> dict[str, object]:
    _ = (server_url, tool_name, arguments, timeout_seconds)
    raise NotImplementedError(REMOTE_MCP_UNAVAILABLE_MESSAGE)


def lookup_langchain_official_docs(
    *,
    request: OfficialDocsLookupRequest,
    settings: Settings | None = None,
    mcp_call_fn: MCPToolCallFn = send_mcp_tools_call_request,
) -> OfficialDocsLookupResult:
    _ = (request, settings, mcp_call_fn)
    raise NotImplementedError(REMOTE_MCP_UNAVAILABLE_MESSAGE)


def lookup_openai_official_docs(
    *,
    request: OfficialDocsLookupRequest,
    settings: Settings | None = None,
    mcp_call_fn: MCPToolCallFn = send_mcp_tools_call_request,
) -> OfficialDocsLookupResult:
    _ = (request, settings, mcp_call_fn)
    raise NotImplementedError(REMOTE_MCP_UNAVAILABLE_MESSAGE)
