from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from pydantic import ValidationError

from src.schemas import OfficialDocsAnswerResult, OfficialDocsLookupRequest


LOOKUP_OFFICIAL_DOCS_TOOL_NAME = "lookup_official_docs"
LOOKUP_OFFICIAL_DOCS_TOOL_DEFINITION = {
    "name": LOOKUP_OFFICIAL_DOCS_TOOL_NAME,
    "description": "Look up official documentation and return a normalized answer payload.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "library": {
                "type": "string",
                "enum": ["langchain", "openai", "streamlit", "chroma"],
            },
        },
        "required": ["query", "library"],
    },
}

class OfficialDocsServiceFn(Protocol):
    def __call__(
        self,
        *,
        request: OfficialDocsLookupRequest,
    ) -> OfficialDocsAnswerResult:
        ...


def handle_mcp_jsonrpc_request(
    payload: Mapping[str, object],
    *,
    service_fn: OfficialDocsServiceFn,
) -> dict[str, object]:
    request_id = payload.get("id")
    method = payload.get("method")

    if method == "tools/list":
        return _build_success_response(
            request_id,
            {"tools": [LOOKUP_OFFICIAL_DOCS_TOOL_DEFINITION]},
        )

    if method != "tools/call":
        return _build_error_response(
            request_id,
            code=-32601,
            message=f"Unsupported MCP method: {method}",
        )

    params = payload.get("params")
    if not isinstance(params, Mapping):
        return _build_error_response(
            request_id,
            code=-32602,
            message="tools/call requires an object params payload.",
        )

    if params.get("name") != LOOKUP_OFFICIAL_DOCS_TOOL_NAME:
        return _build_error_response(
            request_id,
            code=-32602,
            message="Unknown tool requested.",
        )

    arguments = params.get("arguments", {})
    if not isinstance(arguments, Mapping):
        return _build_error_response(
            request_id,
            code=-32602,
            message="Tool arguments must be an object.",
        )

    try:
        lookup_request = OfficialDocsLookupRequest.model_validate(arguments)
    except ValidationError as exc:
        return _build_error_response(
            request_id,
            code=-32602,
            message="Invalid official docs tool arguments.",
            data={"validation_error": str(exc)},
        )

    try:
        result = service_fn(request=lookup_request)
    except Exception as exc:
        return _build_error_response(
            request_id,
            code=-32000,
            message="Official docs service failed.",
            data={"tool_error": str(exc)},
        )

    return _build_success_response(
        request_id,
        {
            "structuredContent": result.model_dump(),
            "content": [{"type": "text", "text": result.answer}],
        },
    )


def _build_success_response(
    request_id: object,
    result: Mapping[str, object],
) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": dict(result),
    }


def _build_error_response(
    request_id: object,
    *,
    code: int,
    message: str,
    data: Mapping[str, object] | None = None,
) -> dict[str, object]:
    error_payload: dict[str, object] = {
        "code": code,
        "message": message,
    }
    if data is not None:
        error_payload["data"] = dict(data)

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": error_payload,
    }
