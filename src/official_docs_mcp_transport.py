from __future__ import annotations

import json
from collections.abc import Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request


def send_mcp_jsonrpc_request(
    *,
    server_url: str,
    method: str,
    params: dict[str, object] | None,
    timeout_seconds: float,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "jsonrpc": "2.0",
        "id": f"official-docs-{method}",
        "method": method,
    }
    if params is not None:
        payload["params"] = params

    body = json.dumps(payload).encode("utf-8")
    http_request = urllib_request.Request(
        server_url,
        data=body,
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8").strip()
    except (urllib_error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Official docs MCP request failed: {exc}") from exc

    rpc_payload = _parse_jsonrpc_response_body(response_body)

    response_error = rpc_payload.get("error")
    if isinstance(response_error, Mapping):
        error_message = response_error.get("message", "Unknown MCP error")
        raise RuntimeError(f"Official docs MCP returned an error: {error_message}")

    result = rpc_payload.get("result")
    if not isinstance(result, Mapping):
        raise RuntimeError("Official docs MCP response did not include a valid result object.")

    return dict(result)


def _parse_jsonrpc_response_body(response_body: str) -> dict[str, object]:
    direct_payload = _parse_json_object(response_body)
    if direct_payload is not None:
        return direct_payload

    sse_payload = _parse_json_object_from_sse_body(response_body)
    if sse_payload is not None:
        return sse_payload

    raise RuntimeError("Official docs MCP response was not valid JSON.")


def _parse_json_object(response_body: str) -> dict[str, object] | None:
    try:
        parsed_payload = json.loads(response_body)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed_payload, Mapping):
        return None

    return dict(parsed_payload)


def _parse_json_object_from_sse_body(response_body: str) -> dict[str, object] | None:
    data_lines: list[str] = []

    for raw_line in response_body.splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line:
            if data_lines:
                parsed_payload = _parse_json_object("\n".join(data_lines))
                if parsed_payload is not None:
                    return parsed_payload
                data_lines = []
            continue

        if stripped_line.startswith("data:"):
            data_lines.append(stripped_line.removeprefix("data:").strip())

    if data_lines:
        return _parse_json_object("\n".join(data_lines))

    return None
