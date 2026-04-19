from __future__ import annotations

import json

import streamlit as st


def format_tool_name_label(value: object) -> str | None:
    labels = {
        "estimate_openai_cost": "Estimate OpenAI Cost",
        "diagnose_stack_error": "Diagnose Stack Error",
        "recommend_retrieval_config": "Recommend Retrieval Config",
    }
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    return labels.get(cleaned, cleaned.replace("_", " ").title())


def format_tool_field_label(key: str) -> str:
    labels = {
        "model": "Model",
        "input_tokens": "Input tokens",
        "output_tokens": "Output tokens",
        "num_calls": "Number of calls",
        "estimated_input_cost_usd": "Estimated input cost (USD)",
        "estimated_output_cost_usd": "Estimated output cost (USD)",
        "estimated_total_cost_usd": "Estimated total cost (USD)",
        "library": "Library",
        "error_message": "Error message",
        "code_context_summary": "Code context summary",
        "error_category": "Error category",
        "likely_causes": "Likely causes",
        "recommended_checks": "Recommended checks",
        "content_type": "Content type",
        "document_length": "Document length",
        "task_type": "Task type",
        "chunk_size": "Chunk size",
        "chunk_overlap": "Chunk overlap",
        "top_k": "Top K",
        "use_metadata_filters": "Use metadata filters",
        "rationale": "Rationale",
    }
    return labels.get(key, key.replace("_", " ").title())


def render_tool_result_fields(fields: list[dict[str, object]]) -> None:
    for field in fields:
        label = field["label"]
        lines = field["lines"]
        if len(lines) == 1:
            st.write(f"- {label}: {lines[0]}")
            continue

        st.write(f"**{label}**")
        for line in lines:
            st.write(f"- {line}")


def _format_tool_field_lines(key: str, value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [
            line
            for item in value
            if (line := _format_tool_scalar_value(key, item)) is not None
        ]

    formatted_value = _format_tool_scalar_value(key, value)
    if formatted_value is None:
        return []
    return [formatted_value]


def _format_tool_scalar_value(key: str, value: object) -> str | None:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if "cost" in key:
            return f"${value:.6f}"
        return f"{value:g}"
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def parse_source_string(source: str) -> dict[str, object] | None:
    segments = [segment.strip() for segment in source.split("|")]
    if not segments or not segments[0]:
        return None

    title = segments[0]
    metadata: dict[str, str] = {}
    for segment in segments[1:]:
        if not segment:
            continue
        if "=" not in segment:
            return None
        key, value = segment.split("=", 1)
        cleaned_key = key.strip()
        cleaned_value = value.strip()
        if not cleaned_key or not cleaned_value:
            return None
        metadata[cleaned_key] = cleaned_value

    return {
        "title": title,
        "metadata": metadata,
    }


def format_source_display(source: str) -> dict[str, object]:
    parsed = parse_source_string(source)
    if parsed is None:
        return {
            "title": "Source",
            "metadata_fragments": [],
            "source_path": None,
            "raw_source": source,
            "parse_failed": True,
        }

    metadata = parsed["metadata"]
    metadata_fragments = [
        _format_source_metadata_fragment(key, value)
        for key, value in metadata.items()
        if key != "source"
    ]
    return {
        "title": parsed["title"],
        "metadata_fragments": metadata_fragments,
        "source_path": metadata.get("source"),
        "raw_source": source,
        "parse_failed": False,
    }


def group_source_displays(sources: list[str]) -> list[dict[str, object]]:
    grouped_sources: list[dict[str, object]] = []
    grouped_index: dict[tuple[str, object], dict[str, object]] = {}

    for source in sources:
        parsed_source = parse_source_string(source)
        source_display = format_source_display(source)
        if source_display["parse_failed"]:
            grouped_sources.append(
                {
                    **source_display,
                    "chunk_count": 1,
                    "chunk_indices": [],
                }
            )
            continue

        group_key = (str(source_display["title"]), source_display["source_path"])
        if group_key not in grouped_index:
            grouped_display = {
                **source_display,
                "metadata_fragments": [
                    fragment
                    for fragment in source_display["metadata_fragments"]
                    if not str(fragment).startswith("Chunk: ")
                ],
                "chunk_count": 0,
                "chunk_indices": [],
            }
            grouped_index[group_key] = grouped_display
            grouped_sources.append(grouped_display)

        grouped_index[group_key]["chunk_count"] = (
            int(grouped_index[group_key]["chunk_count"]) + 1
        )
        chunk_index = _parse_chunk_index(
            parsed_source["metadata"].get("chunk")
            if parsed_source is not None
            else None
        )
        if chunk_index is not None:
            grouped_index[group_key]["chunk_indices"].append(chunk_index)

    for source_display in grouped_sources:
        source_display["chunk_indices"] = sorted(source_display["chunk_indices"])

    return grouped_sources


def format_grouped_source_section_label(source_display: dict[str, object]) -> str:
    chunk_count = int(source_display["chunk_count"])
    section_count_label = (
        "1 relevant section"
        if chunk_count == 1
        else f"{chunk_count} relevant sections"
    )
    chunk_indices = source_display.get("chunk_indices")
    if not isinstance(chunk_indices, list) or not chunk_indices:
        return section_count_label

    chunk_label = "chunk" if len(chunk_indices) == 1 else "chunks"
    return (
        f"{section_count_label} "
        f"({chunk_label}: "
        + ", ".join(str(index) for index in chunk_indices)
        + ")"
    )


def format_official_docs_library_label(value: object) -> str | None:
    labels = {
        "langchain": "LangChain",
        "openai": "OpenAI",
        "streamlit": "Streamlit",
        "chroma": "Chroma",
    }
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    return labels.get(cleaned, cleaned.title())


def format_official_docs_provider_label(value: object) -> str | None:
    labels = {
        "official_mcp": "Provider: Official MCP",
        "official_fallback": "Provider: Local official-docs fallback",
    }
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    return labels.get(cleaned, f"Provider: {cleaned.replace('_', ' ')}")


def _clean_display_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _format_source_metadata_fragment(key: str, value: str) -> str:
    labels = {
        "topic": "Topic",
        "library": "Library",
        "doc_type": "Type",
        "difficulty": "Difficulty",
        "error_family": "Error family",
        "chunk": "Chunk",
    }
    label = labels.get(key, key.replace("_", " ").title())
    return f"{label}: {value}"


def _parse_chunk_index(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        return int(value)
    except ValueError:
        return None
