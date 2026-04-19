from rendering.structured_display import (
    format_grouped_source_section_label,
    group_source_displays,
)


def test_group_source_displays_groups_chunks_by_title_and_source_path() -> None:
    grouped_sources = group_source_displays(
        [
            "Chroma Persistence Guide | topic=chroma | library=chroma | "
            "source=data/raw/chroma_persistence_guide.md | chunk=9",
            "Chroma Persistence Guide | topic=chroma | library=chroma | "
            "source=data/raw/chroma_persistence_guide.md | chunk=1",
            "Chroma Persistence Guide | topic=chroma | library=chroma | "
            "source=data/raw/chroma_persistence_guide.md | chunk=0",
            "Streamlit Chat Patterns | topic=streamlit | library=streamlit | "
            "source=data/raw/streamlit_chat_patterns.md | chunk=0",
        ]
    )

    assert len(grouped_sources) == 2
    assert grouped_sources[0]["title"] == "Chroma Persistence Guide"
    assert grouped_sources[0]["source_path"] == "data/raw/chroma_persistence_guide.md"
    assert grouped_sources[0]["chunk_count"] == 3
    assert grouped_sources[0]["chunk_indices"] == [0, 1, 9]
    assert grouped_sources[0]["metadata_fragments"] == [
        "Topic: chroma",
        "Library: chroma",
    ]
    assert grouped_sources[1]["title"] == "Streamlit Chat Patterns"
    assert grouped_sources[1]["chunk_count"] == 1
    assert grouped_sources[1]["chunk_indices"] == [0]


def test_group_source_displays_keeps_same_title_with_different_paths_separate() -> None:
    grouped_sources = group_source_displays(
        [
            "Guide | source=data/raw/first.md | chunk=0",
            "Guide | source=data/raw/second.md | chunk=0",
        ]
    )

    assert len(grouped_sources) == 2
    assert [source["source_path"] for source in grouped_sources] == [
        "data/raw/first.md",
        "data/raw/second.md",
    ]
    assert [source["chunk_count"] for source in grouped_sources] == [1, 1]
    assert [source["chunk_indices"] for source in grouped_sources] == [[0], [0]]


def test_group_source_displays_preserves_malformed_sources_as_individual_rows() -> None:
    grouped_sources = group_source_displays(
        [
            "Malformed Source | bad-fragment",
            "Malformed Source | bad-fragment",
        ]
    )

    assert len(grouped_sources) == 2
    assert all(source["parse_failed"] for source in grouped_sources)
    assert [source["chunk_count"] for source in grouped_sources] == [1, 1]
    assert [source["chunk_indices"] for source in grouped_sources] == [[], []]


def test_format_grouped_source_section_label_includes_chunk_indices() -> None:
    assert (
        format_grouped_source_section_label(
            {
                "chunk_count": 1,
                "chunk_indices": [0],
            }
        )
        == "1 relevant section (chunk: 0)"
    )
    assert (
        format_grouped_source_section_label(
            {
                "chunk_count": 3,
                "chunk_indices": [0, 1, 9],
            }
        )
        == "3 relevant sections (chunks: 0, 1, 9)"
    )
