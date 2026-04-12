from dataclasses import dataclass
from pathlib import Path

from src.config import Settings, get_settings
from src.kb_status import write_kb_manifest
from src.knowledge_base import KnowledgeBaseError, build_index


@dataclass(frozen=True)
class KBRebuildResult:
    indexed_chunk_count: int
    collection_name: str
    persist_directory: Path
    manifest_path: Path


def rebuild_knowledge_base(settings: Settings) -> KBRebuildResult:
    vector_store = build_index(settings=settings)
    indexed_count = len(vector_store.get()["ids"])
    manifest_path = write_kb_manifest(
        settings=settings,
        indexed_chunk_count=indexed_count,
    )
    return KBRebuildResult(
        indexed_chunk_count=indexed_count,
        collection_name=settings.chroma_collection_name,
        persist_directory=settings.chroma_persist_dir,
        manifest_path=manifest_path,
    )


def main() -> None:
    settings = get_settings()
    result = rebuild_knowledge_base(settings)
    print(
        "Indexed "
        f"{result.indexed_chunk_count} chunks into '{result.collection_name}' "
        f"at {result.persist_directory}"
    )
    print(f"Wrote KB manifest to {result.manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except KnowledgeBaseError as exc:
        raise SystemExit(f"Knowledge base build failed: {exc}") from exc
    except ValueError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
