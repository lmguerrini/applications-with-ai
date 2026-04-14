---
title: LangChain Retrieval Patterns for Small RAG Apps
topic: langchain
library: langchain
doc_type: concept
difficulty: intermediate
error_family: retrieval
---
# LangChain Retrieval Patterns for Small RAG Apps

## When This Matters

Use this document when you need the project’s intended retrieval design, not a troubleshooting checklist. This file describes the steady-state retrieval pattern implemented in `src/retrieval.py` and used by `answer_query(...)` in `src/chains.py`. It is the right reference for how the app is supposed to retrieve context, format sources, and hand grounded evidence to the model.

This is the best document when you are deciding:

- how `retrieve_chunks(...)` should behave in the normal path
- why `rewrite_query(...)` exists before similarity search
- when metadata filters should be used
- how fallback search fits into a fixed, non-agentic RAG flow
- what chunking and `top_k` tradeoffs make sense for compact internal markdown docs

If you are actively debugging bad retrieval outcomes, the companion document `langchain_retrieval_debugging_playbook.md` is more direct.

## Recommended Pattern

This project intentionally uses one explicit retrieval flow instead of several competing strategies.

The pattern is:

1. Validate the request with `RetrievalRequest`.
2. Rewrite the original question with `rewrite_query(...)`.
3. Infer optional metadata filters with `infer_metadata_filters(...)`.
4. Run filtered Chroma similarity search when there is a confident filter set.
5. Fall back to unfiltered similarity search when filtered retrieval returns nothing.
6. Remove weak documents with `_filter_usable_documents(...)`.
7. Convert valid documents into `RetrievedChunk` objects.
8. Format explicit source strings with `format_sources(...)`.
9. Pass only the retrieved context and sources into `answer_query(...)`.

That pattern is simple on purpose. It covers the Sprint-2 requirement for advanced RAG without turning the app into an agent system.

`rewrite_query(...)` is intentionally lightweight. It removes common noise words, normalizes case, and keeps meaningful tokens in a compact retrieval-oriented string. That helps similarity search focus on the domain terms that actually matter, such as `chroma`, `streamlit`, `persist`, `retrieval`, or `filters`.

Metadata filters are also intentionally conservative. `infer_metadata_filters(...)` only sets `topic`, `library`, `doc_type`, or `error_family` when the query contains strong signals. This is a useful middle ground:

- enough structure to narrow retrieval when the intent is obvious
- enough restraint to avoid collapsing recall for ordinary questions

## Common Failure Modes

- Overusing metadata filters.
  Filters are most helpful when the query clearly names a library or troubleshooting intent. For weaker queries, aggressive filtering can damage recall.

- Treating fallback search as a bug.
  `used_fallback=True` is often the correct outcome. It means the system tried the stricter filtered path first and then recovered with broader similarity search.

- Using large, unfocused markdown documents.
  This project’s corpus works better with compact, single-purpose docs. Retrieval quality improves when one file maps to one operational question family.

- Assuming every similarity hit is usable context.
  After retrieval, documents still pass through `_filter_usable_documents(...)`. This pattern protects the app from answering with semantically loose but weakly grounded chunks.

- Setting `top_k` too high for short internal docs.
  `RetrievalRequest` allows `top_k` from `1` to `10`, but the app default is `3`. For this corpus, a small `top_k` usually keeps prompt context tighter and easier to ground.

## Implementation Notes

Current retrieval behavior is tightly coupled to the markdown corpus structure:

- frontmatter fields become chunk metadata
- `topic`, `library`, `doc_type`, and `error_family` support filtered retrieval
- `title`, `source_path`, and `chunk_index` show up in formatted sources

The usable-document step matters almost as much as the initial Chroma search. `_filter_usable_documents(...)` compares meaningful tokens from the rewritten query against both document text and key metadata fields:

- `title`
- `topic`
- `library`
- `doc_type`
- `error_family`

That means retrieval quality depends on both chunk body wording and metadata quality.

For this codebase, chunking defaults come from `Settings`:

- `CHUNK_SIZE=800`
- `CHUNK_OVERLAP=120`

Those values are a practical default for mixed how-to and troubleshooting notes. Smaller chunks can improve precision, but if they get too small they may lose the operational detail needed for grounded answers. Larger chunks can increase recall but make source attribution and overlap filtering less crisp.

The source formatting path is also part of the pattern, not an afterthought. `format_sources(...)` produces explicit metadata-rich source lines that the app can later parse and render in a readable way.

## Retrieval Hints

- `What retrieval pattern does this LangChain RAG app use?`
- `How does retrieve_chunks work in src/retrieval.py?`
- `Why does the app rewrite the query before Chroma similarity search?`
- `When should metadata filters be used for topic library doc_type and error_family?`
- `How does filtered retrieval fall back to plain similarity search in this project?`
- `How does usable-document filtering protect grounded answers from weak Chroma matches?`
- `What chunk_size chunk_overlap and top_k pattern fits small internal markdown docs?`
