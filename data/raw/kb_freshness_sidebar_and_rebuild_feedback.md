---
title: Knowledge Base Freshness Sidebar and Rebuild Feedback
topic: chroma
library: chroma
doc_type: how_to
difficulty: intermediate
error_family: persistence
---
# Knowledge Base Freshness Sidebar and Rebuild Feedback

## When This Matters

Use this document when the sidebar KB status is confusing, when the rebuild button appears or disappears unexpectedly, or when you need to understand how rebuild success and error feedback is shown to the user. This file is about the UI-facing freshness layer in `app.py`, not the lower-level Chroma persistence model itself.

The key project terms are:

- `get_kb_status(...)`
- `format_kb_status_label(...)`
- `should_show_kb_rebuild_trigger(...)`
- `KB_REBUILD_FEEDBACK_KEY`
- `run_kb_rebuild_action(...)`

## Recommended Pattern

The KB sidebar behavior is intentionally simple:

1. compute a `KBStatusResult` with `get_kb_status(settings)`
2. render a short label such as `Status: Missing`
3. render the detailed status explanation from `KBStatusResult.detail`
4. show the rebuild button only when the state is `missing` or `outdated`
5. store one short rebuild feedback message in `kb_rebuild_feedback`
6. display that feedback once on the next sidebar render and then clear it

The freshness states mean:

- `missing`
  no usable local Chroma index was found
- `outdated`
  an index exists, but it does not match the current raw markdown snapshot or manifest expectations
- `up_to_date`
  the local index matches the current raw markdown snapshot

This is a user-facing summary of the lower-level checks in `src/kb_status.py`. The sidebar is not making its own freshness decisions. It renders the result of `get_kb_status(...)`.

## Common Failure Modes

- The rebuild button is shown when the user thinks the KB should be healthy.
  This usually means the status is `missing` or `outdated`, which is exactly when `should_show_kb_rebuild_trigger(...)` returns `True`.

- A rebuild succeeded, but the message disappears quickly.
  That is intentional. `render_help_section(...)` does `st.session_state.pop(KB_REBUILD_FEEDBACK_KEY, None)`, so feedback is shown once and then cleared.

- The sidebar keeps saying `Outdated` after a rebuild.
  That usually means the rebuilt index still does not match the expected raw snapshot or collection metadata. The sidebar is just reflecting `get_kb_status(...)`.

- A rebuild error is shown without a rerun.
  That is also intentional. Failed rebuilds stay on the current run and show an error message immediately instead of rerunning the app.

- A successful rebuild changes too much session state.
  In the current design it should not. Rebuild success clears the cached vector store and preserves conversation history, export state, model selection, and rate-limit timestamps.

## Implementation Notes

The sidebar rendering path lives in `render_help_section(...)` in `app.py`.

The relevant UI helpers are:

- `format_kb_status_label(status)`
- `should_show_kb_rebuild_trigger(status)`
- `build_kb_rebuild_success_message(result)`
- `build_kb_rebuild_error_message(exc)`
- `run_kb_rebuild_action(...)`

The feedback lifecycle is:

1. user clicks `Rebuild knowledge base`
2. the app opens `st.status("Rebuilding knowledge base...", expanded=True)`
3. `run_kb_rebuild_action(...)` calls `rebuild_knowledge_base(settings)`
4. on success, the app clears the cached vector store and stores a feedback dict under `kb_rebuild_feedback`
5. the app reruns
6. the next sidebar render pops the stored feedback and shows `st.success(...)`

On failure, `run_kb_rebuild_action(...)` returns a feedback dict with:

- `kind: "error"`
- a user-facing message from `build_kb_rebuild_error_message(...)`

In that case the app updates the status widget to error and calls `st.error(...)` without storing success feedback or rerunning.

The sidebar also shows the manual rebuild command when it is available:

- `python build_index.py`

That keeps the user-facing path consistent with the lower-level KB tools. The sidebar is not trying to implement a second rebuild system; it is a presentation layer over the same backend rebuild flow and KB status logic.

## Retrieval Hints

- `What do missing outdated and up_to_date mean in get_kb_status?`
- `Why does the sidebar show Status Missing or Status Outdated?`
- `When does should_show_kb_rebuild_trigger display the rebuild button?`
- `How does kb_rebuild_feedback work in the Streamlit sidebar?`
- `Why is rebuild feedback shown once and then cleared on the next rerun?`
- `What happens in run_kb_rebuild_action after a successful rebuild?`
- `How does the app present KB freshness and rebuild state to the user?`
