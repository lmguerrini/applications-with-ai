---
title: Streamlit Chat Patterns for RAG Interfaces
topic: streamlit
library: streamlit
doc_type: example
difficulty: intermediate
error_family: ui
---
# Streamlit Chat Patterns for RAG Interfaces

## When This Matters

Use this document when the Streamlit app can technically answer questions, but the chat UI feels unclear, inconsistent, or hard to trust. This file is about visible chat behavior in `app.py`: how previous turns are rendered, how different response paths are labeled, where sources and structured results appear, and how the assistant keeps grounded answers visually distinct from tool output and official-docs output.

This is the right document when you are working on:

- `render_latest_turn()`
- turn-by-turn chat rendering with `st.chat_message(...)`
- response captions and explanation text
- `Sources`, `Tool Result`, and `Official Docs Result` expanders
- usage display with `format_request_usage_label(...)`
- export-friendly turn records built by `build_turn_record(...)`

If the problem is mostly about `st.session_state` initialization, `request_timestamps`, or rerun persistence, the companion file `streamlit_session_state_and_chat_history.md` is the better fit. This file stays focused on what the user sees inside each chat turn.

## Recommended Pattern

The current app uses a simple, reviewable chat-rendering pattern:

1. Rebuild the visible conversation from `conversation_history`.
2. Render each stored user question in a `user` message block.
3. Render each stored assistant result in an `assistant` message block.
4. Show a short response-type caption before the answer text.
5. Show a second short summary line that explains what kind of answer the user is looking at.
6. Render the answer body.
7. Put supporting details into focused expanders instead of mixing them directly into the main answer text.

The response type distinctions are important:

- `Grounded answer`
- `Official docs answer`
- `Tool result`
- `No-context fallback`

Those labels come from `get_response_type_label(...)`, and the one-line explanation comes from `get_response_summary_line(...)`. The fuller explanation lives in the `How This Answer Was Generated` expander through `get_response_generation_explanation(...)`.

This pattern works well for a project-specific RAG app because the user can immediately tell whether the answer came from:

- local knowledge-base retrieval
- official documentation evidence
- a deterministic built-in tool
- no usable local context

That clarity matters more than flashy UI behavior. The current frontend is intentionally conservative.

## Common Failure Modes

- The answer text is present, but the route is unclear.
  If you do not show the response type and summary captions, users cannot tell the difference between a grounded answer and a fallback. In this project, that difference is central.

- Tool output is dumped into the main answer body.
  Tool responses should stay structured. The app currently keeps the top-level answer short and shows the full payload under `Tool Result`.

- Official docs evidence is treated like local KB sources.
  The UI should not mix these paths. Official-docs answers use the `Official Docs Result` expander and do not appear under the normal `Sources` section.

- Source metadata is hard to scan.
  Raw source strings are not ideal for direct display. The app parses them with `parse_source_string(...)` and formats them with `format_source_display(...)` so the user sees a readable title, metadata fragments, and optional source path.

- Usage captions appear in the wrong place.
  In the current UI, usage belongs at the bottom of the assistant turn, after sources or structured result blocks. That keeps the main answer readable while preserving cost and token visibility.

- No-context fallback looks like a failure instead of an intentional safeguard.
  The UI should make it clear that the app deliberately refused to generate a grounded answer because retrieval did not provide safe context.

## Implementation Notes

The assistant turn in `render_latest_turn()` follows a stable order:

- response type caption
- summary caption
- answer text
- `How This Answer Was Generated` expander
- optional `Tool Result` expander
- optional `Official Docs Result` expander
- optional `Sources` expander
- usage caption from `format_request_usage_label(...)`

That order is useful because it moves from the highest-signal information to the lower-level details. The user sees the answer first, then the mechanism, then the evidence, then the token/cost line.

Turn records also need to stay export-friendly. `build_turn_record(...)` stores:

- `query`
- `answer`
- `used_context`
- `sources`
- `tool_result`
- `official_docs_result`
- `usage`

That is enough to reconstruct the chat in the live UI and in Markdown, JSON, CSV, and PDF exports without re-running the backend.

For source-heavy answers, compact formatting matters. The app’s readable source display helps more than dumping raw metadata because grounded answers often include several source fields such as `topic`, `library`, `doc_type`, `difficulty`, `chunk`, and `error_family`.

## Retrieval Hints

- `How does render_latest_turn work in app.py?`
- `How does the Streamlit UI show grounded answers versus tool results?`
- `Where does the app label Official docs answer versus No-context fallback?`
- `How are Sources Tool Result and Official Docs Result rendered in the chat UI?`
- `How does format_request_usage_label display token usage in each assistant turn?`
- `What fields are stored in build_turn_record for Streamlit chat rendering and exports?`
- `How does the app format source metadata for display in the chat interface?`
