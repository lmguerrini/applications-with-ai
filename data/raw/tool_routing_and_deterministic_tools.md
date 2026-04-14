---
title: Tool Routing and Deterministic Tools
topic: tool_calling
library: general
doc_type: how_to
difficulty: intermediate
---
# Tool Routing and Deterministic Tools

## When This Matters

Use this document when a question seems like it should be answered by a built-in tool instead of the knowledge base, or when you need to understand why the app returned a structured tool result instead of a grounded answer. This project has a fixed route order in `run_backend_query(...)` inside `src/chains.py`, and deterministic tools are intentionally checked before both the official-docs path and the local KB RAG path.

This is the right document for questions such as:

- why did a pricing question not use retrieval?
- how does `maybe_invoke_tool(...)` decide whether to route a query?
- what happens when a tool query is missing parameters?
- when does the app show `Tool Result` in the UI?
- what tool families exist in this codebase?

## Recommended Pattern

The backend route order is deliberately simple:

1. try `maybe_invoke_tool(request.query)`
2. if no tool matches, try `maybe_match_official_docs_query(...)`
3. if neither matches, call `answer_query(...)` for grounded KB retrieval

That means deterministic tool routing has priority over both official docs and local retrieval. The reason is practical: if the user is really asking for a calculation or a rule-based diagnosis, the app should not spend tokens on KB retrieval or answer generation first.

The current tool families are:

- `estimate_openai_cost`
- `diagnose_stack_error`
- `recommend_retrieval_config`

These tools are not generic LLM wrappers. They are rule-based helpers implemented in `src/tools.py`, with structured input and structured output models. The tool path returns a short top-level answer through `format_tool_answer(...)`, while the full payload is preserved in `tool_result`.

This design is useful because the user gets:

- a concise result summary in the main assistant message
- a full structured payload in the `Tool Result` expander
- no misleading grounded-source display for a non-RAG answer

## Common Failure Modes

- A tool-like query falls through to KB retrieval.
  This usually means the query did not match the router strongly enough. For example, the tool path expects clear signals for price, diagnostics, or retrieval-config recommendations.

- A tool route is selected, but the result is an error-like message.
  That can be intentional. Some parsing failures return a `ToolInvocationResult` with `tool_error` instead of falling back to KB retrieval. For example, `estimate_openai_cost` can return a message saying that model, `input_tokens`, and `output_tokens` are required.

- A user expects sources for a tool answer.
  Tool answers do not use KB chunks, so `answer_sources` is empty and the normal `Sources` expander should not appear.

- A docs request gets intercepted by a tool.
  That is possible and expected if the wording matches a deterministic tool first. Tool priority is intentional.

- A tool result is mistaken for a no-context fallback.
  In the UI these are distinct. `get_response_type_label(...)` returns `Tool result` for tool-backed answers, while no-context fallback is only used when the KB retrieval path runs and returns no usable chunks.

## Implementation Notes

`maybe_invoke_tool(...)` currently checks the three tool families in a fixed order:

1. cost estimation
2. stack-error diagnosis
3. retrieval-config recommendation

That order matters for ambiguous phrasing. Once a tool builder returns a `ToolInvocationResult`, later routes do not run.

The cost tool supports both structured and natural phrasing. Examples that the parser is designed to understand include:

- `model=gpt-4.1-mini, input_tokens=1000, output_tokens=500, num_calls=3`
- `Estimate OpenAI cost for model gpt-4.1-mini with 1000 input tokens, 500 output tokens, and 3 calls`

The diagnosis tool maps common LangChain, Chroma, Streamlit, and OpenAI error patterns into structured categories such as `imports`, `persistence`, `ui`, and `api`.

The retrieval-config tool returns a deterministic recommendation for:

- `chunk_size`
- `chunk_overlap`
- `top_k`
- `use_metadata_filters`

In the UI, tool-backed answers behave like this:

- top-level answer text comes from `format_tool_answer(...)`
- `used_context` is `False`
- `tool_result` is present
- `usage` is `None`
- `format_request_usage_label(...)` shows `No LLM usage`
- the assistant turn renders a `Tool Result` expander with the full structured payload

This is a clean separation from KB and official-docs paths. The app is not trying to disguise tool output as retrieved documentation.

## Retrieval Hints

- `How does maybe_invoke_tool work in this project?`
- `What is the route priority between tools official docs and grounded KB answers?`
- `Why did the app return a Tool Result instead of sources?`
- `Which deterministic tools exist in src/tools.py?`
- `How does estimate_openai_cost parse model input_tokens output_tokens and calls?`
- `Why does a tool validation error stay on the tool path instead of falling back to RAG?`
- `How are tool results displayed in the Streamlit UI?`
