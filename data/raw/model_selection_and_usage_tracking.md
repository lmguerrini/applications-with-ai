---
title: Model Selection and Usage Tracking
topic: streamlit
library: openai
doc_type: how_to
difficulty: intermediate
---
# Model Selection and Usage Tracking

## When This Matters

Use this document when the selected chat model seems wrong, when usage lines in the UI are confusing, or when session-level totals do not match what you expected from individual requests. This file covers the selection and tracking path for LLM-backed responses in the current app, not general OpenAI billing advice.

The key project terms are:

- `selected_chat_model`
- `validate_selected_chat_model(...)`
- `format_request_usage_label(...)`
- `build_session_usage_totals(...)`
- `format_session_usage_label(...)`

## Recommended Pattern

The app keeps model selection simple and explicit.

The flow is:

1. initialize the session model key from `settings.default_chat_model`
2. show the sidebar `Chat model` selectbox
3. validate the selected value with `validate_selected_chat_model(...)`
4. build the `ChatOpenAI` client through `get_chat_model(...)`
5. run the backend request
6. store real request usage in the turn record if usage metadata exists

The supported chat models currently come from `SUPPORTED_CHAT_MODELS` in `src/config.py`:

- `gpt-4.1-mini`
- `gpt-4.1`
- `gpt-4o-mini`

Validation is strict. `validate_selected_chat_model(...)` rejects blank values and then delegates to `settings.ensure_supported_chat_model(...)`. That keeps the UI selection, model cache key, and actual `ChatOpenAI` client aligned.

Usage tracking is also explicit. The app only records real usage metadata from model responses through `extract_request_usage(...)` in `src/llm_response_utils.py`. It does not invent token counts for tool results or no-context fallback.

## Common Failure Modes

- The selected model appears to reset.
  The app stores the sidebar selection in the session key `selected_chat_model`. If that key is missing or invalid, the UI falls back to `settings.default_chat_model`.

- A supported model is selected, but the request still fails.
  Validation only checks whether the model is in the supported list. Access can still fail later if the API key does not have permission for that model. The app maps that case to a user-facing unavailable-model message.

- A grounded answer shows `Usage unavailable`.
  That means the request used an LLM path, but the response metadata did not include usable token counts. The app preserves that uncertainty instead of estimating.

- Session totals seem too low.
  `build_session_usage_totals(...)` only includes turns where `usage` is a real dictionary. Tool results and no-context fallback turns do not contribute because they have no tracked LLM usage.

- Users expect every answer type to show token counts.
  That is not the current behavior. Different answer paths intentionally report usage differently.

## Implementation Notes

Per-turn usage behavior is:

- grounded answer with usage metadata:
  `format_request_usage_label(...)` shows model name, input tokens, output tokens, total tokens, and cost
- grounded answer without usage metadata:
  `format_request_usage_label(...)` shows `Usage unavailable`
- official-docs answer with usage metadata:
  the same label format is used because `official_docs_result.usage` is passed through to the turn record
- official-docs answer without usage metadata:
  `Usage unavailable`
- tool result:
  `No LLM usage`
- no-context fallback:
  `No LLM usage`

Session totals come from `build_session_usage_totals(...)`, which aggregates only real usage entries from `conversation_history`. `format_session_usage_label(...)` then renders a compact summary such as:

- request count
- total tokens
- total estimated cost when every included turn has numeric cost data

The cost calculation itself comes from `estimate_cost_usd(...)` in `src/llm_response_utils.py`, which uses the pricing table for supported chat models. If the model name is missing or not priced, cost stays `None`.

Model caching also depends on explicit model selection. `build_chat_model_cache_inputs(...)` includes:

- `api_key`
- `model_name`
- `temperature`

That means switching models changes the cache key for the chat model resource without touching unrelated caches such as exports or the vector store.

## Retrieval Hints

- `How does selected_chat_model work in this Streamlit app?`
- `What models are supported by validate_selected_chat_model?`
- `Why does format_request_usage_label show Usage unavailable?`
- `How are per-turn usage and session totals tracked in conversation_history?`
- `Why do tool results show No LLM usage in the UI?`
- `How does format_session_usage_label calculate request count token totals and cost?`
- `How does the app cache ChatOpenAI clients when the selected model changes?`
