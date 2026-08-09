# Provider Adapter Interface

These are the shared LLM provider adapters. Both `ci-article-review` (the
review pipeline) and `ci-style-profile` (corpus synthesis) call them, so the
interface below is a cross-package contract — see `docs/NAMING.md`.

Each adapter exposes a `call()` function with this signature:

```python
def call(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    retry: bool = True,
    retry_delay: int = 10,
    model: str | None = None,
    provider_config: dict | None = None,
) -> dict:
```

`provider_config` is the normalized model config dict from
`ci_core.config_helpers.normalize_model_configs` (e.g.
`{"provider": "vertex_ai", "model": "gemini-2.5-flash", "project": "my-project", ...}`).
When `None` or `{}`, each adapter falls back to its default public-API behavior.
The `model` parameter is kept for backward compatibility; `provider_config["model"]` takes
precedence when both are supplied.

**Return dict keys:**

| Key | Type | Notes |
|---|---|---|
| `failed` | bool | True if the call did not produce usable output |
| `data` | dict | Parsed JSON response (present when `failed=False`) |
| `model` | str | Model identifier actually used |
| `tokens` | dict | `{"prompt": int, "completion": int}` |
| `error` | str | Error message (present when `failed=True`) |
| `raw` | str | Assembled response text — always present on success, and on failures where text was received |
| `fallback_from` | str | Set when the adapter fell back to a secondary model |
| `provider` | str | Set by Azure backends to identify the provider tier |

**Provider routing per adapter:**

| Adapter | `provider` values | Notes |
|---|---|---|
| `gemini.py` | `ai_studio` (default), `vertex_ai` | Vertex requires `google-auth` and a GCP project |
| `openai.py` | `openai` (default), `azure` | Azure requires `endpoint` and `deployment` |
| `mistral.py` | `mistral` (default), `azure` | Azure is an endpoint-only swap, same Bearer auth |
| `grok.py` | `grok` (default) | No alternate providers |
| `claude.py` | `anthropic` (default) | Falls back through Sonnet/Haiku on capacity (529) |
| `perplexity.py` | `perplexity` (default) | Always grounded; returns `citations` / `search_results` |

**Streaming (SSE):**

All adapters POST with `stream=True` and assemble the provider's SSE delta stream
via the shared helpers in [`../streaming.py`](../streaming.py) before running the existing
JSON parse/validation on the accumulated text. Consequences for the return dict and
config:

- Token usage is captured from the final SSE chunk (`stream_options: {include_usage: true}`
  for OpenAI-compatible providers; `message_delta` for Anthropic; cumulative
  `usageMetadata` for Gemini), so `tokens` is populated exactly as before.
- The HTTP `timeout` is a `(connect, read_gap)` tuple where `read_gap` is the
  **inter-token** stall allowance — a small constant (`streaming.DEFAULT_READ_TIMEOUT`,
  or `provider_config["stream_read_timeout"]`), **not** `timeout_seconds`. The big
  sliding-scale `timeout_seconds` is the pipeline's per-task wall-clock backstop only.
- A stall between tokens surfaces as a `requests` read timeout while iterating the
  stream; the adapter returns `failed=True` like any other call failure.

When writing a new adapter, build its payload with `stream=True`, pass
`streaming.stream_timeout(cfg, <default_read_gap>)` as the request `timeout`, and feed
the response to the matching `streaming.accumulate_*` helper.

**Two ways to call an adapter:**

- `call_provider(name, ...)` — dispatch by adapter name; returns the result dict above unchanged.
- `call_text(name, ...)` — returns the assembled text instead of the JSON verdict, for callers
  that parse it themselves. A `Malformed JSON response` failure carrying `raw` text becomes a
  success there; every other failure stays a failure. `ci-style-profile` uses this.

**To add a new adapter:**
1. Create a new file in `ci_core/llm/adapters/`
2. Implement `call()` with the signature above, including `raw` in the success return
3. Register it in `ADAPTER_MODULES` in `__init__.py`
4. For the review pipeline, also register it in `ci_article_review/pipeline.py`'s runner
   list and `consolidation.py`'s `_all_for_additional`
