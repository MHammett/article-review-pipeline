"""ci_core.llm — the shared LLM call layer.

This is the single path by which every CI tool talks to a model provider. It is
synchronous, streaming, and built on ``requests``:

  * :mod:`ci_core.llm.streaming` — SSE parsing and per-provider accumulators,
    plus :func:`~ci_core.llm.streaming.stream_timeout` (the inter-token
    read-gap timeout).
  * :mod:`ci_core.llm.json_utils` — JSON extraction from model output,
    including salvage of arrays truncated at the output-token ceiling.
  * :mod:`ci_core.llm.adapters` — the six provider adapters. Each exposes
    ``call(system_prompt, user_prompt, api_key, ...)`` returning a generic
    result dict (``failed``, ``data``, ``raw``, ``model``, ``tokens``,
    ``elapsed_seconds``, ...).
  * :mod:`ci_core.llm.cost` — token/cost calculation from ``configs/pricing.yaml``.
  * :mod:`ci_core.llm.timeout_model` — the sliding-scale wall-clock backstop
    (size x model x effort) from ``configs/timeouts.yaml``.
  * :mod:`ci_core.llm.model_registry` — model deprecation tracking from
    ``configs/model_registry.yaml``.

Callers that just want the assembled text back — rather than the adapters'
JSON-parsing verdict — should use :func:`ci_core.llm.adapters.call_text`.
"""

from . import cost, json_utils, model_registry, streaming, timeout_model
from .adapters import call_provider, call_text

__all__ = [
    "call_provider",
    "call_text",
    "cost",
    "json_utils",
    "model_registry",
    "streaming",
    "timeout_model",
]
