"""Provider adapters — one module per provider, one uniform entry point.

Every adapter module exposes::

    call(system_prompt, user_prompt, api_key,
         retry=True, retry_delay=10, model=None, provider_config=None) -> dict

and returns a provider-generic result dict:

  ``failed``           bool
  ``raw``              the assembled response text (present on success, and on
                       failures where text was actually received)
  ``data``             the parsed JSON payload (success only)
  ``model``            the model that actually answered (may differ from the
                       requested one when an adapter fell back)
  ``tokens``           ``{"prompt": int, "completion": int}``
  ``elapsed_seconds``  float
  ``error``            redacted exception text (failures only)
  ``error_body``       redacted excerpt of the HTTP error response body

plus per-provider extras (``grounding_available``, ``citations``,
``truncated``, ``fallback_from``, ...).

Backend selection within a provider — Azure vs openai.com, Vertex AI vs AI
Studio — is handled inside each adapter from ``provider_config["provider"]``.
Callers dispatch on the *adapter* name (``openai``, ``gemini``, ...), not the
backend.
"""

import importlib
import logging

log = logging.getLogger(__name__)

# Adapter name -> module name within this package.
ADAPTER_MODULES = {
    "openai": "openai",
    "gemini": "gemini",
    "mistral": "mistral",
    "grok": "grok",
    "claude": "claude",
    "perplexity": "perplexity",
}

__all__ = ["ADAPTER_MODULES", "get_adapter", "call_provider", "call_text"]


def get_adapter(name):
    """Import and return the adapter module for ``name``.

    Raises ``KeyError`` for an unknown adapter name. Imports lazily so a caller
    that only uses two providers does not pay for loading all six.
    """
    if name not in ADAPTER_MODULES:
        raise KeyError(
            f"Unknown adapter {name!r}. Known adapters: "
            f"{', '.join(sorted(ADAPTER_MODULES))}"
        )
    return importlib.import_module(f".{ADAPTER_MODULES[name]}", __package__)


def call_provider(
    name,
    system_prompt,
    user_prompt,
    api_key,
    retry=True,
    retry_delay=10,
    model=None,
    provider_config=None,
):
    """Call adapter ``name`` and return its raw result dict unchanged."""
    adapter = get_adapter(name)
    return adapter.call(
        system_prompt,
        user_prompt,
        api_key,
        retry=retry,
        retry_delay=retry_delay,
        model=model,
        provider_config=provider_config,
    )


def call_text(
    name,
    system_prompt,
    user_prompt,
    api_key,
    retry=True,
    retry_delay=10,
    model=None,
    provider_config=None,
):
    """Call adapter ``name`` and return the assembled text, not parsed JSON.

    The adapters parse their response as JSON and report a non-JSON body as a
    failure, because the review pipeline requires structured findings. Callers
    that do their own parsing downstream (ci-style-profile runs
    :func:`ci_core.llm.json_utils.extract_json` over several passes, and feeds
    some model output back into a later prompt verbatim) need the text itself,
    and a prose response is usable to them rather than fatal.

    So a ``"Malformed JSON response"`` failure carrying ``raw`` text is
    reported here as a success with that text. Every other failure — HTTP
    error, timeout, empty response, in-band stream error — stays a failure.

    Returns ``{"content": str, "failed": bool, "tokens": dict,
    "elapsed": float, "model": str}``, plus ``error`` when failed.
    """
    result = call_provider(
        name,
        system_prompt,
        user_prompt,
        api_key,
        retry=retry,
        retry_delay=retry_delay,
        model=model,
        provider_config=provider_config,
    )

    failed = bool(result.get("failed"))
    raw = result.get("raw") or ""

    if (
        failed
        and raw
        and str(result.get("error", "")).startswith("Malformed JSON response")
    ):
        # Not a transport failure — the model answered, just not in JSON.
        # The caller parses this itself.
        log.debug(
            "%s returned non-JSON content; passing %d chars through as text",
            name,
            len(raw),
        )
        failed = False

    out = {
        "content": raw,
        "failed": failed,
        "tokens": result.get("tokens") or {},
        "elapsed": result.get("elapsed_seconds", 0.0),
        "model": result.get("model", model or name),
    }
    if failed:
        out["error"] = result.get("error", "")
        if result.get("error_body"):
            out["error_body"] = result["error_body"]
    return out
