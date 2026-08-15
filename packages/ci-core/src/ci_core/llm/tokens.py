"""Normalization of provider-native token counts to the shared contract.

Every adapter returns ``tokens`` as ``{"prompt": int, "completion": int}``.
That shape is a contract: :mod:`ci_core.llm.cost` reads exactly those two keys,
and the run summary prints them. An adapter that passes a provider's own usage
dict through instead produces silent zeros downstream — no exception, just a
call that appears to have cost nothing.

The providers disagree on spelling:

===========  ===========================  ==============================
provider     prompt key                   completion key
===========  ===========================  ==============================
openai       ``prompt_tokens``            ``completion_tokens``
  (Responses) ``input_tokens``            ``output_tokens``
claude       ``input_tokens``             ``output_tokens``
gemini       ``promptTokenCount``         ``candidatesTokenCount``
grok         ``prompt_tokens``            ``completion_tokens``
mistral      ``prompt_tokens``            ``completion_tokens``
perplexity   ``prompt_tokens``            ``completion_tokens``
===========  ===========================  ==============================

:func:`normalize_tokens` accepts any of them — plus an already-normalized dict,
so it is safe to apply twice — and always returns the two-key contract.
"""

# Provider-native aliases, most specific first. Already-normalized keys are
# listed too so normalize_tokens is idempotent.
_PROMPT_KEYS = (
    "prompt",
    "prompt_tokens",
    "input_tokens",
    "promptTokenCount",
)
_COMPLETION_KEYS = (
    "completion",
    "completion_tokens",
    "output_tokens",
    "candidatesTokenCount",
)


def _first_int(usage, keys):
    """Return the first key in ``keys`` present in ``usage`` as an int, or 0."""
    for key in keys:
        value = usage.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def normalize_tokens(usage):
    """Return ``{"prompt": int, "completion": int}`` from any provider's usage dict.

    Unknown, empty, or non-dict input normalizes to zeros rather than raising —
    a missing token count must not turn a usable response into a failure.

    Gemini's ``thoughtsTokenCount`` is added to ``completion``. It is disjoint
    from ``candidatesTokenCount`` — Gemini's own ``totalTokenCount`` equals
    ``promptTokenCount + candidatesTokenCount + thoughtsTokenCount`` — and
    Google bills thinking tokens at the output rate, so leaving it out
    understates cost by however much the model thought (frequently more than
    the visible answer itself).
    """
    if not isinstance(usage, dict):
        return {"prompt": 0, "completion": 0}
    completion = _first_int(usage, _COMPLETION_KEYS)
    if "thoughtsTokenCount" in usage and "completion" not in usage:
        # Guarded on the normalized key so re-normalizing an already-normalized
        # dict does not add the thinking tokens a second time.
        completion += _first_int(usage, ("thoughtsTokenCount",))

    prompt = _first_int(usage, _PROMPT_KEYS)
    # Cached input is still input. Anthropic reports it in separate fields and
    # drops ``input_tokens`` to only the uncached remainder, so reading that key
    # alone under-reports a cached call enormously — a 4,800-token system prompt
    # showed up as 20. Both cache fields are disjoint from ``input_tokens`` and
    # from each other, so they add. Guarded on the normalized key so
    # re-normalizing does not double-count.
    if "prompt" not in usage:
        prompt += _first_int(usage, ("cache_creation_input_tokens",))
        prompt += _first_int(usage, ("cache_read_input_tokens",))
    # How much of `prompt` came from the provider's cache. Kept as a separate
    # key rather than subtracted, so `prompt` stays the true total input and only
    # cost.py needs to know cached tokens are cheaper.
    cached = 0
    if "cached" in usage:
        cached = _first_int(usage, ("cached",))
    else:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached += _first_int(details, ("cached_tokens",))
        cached += _first_int(usage, ("cache_read_input_tokens",))
        cached += _first_int(usage, ("cachedContentTokenCount",))

    out = {"prompt": prompt, "completion": completion}
    if cached:
        out["cached"] = min(cached, prompt)
    return out
