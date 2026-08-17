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
    # Cached input is still input, and whether it is already counted depends on
    # which spelling of the total arrived.
    #
    # Anthropic's own API reports ``input_tokens`` as the *uncached remainder*
    # and puts the rest in separate cache fields, so reading that key alone
    # under-reports a cached call enormously — a 4,800-token system prompt
    # showed up as 20. There the cache fields must be added.
    #
    # litellm does not pass that shape through. It normalises to
    # ``prompt_tokens``, which is the **inclusive** total, and *also* leaves the
    # cache fields in place. Adding them there double-counts: measured
    # 2026-08-16, a 5,424-token cached call reported 10,840. That was invisible
    # until caching actually started working, because every cache field was
    # zero — the bug and the feature arrive together.
    #
    # So: trust an inclusive total when one is present, and only reconstruct it
    # from the parts when all we have is the remainder.
    has_inclusive_total = any(
        key in usage for key in ("prompt", "prompt_tokens", "promptTokenCount")
    )
    if not has_inclusive_total:
        prompt += _first_int(usage, ("cache_creation_input_tokens",))
        prompt += _first_int(usage, ("cache_read_input_tokens",))
    # How much of `prompt` came from the provider's cache. Kept as a separate
    # key rather than subtracted, so `prompt` stays the true total input and only
    # cost.py needs to know cached tokens are cheaper.
    cached = 0
    if "cached" in usage:
        cached = _first_int(usage, ("cached",))
    else:
        # OpenAI reports this under two different names. Chat Completions uses
        # `prompt_tokens_details`; the Responses API uses `input_tokens_details`
        # — and the Responses API is the pipeline's primary OpenAI path, so
        # reading only the Chat Completions name reported 0 cached tokens for
        # every OpenAI call in every run while the cache was in fact serving
        # ~95% of the prompt. Measured against the live API 2026-08-15.
        #
        # These are alternative spellings of one number, not parts of a sum.
        # litellm reports Anthropic's cache reads in BOTH
        # ``prompt_tokens_details.cached_tokens`` and
        # ``cache_read_input_tokens``; adding them reported 10,832 cached tokens
        # for a call that read 5,416. First one wins.
        for key in ("prompt_tokens_details", "input_tokens_details"):
            details = usage.get(key)
            if isinstance(details, dict):
                cached = _first_int(details, ("cached_tokens",))
                if cached:
                    break
        if not cached:
            cached = _first_int(usage, ("cache_read_input_tokens",))
        if not cached:
            cached = _first_int(usage, ("cachedContentTokenCount",))

    out = {"prompt": prompt, "completion": completion}
    if cached:
        out["cached"] = min(cached, prompt)
    return out
