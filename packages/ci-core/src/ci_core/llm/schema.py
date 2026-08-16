"""Asking a provider to guarantee a response *shape*, not just valid JSON.

``response_format: {"type": "json_object"}`` buys well-formedness and nothing
else — the model still picks its own keys. Measured 2026-08-16 against
gpt-5.4-mini with a loose instruction, it answered with ``{"ai_speak": ...,
"suggestion": ...}`` when the caller wanted ``{"flags": [...]}``. Downstream
reads named buckets, so a response like that contributes nothing while looking
like a success.

A JSON *schema* removes the question. Verified live on 2026-08-16, every
provider here enforces one:

===========  =========================  ===================================
provider     how                        result
===========  =========================  ===================================
openai       ``text.format``            exact schema (NOT ``response_format``,
                                        which this API accepts and ignores)
grok         ``response_format``        exact schema
mistral      ``response_format``        exact schema, with or without
                                        reasoning_effort
perplexity   ``response_format``        exact schema
claude       ``response_format``        exact schema
gemini       ``response_format``        exact schema **only when ungrounded**
===========  =========================  ===================================

Gemini is the one exception and it is a hard one: with the ``googleSearch``
tool active, a schema request is a 400 from the provider. Fact-checking is
exactly where grounding matters most, so gemini keeps prompt-only JSON there
rather than losing its search. That constraint was documented in the adapter
this replaced, and it still holds — unlike the two claims beside it, that
Perplexity does not support schemas and that Mistral cannot combine one with
reasoning, both of which were checked and are no longer true.

This module knows nothing about review domains. The schemas themselves live
with the prompts they mirror, in ci-article-review; ci-core only knows how to
put one on the wire for a given provider. See docs/NAMING.md on dependency
direction.
"""

__all__ = ["SCHEMA_CAPABLE", "supports_schema", "as_request_params"]

#: Providers that enforce a JSON schema. Gemini is present but conditional —
#: see :func:`supports_schema`.
SCHEMA_CAPABLE = frozenset(
    {"openai", "grok", "mistral", "perplexity", "claude", "gemini"}
)


def supports_schema(provider, grounded=False):
    """Whether ``provider`` will honour a schema on this particular call.

    ``grounded`` is Gemini's disqualifier and nobody else's: the search tool and
    a schema request cannot coexist, and the provider rejects the combination
    outright rather than degrading.
    """
    if provider not in SCHEMA_CAPABLE:
        return False
    if provider == "gemini" and grounded:
        return False
    return True


def as_request_params(provider, name, schema, grounded=False):
    """Request parameters that put ``schema`` on the wire, or ``{}``.

    Returns ``{}`` when the provider cannot take one, so callers can merge the
    result unconditionally instead of branching at every call site.

    ``strict`` is requested where the provider offers it. It is the difference
    between a schema the model treats as a suggestion and one the API enforces,
    and the whole point here is the guarantee.
    """
    if not supports_schema(provider, grounded=grounded):
        return {}

    if provider == "openai":
        # The Responses API spells this `text.format`, not `response_format`.
        # It accepts `response_format` too — and ignores it, which is worse than
        # rejecting it, because the call succeeds and the shape is not enforced.
        return {
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "strict": True,
                    "schema": schema,
                }
            }
        }

    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema, "strict": True},
        }
    }
