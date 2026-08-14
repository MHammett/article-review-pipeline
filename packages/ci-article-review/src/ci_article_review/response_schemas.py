"""JSON Schemas for each review domain's response.

Why
---
Every provider was being asked for JSON in prose ("RETURN FORMAT (JSON only, no
other text)") and then trusted to comply. When one didn't, the pipeline reported
``Malformed JSON response`` — a message that says nothing about the cause and,
on 2026-08-11, sent a debugging session after a JSON parser when the real answer
was an empty OpenAI wallet. Genuine malformed responses happen too: a pass was
lost to one on 2026-08-10.

All five active providers can enforce a schema server-side instead. Verified
live 2026-08-12:

  * OpenAI     ``text.format`` json_schema  — works, and composes with reasoning
  * Grok       ``response_format`` json_schema
  * Mistral    ``response_format`` json_schema
  * Perplexity ``response_format`` json_schema  (previously had NO enforcement
                                                 of any kind, not even json_object)
  * Gemini     ``responseSchema``  — but **not** alongside ``google_search``:
                "Tool use with a response mime type: 'application/json' is
                unsupported" (HTTP 400). Gemini's grounded fact-check therefore
                stays prompt-based; only its ungrounded fallback can use this.

Strictness
----------
OpenAI's ``strict: true`` requires every property to appear in ``required`` and
``additionalProperties: false`` throughout. Fields the prompts describe as
optional (fact_check's ``note``) are therefore declared required but nullable —
the model can still decline to fill them, it just has to say so explicitly.

These schemas are the prompts' ``RETURN FORMAT`` blocks transcribed. If a prompt's
shape changes, this must change with it; ``test_response_schemas`` asserts every
top-level key here still appears in the corresponding prompt file, so the two
cannot drift silently.
"""

_OBSERVATION_ITEM = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "passage": {"type": "string"},
        "observation": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["category", "passage", "observation", "confidence"],
    "additionalProperties": False,
}


def _array(item_schema):
    return {"type": "array", "items": item_schema}


def _obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_STR = {"type": "string"}
_NULLABLE_STR = {"type": ["string", "null"]}
_CONFIDENCE = {"type": "string", "enum": ["high", "medium", "low"]}


VOICE_STYLE = _obj(
    {
        "flags": _array(
            _obj(
                {
                    "passage": _STR,
                    "problem": _STR,
                    "suggested_rewrite": _STR,
                    "steelman_considered": _STR,
                }
            )
        ),
        "low_confidence": _array(_obj({"passage": _STR, "observation": _STR})),
        "additional_observations": _array(_OBSERVATION_ITEM),
    }
)

ARGUMENT_INTEGRITY = _obj(
    {
        "flags": _array(
            _obj(
                {
                    "passage": _STR,
                    "logical_problem": _STR,
                    "steelman_considered": _STR,
                    "why_it_survived": _STR,
                }
            )
        ),
        "low_confidence": _array(_obj({"passage": _STR, "observation": _STR})),
        "additional_observations": _array(_OBSERVATION_ITEM),
    }
)

COMPLETENESS = _obj(
    {
        "flags": _array(
            _obj(
                {
                    "what_is_missing": _STR,
                    "passage_reference": _STR,
                    "audience_affected": _STR,
                    "steelman_considered": _STR,
                }
            )
        ),
        "low_confidence": _array(
            _obj({"observation": _STR, "passage_reference": _NULLABLE_STR})
        ),
        "additional_observations": _array(_OBSERVATION_ITEM),
    }
)

FACT_CHECK = _obj(
    {
        "confirmed": _array(
            _obj(
                {
                    "claim": _STR,
                    "source": _STR,
                    "confidence": _CONFIDENCE,
                    # Described as optional in the prompt; nullable rather than
                    # absent, so the schema can stay strict.
                    "note": _NULLABLE_STR,
                }
            )
        ),
        "outdated": _array(
            _obj(
                {
                    "claim": _STR,
                    "current_value": _STR,
                    "source": _STR,
                    "confidence": _CONFIDENCE,
                }
            )
        ),
        "contradicted": _array(
            _obj(
                {
                    "claim": _STR,
                    "contradiction": _STR,
                    "source": _STR,
                    "confidence": _CONFIDENCE,
                }
            )
        ),
        "unverifiable": _array(_obj({"claim": _STR, "checked": _STR, "reason": _STR})),
        "primary_source_needed": _array(
            _obj({"claim": _STR, "best_candidate_source": _STR})
        ),
        "additional_observations": _array(_OBSERVATION_ITEM),
    }
)

RED_TEAM = _obj(
    {
        "most_vulnerable_claim": _obj(
            {
                "passage": _STR,
                "attack_vector": _STR,
                "supporting_evidence_for_attack": _STR,
            }
        ),
        "highest_audience_risk": _obj(
            {"passage": _STR, "risk": _STR, "audience_segment": _STR}
        ),
        "highest_credibility_risk": _obj(
            {"passage": _STR, "risk": _STR, "attack_vector": _STR}
        ),
        "additional_observations": _array(_OBSERVATION_ITEM),
    }
)


BY_DOMAIN = {
    "voice_style": VOICE_STYLE,
    "argument_integrity": ARGUMENT_INTEGRITY,
    "completeness": COMPLETENESS,
    "fact_check": FACT_CHECK,
    "red_team": RED_TEAM,
}


def for_domain(domain):
    """Return the JSON Schema for ``domain``, or ``None`` if it has none.

    ``None`` for custom domains supplied by a publication config: their prompt is
    user-written, so there is no shape to enforce. Those keep the previous
    prose-requested-JSON behaviour.
    """
    return BY_DOMAIN.get(domain)
