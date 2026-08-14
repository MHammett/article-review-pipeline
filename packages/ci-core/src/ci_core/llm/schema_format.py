"""Translate one JSON Schema into each provider's own way of enforcing it.

The caller supplies a plain JSON Schema (see
``ci_article_review.response_schemas``); every provider wants it wrapped
differently, and one of them cannot accept it at all in the configuration this
pipeline uses. Keeping the translation here means the adapters carry a single
line each rather than five copies of the same wrapping logic, and the
provider-specific quirks are documented once.

All shapes verified against the live APIs on 2026-08-12.
"""


def openai_text_format(schema, name="review_response"):
    """``text.format`` for the Responses API.

    ``strict`` is what makes this a guarantee rather than a hint. It requires
    every property to be listed in ``required`` and ``additionalProperties:
    false`` everywhere — the schemas are built that way.
    """
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": schema,
        }
    }


def chat_completions_response_format(schema, name="review_response", strict=True):
    """``response_format`` for the OpenAI-compatible providers.

    Shared by Grok, Mistral, and Perplexity. Replaces ``{"type":
    "json_object"}``, which only promised *some* JSON, with one that pins the
    shape. Perplexity had no enforcement at all before this.
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": strict},
    }


def gemini_response_schema(schema):
    """``generationConfig`` entries for Gemini, or ``None`` when unusable.

    Gemini expresses types in upper case and rejects several JSON Schema
    keywords, so the schema is converted rather than passed through.

    Returns ``None`` for a schema that cannot be converted. Callers must also
    skip this entirely on grounded calls: Gemini rejects a response mime type
    alongside ``google_search`` with HTTP 400 ("Tool use with a response mime
    type: 'application/json' is unsupported"), so the grounded fact-check keeps
    asking for JSON in the prompt.
    """
    converted = _to_gemini(schema)
    if converted is None:
        return None
    return {"responseMimeType": "application/json", "responseSchema": converted}


_TYPE_NAMES = {
    "object": "OBJECT",
    "array": "ARRAY",
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
}


def _to_gemini(schema):
    """Convert a JSON Schema subset to Gemini's Schema shape.

    Gemini has no ``additionalProperties`` and no union types. A nullable field
    (``["string", "null"]``) becomes ``STRING`` with ``nullable: true``.
    """
    if not isinstance(schema, dict):
        return None
    stype = schema.get("type")
    nullable = False
    if isinstance(stype, list):
        non_null = [t for t in stype if t != "null"]
        nullable = "null" in stype
        if len(non_null) != 1:
            return None
        stype = non_null[0]
    if stype not in _TYPE_NAMES:
        return None

    out = {"type": _TYPE_NAMES[stype]}
    if nullable:
        out["nullable"] = True
    if "enum" in schema:
        out["enum"] = list(schema["enum"])
    if stype == "object":
        props = {}
        for key, value in (schema.get("properties") or {}).items():
            converted = _to_gemini(value)
            if converted is None:
                return None
            props[key] = converted
        out["properties"] = props
        if schema.get("required"):
            out["required"] = list(schema["required"])
    elif stype == "array":
        items = _to_gemini(schema.get("items") or {})
        if items is None:
            return None
        out["items"] = items
    return out
