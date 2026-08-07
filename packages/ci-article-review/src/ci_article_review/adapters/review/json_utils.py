"""Best-effort JSON extraction shared across review adapters.

Reasoning/grounded models (sonar-reasoning-pro, gemini-2.5-pro, etc.) sometimes
wrap their JSON in markdown fences, prepend a chain-of-thought / ``<think>``
preamble, or surround it with prose. A bare ``json.loads`` fails on all of those.

This helper tries, in order:
  1. Direct parse of the raw content.
  2. Direct parse after stripping any ``<think>...</think>`` block(s) — sonar-
     reasoning-pro can emit more than one, and reasoning prose frequently
     contains stray ``{``/``}`` characters (e.g. discussing the target schema)
     that would otherwise poison step 4's brace-span search.
  3. A fenced code block (```` ```json ... ``` ```` or ```` ``` ... ``` ````),
     found anywhere in the (think-stripped) text — not only when the fence is
     the very first thing in the string, so leading prose before the fence
     doesn't defeat detection.
  4. The outermost ``{`` ... ``}`` span in the think-stripped text.

Returns the parsed object, or None if nothing parses.
"""

import json
import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCED_BLOCK = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_SPAN = re.compile(r"\{.*\}", re.DOTALL)


def _try_parse(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def extract_json(content):
    if not content:
        return None

    # 1. Direct parse.
    parsed = _try_parse(content.strip())
    if parsed is not None:
        return parsed

    # 2. Strip <think>...</think> block(s) — may be more than one, and may
    # contain braces that would otherwise corrupt the brace-span fallback.
    stripped = _THINK_BLOCK.sub("", content).strip()
    parsed = _try_parse(stripped)
    if parsed is not None:
        return parsed

    # 3. A fenced block, found anywhere (handles prose before AND after it).
    fence_match = _FENCED_BLOCK.search(stripped)
    if fence_match:
        parsed = _try_parse(fence_match.group(1).strip())
        if parsed is not None:
            return parsed

    # 4. Outermost {...} span in the think-stripped text.
    span_match = _JSON_SPAN.search(stripped)
    if span_match:
        parsed = _try_parse(span_match.group(0))
        if parsed is not None:
            return parsed

    return None
