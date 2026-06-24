"""Best-effort JSON extraction shared across review adapters.

Reasoning/grounded models (sonar-reasoning-pro, gemini-2.5-pro, etc.) sometimes
wrap their JSON in markdown fences, prepend a chain-of-thought / ``<think>``
preamble, or surround it with prose. A bare ``json.loads`` fails on all of those.
This helper tries, in order: direct parse; a fenced block; and finally the
outermost ``{`` … ``}`` span. Returns the parsed object, or None if nothing parses.
"""
import json
import re

_JSON_SPAN = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(content):
    if not content:
        return None
    s = content.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Strip a leading markdown fence (```json / ```) and trailing fence.
    if s.startswith("```"):
        inner = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass
    # Fall back to the outermost {...} span, skipping any prose/think preamble.
    match = _JSON_SPAN.search(content)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None
