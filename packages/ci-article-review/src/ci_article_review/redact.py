"""Shared secret-redaction helpers.

Some provider APIs (notably Gemini AI Studio) carry the API key as a URL query
parameter.  When a network-level exception fires, the requests library embeds
the full URL — key included — in the exception string.  Printing that string
to the terminal or a log file leaks the credential.  These helpers scrub keys
out before any error text is surfaced.
"""

import re

# Matches ?key=..., &apiKey=..., &api_key=... in a URL and replaces the value.
# Stops at the next &, whitespace, quote, or closing bracket so we don't eat
# the rest of the message.
_KEY_QUERY_RE = re.compile(
    r"([?&](?:key|apikey|api_key|access_token|token)=)[^&\s'\"<>)\]]+",
    re.IGNORECASE,
)


def redact_url_keys(text):
    """Replace key/token query-parameter values in any string with [REDACTED].

    Provider-agnostic: works even when the key value isn't available to compare
    against, because it matches on the parameter name.
    """
    return _KEY_QUERY_RE.sub(r"\1[REDACTED]", str(text))


def redact_value(text, secret):
    """Replace a known secret value with [REDACTED] wherever it appears."""
    if secret and secret in str(text):
        return str(text).replace(secret, "[REDACTED]")
    return str(text)


def truncate_excerpt(text, head=2000, tail=500):
    """Truncate long text to a head/tail excerpt with an omitted-chars marker.

    Used to persist a diagnosable slice of a raw provider response (e.g. a
    malformed-JSON failure) without bloating the report with the full payload.
    Text at or under ``head + tail`` chars is returned unchanged.
    """
    text = str(text)
    if len(text) <= head + tail:
        return text
    omitted = len(text) - head - tail
    return f"{text[:head]}\n...[{omitted} chars omitted]...\n{text[-tail:]}"
