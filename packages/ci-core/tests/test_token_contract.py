"""The ``tokens`` contract, enforced across every adapter and every return path.

Every adapter result — success *and* failure — must carry ``tokens`` as
``{"prompt": int, "completion": int}``. Provider-native spellings
(``prompt_tokens``, ``input_tokens``, ``promptTokenCount``, ...) must never
escape an adapter.

This is not cosmetic. :func:`ci_core.llm.cost._entry_cost` reads exactly
``tokens["prompt"]`` and ``tokens["completion"]``, so a leaked provider dict
does not raise — it silently prices the call at $0. That is how a Gemini
fact_check call showed up in a live run summary as ``0+0 tok``.

The static test below scans the adapter sources so a *new* return path added
later is covered without anyone remembering to write a test for it.
"""

import ast
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ci_core.llm import cost
from ci_core.llm.adapters import ADAPTER_MODULES, get_adapter
from ci_core.llm.tokens import normalize_tokens

ADAPTER_NAMES = sorted(ADAPTER_MODULES)
ADAPTER_DIR = Path(get_adapter("openai").__file__).parent


# ---------------------------------------------------------------------------
# The normalizer itself
# ---------------------------------------------------------------------------


class TestNormalizeTokens:
    @pytest.mark.parametrize(
        "usage,expected",
        [
            # OpenAI chat completions / grok / mistral / perplexity
            (
                {"prompt_tokens": 50, "completion_tokens": 20},
                {"prompt": 50, "completion": 20},
            ),
            # OpenAI Responses API / Claude
            (
                {"input_tokens": 50, "output_tokens": 20},
                {"prompt": 50, "completion": 20},
            ),
            # Gemini
            (
                {"promptTokenCount": 50, "candidatesTokenCount": 20},
                {"prompt": 50, "completion": 20},
            ),
            # Already normalized — idempotent
            ({"prompt": 50, "completion": 20}, {"prompt": 50, "completion": 20}),
            # Degenerate inputs never raise
            ({}, {"prompt": 0, "completion": 0}),
            (None, {"prompt": 0, "completion": 0}),
            ("nonsense", {"prompt": 0, "completion": 0}),
            ({"totalTokenCount": 99}, {"prompt": 0, "completion": 0}),
            # Non-numeric values are ignored rather than propagated
            (
                {"prompt_tokens": None, "completion_tokens": "x"},
                {"prompt": 0, "completion": 0},
            ),
        ],
    )
    def test_normalizes_every_provider_spelling(self, usage, expected):
        assert normalize_tokens(usage) == expected

    def test_gemini_thinking_tokens_count_as_output(self):
        # Live Gemini usageMetadata: totalTokenCount == prompt + candidates +
        # thoughts, so thinking tokens are disjoint from candidatesTokenCount.
        # Google bills them at the output rate, so they belong in `completion`.
        usage = {
            "promptTokenCount": 32945,
            "candidatesTokenCount": 1917,
            "thoughtsTokenCount": 3993,
            "totalTokenCount": 38855,
        }
        assert usage["totalTokenCount"] == (
            usage["promptTokenCount"]
            + usage["candidatesTokenCount"]
            + usage["thoughtsTokenCount"]
        )
        assert normalize_tokens(usage) == {"prompt": 32945, "completion": 5910}

    def test_normalizing_twice_does_not_double_count_thinking(self):
        usage = {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "thoughtsTokenCount": 7,
        }
        once = normalize_tokens(usage)
        assert normalize_tokens(once) == once


# ---------------------------------------------------------------------------
# Static guard: no adapter return path may leak a provider-native dict
# ---------------------------------------------------------------------------


_NATIVE_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "promptTokenCount",
    "candidatesTokenCount",
    "thoughtsTokenCount",
    "totalTokenCount",
}


def _token_values_in_returns(path):
    """Yield (lineno, ast node) for every ``"tokens": <value>`` in a dict literal."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "tokens":
                yield node.lineno, value


@pytest.mark.parametrize("name", ADAPTER_NAMES)
def test_no_adapter_return_path_leaks_provider_native_tokens(name):
    """Every ``"tokens":`` value is normalize_tokens(...), {}, or a normalized var.

    Catches the regression class directly: a dict literal spelling the
    provider's own key names, or a bare pass-through of the raw usage dict.
    """
    path = ADAPTER_DIR / f"{name}.py"
    offenders = []
    for lineno, value in _token_values_in_returns(path):
        if isinstance(value, ast.Call):
            func = value.func
            fname = getattr(func, "id", None) or getattr(func, "attr", None)
            if fname == "normalize_tokens":
                continue
            offenders.append(f"{name}.py:{lineno} calls {fname}()")
        elif isinstance(value, ast.Dict):
            keys = {k.value for k in value.keys if isinstance(k, ast.Constant)}
            if not keys:  # `{}` — no usage was captured
                continue
            if keys <= {"prompt", "completion"}:
                continue
            offenders.append(
                f"{name}.py:{lineno} builds tokens with provider-native keys "
                f"{sorted(keys & _NATIVE_KEYS) or sorted(keys)}"
            )
        elif isinstance(value, ast.Name):
            # A variable is only acceptable if it was assigned from
            # normalize_tokens(...) somewhere in the module.
            src = path.read_text(encoding="utf-8")
            if f"{value.id} = normalize_tokens(" not in src:
                offenders.append(
                    f"{name}.py:{lineno} returns bare variable {value.id!r} "
                    f"— not provably normalized"
                )
        else:
            offenders.append(f"{name}.py:{lineno} returns unrecognized token value")

    assert not offenders, (
        "These adapter return paths do not honor the {prompt, completion} token "
        "contract:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Live guard: drive each adapter and assert the shape of what comes back
# ---------------------------------------------------------------------------


def _sse(lines):
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"Content-Type": "text/event-stream"}
    resp.iter_lines.return_value = [ln.encode("utf-8") for ln in lines]
    resp.raise_for_status.return_value = None
    return resp


# Per-adapter SSE fixtures: (stream lines, expected normalized tokens).
# The payload is deliberately NOT valid JSON so every adapter lands on its
# malformed-JSON failure path — the path that used to leak.
_PROSE = "prose with no JSON payload at all"

_FIXTURES = {
    "openai": (
        [
            "data: "
            + json.dumps(
                {
                    "type": "response.output_text.delta",
                    "delta": _PROSE,
                }
            ),
            "data: "
            + json.dumps(
                {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": 50, "output_tokens": 20}},
                }
            ),
            "data: [DONE]",
        ],
        {"prompt": 50, "completion": 20},
    ),
    "gemini": (
        [
            "data: "
            + json.dumps(
                {
                    "candidates": [{"content": {"parts": [{"text": _PROSE}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 50,
                        "candidatesTokenCount": 20,
                    },
                }
            )
        ],
        {"prompt": 50, "completion": 20},
    ),
}

_CHAT_COMPLETION_ADAPTERS = ("mistral", "grok", "perplexity")


def _chat_lines(usage):
    return [
        "data: " + json.dumps({"choices": [{"delta": {"content": _PROSE}}]}),
        "data: " + json.dumps({"choices": [{"delta": {}}], "usage": usage}),
        "data: [DONE]",
    ]


for _name in _CHAT_COMPLETION_ADAPTERS:
    _FIXTURES[_name] = (
        _chat_lines({"prompt_tokens": 50, "completion_tokens": 20}),
        {"prompt": 50, "completion": 20},
    )

_FIXTURES["claude"] = (
    [
        "data: "
        + json.dumps(
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 50, "output_tokens": 0}},
            }
        ),
        "data: "
        + json.dumps(
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": _PROSE},
            }
        ),
        "data: "
        + json.dumps({"type": "message_delta", "usage": {"output_tokens": 20}}),
        "data: " + json.dumps({"type": "message_stop"}),
    ],
    {"prompt": 50, "completion": 20},
)


@pytest.mark.parametrize("name", ADAPTER_NAMES)
def test_failure_path_returns_normalized_tokens(name):
    """A malformed-JSON failure still reports {prompt, completion}, not raw usage."""
    lines, expected = _FIXTURES[name]
    adapter = get_adapter(name)
    with patch(f"ci_core.llm.adapters.{name}.requests.Session") as session_cls:
        session = MagicMock()
        session_cls.return_value = session
        session.post.return_value = _sse(lines)
        result = adapter.call("system", "user", "key", retry=False)

    assert result["failed"] is True, f"{name}: expected the malformed-JSON path"
    tokens = result["tokens"]
    assert set(tokens) == {"prompt", "completion"}, (
        f"{name} leaked provider-native token keys on its failure path: {tokens}"
    )
    assert tokens == expected


@pytest.mark.parametrize("name", ADAPTER_NAMES)
def test_failure_path_tokens_are_priceable(name):
    """cost.py can price what the adapter returns — no silent $0 from key drift."""
    lines, expected = _FIXTURES[name]
    adapter = get_adapter(name)
    with patch(f"ci_core.llm.adapters.{name}.requests.Session") as session_cls:
        session = MagicMock()
        session_cls.return_value = session
        session.post.return_value = _sse(lines)
        result = adapter.call("system", "user", "key", retry=False)

    entry = {
        "pass": "contract",
        "model": result["model"],
        "tokens": result["tokens"],
        # Deliberately not marked failed: _entry_cost short-circuits failed
        # entries to $0, which would mask a token-shape problem.
        "failed": False,
    }
    summary = cost.calculate([entry])
    assert summary["total_usd"] > 0, (
        f"{name}: cost priced a call with {expected} tokens at $0 — the token "
        f"dict did not reach cost.py in the expected shape"
    )
