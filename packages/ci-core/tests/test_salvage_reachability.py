"""Truncation salvage (PR #40) must be reachable from every adapter, not just Mistral.

A response that hit the output-token ceiling is well-formed JSON up to the cut
point. ``extract_json`` correctly refuses it (incomplete, not malformed), so
each adapter has to opt into ``extract_json_with_salvage`` to recover the
complete elements. Originally only ``mistral`` did; the other five discarded a
usable response and reported "Malformed JSON response".

These tests drive each adapter with a genuinely truncated payload and assert the
findings survive, flagged ``truncated: True`` rather than lost.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from ci_core.llm.adapters import ADAPTER_MODULES, get_adapter

ADAPTER_NAMES = sorted(ADAPTER_MODULES)

# Well-formed up to the cut: two complete findings, then the stream stops
# mid-way through a third. Salvage should recover exactly the first two.
TRUNCATED_PAYLOAD = (
    '```json\n{"confirmed": ['
    '{"claim": "first claim", "verdict": "supported"}, '
    '{"claim": "second claim", "verdict": "supported"}, '
    '{"claim": "third claim inte'
)

EXPECTED_RECOVERED = [
    {"claim": "first claim", "verdict": "supported"},
    {"claim": "second claim", "verdict": "supported"},
]


def _sse(lines):
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"Content-Type": "text/event-stream"}
    resp.iter_lines.return_value = [ln.encode("utf-8") for ln in lines]
    resp.raise_for_status.return_value = None
    return resp


def _chat_lines(text):
    return [
        "data: " + json.dumps({"choices": [{"delta": {"content": text}}]}),
        "data: "
        + json.dumps(
            {
                "choices": [{"delta": {}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
        ),
        "data: [DONE]",
    ]


def _lines_for(name, text):
    if name == "gemini":
        return [
            "data: "
            + json.dumps(
                {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": text}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 100,
                        "candidatesTokenCount": 50,
                        "thoughtsTokenCount": 200,
                    },
                }
            )
        ]
    if name == "openai":
        return [
            "data: "
            + json.dumps({"type": "response.output_text.delta", "delta": text}),
            "data: "
            + json.dumps(
                {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": 100, "output_tokens": 50}},
                }
            ),
            "data: [DONE]",
        ]
    if name == "claude":
        return [
            "data: "
            + json.dumps(
                {
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 100, "output_tokens": 0}},
                }
            ),
            "data: "
            + json.dumps(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": text},
                }
            ),
            "data: "
            + json.dumps({"type": "message_delta", "usage": {"output_tokens": 50}}),
            "data: " + json.dumps({"type": "message_stop"}),
        ]
    return _chat_lines(text)


def _call(name, text):
    adapter = get_adapter(name)
    with patch(f"ci_core.llm.adapters.{name}.requests.Session") as session_cls:
        session = MagicMock()
        session_cls.return_value = session
        session.post.return_value = _sse(_lines_for(name, text))
        return adapter.call("system", "user", "key", retry=False)


@pytest.mark.parametrize("name", ADAPTER_NAMES)
def test_truncated_response_is_salvaged_not_discarded(name):
    result = _call(name, TRUNCATED_PAYLOAD)

    assert result["failed"] is False, (
        f"{name} discarded a salvageable truncated response "
        f"({result.get('error')!r}) — PR #40's salvage path is not reachable here"
    )
    assert result["data"]["confirmed"] == EXPECTED_RECOVERED, (
        f"{name} recovered {result['data']} — expected exactly the complete elements"
    )
    assert result.get("truncated") is True, (
        f"{name} salvaged the response but did not flag it as truncated, so "
        f"downstream reporting cannot distinguish it from a clean answer"
    )


@pytest.mark.parametrize("name", ADAPTER_NAMES)
def test_clean_response_is_not_flagged_truncated(name):
    """Salvage must not mark an ordinary complete response as truncated."""
    result = _call(name, '{"confirmed": [{"claim": "only claim"}]}')

    assert result["failed"] is False
    assert result.get("truncated") is not True


@pytest.mark.parametrize("name", ADAPTER_NAMES)
def test_unsalvageable_response_still_fails(name):
    """Salvage must not turn genuine prose into a fabricated success."""
    result = _call(name, "I could not complete this request.")

    assert result["failed"] is True
    assert "Malformed JSON" in result["error"]
