"""Unit tests for ci_core.llm.json_utils.

These outlive the litellm migration unchanged. litellm normalises transports,
not model output: a reasoning model still wraps its JSON in a fence, still
prepends a <think> preamble, and still gets cut off mid-array when it hits the
output-token ceiling. Nothing in litellm recovers any of that.

The extraction cases carry their original name because they were written
against sonar-reasoning-pro's real output; the behaviour is provider-agnostic.
"""


class TestJsonUtilsSalvage:
    def test_clean_response_not_marked_truncated(self):
        from ci_core.llm.json_utils import (
            extract_json_with_salvage,
        )

        data, truncated = extract_json_with_salvage('{"flags": [{"passage": "a"}]}')
        assert data == {"flags": [{"passage": "a"}]}
        assert truncated is False

    def test_truncated_mid_array_between_elements(self):
        from ci_core.llm.json_utils import (
            extract_json_with_salvage,
        )

        # Cut off cleanly after the second element's comma, mid-way through the
        # third element's key — never even opened the third object's value.
        raw = (
            '{"flags": ['
            '{"passage": "a", "problem": "p1"}, '
            '{"passage": "b", "problem": "p2"}, '
            '{"passage": "c", "prob'
        )
        data, truncated = extract_json_with_salvage(raw)
        assert truncated is True
        assert data == {
            "flags": [
                {"passage": "a", "problem": "p1"},
                {"passage": "b", "problem": "p2"},
            ]
        }

    def test_truncated_mid_string_inside_value(self):
        # Truncation lands inside an unterminated string value, not between
        # elements — the escape/string-depth tracking must not let that produce
        # invalid JSON, and the incomplete element must be dropped entirely.
        from ci_core.llm.json_utils import (
            extract_json_with_salvage,
        )

        raw = (
            '{"flags": ['
            '{"passage": "a", "problem": "p1"}, '
            '{"passage": "The Cambridge preprint\'s central claim may be real'
        )
        data, truncated = extract_json_with_salvage(raw)
        assert truncated is True
        assert data == {"flags": [{"passage": "a", "problem": "p1"}]}

    def test_truncated_with_trailing_backslash_before_cut(self):
        # A backslash right at the cut point must not desynchronize the
        # escape-state tracking for whatever came before it.
        from ci_core.llm.json_utils import (
            extract_json_with_salvage,
        )

        raw = '{"flags": [{"passage": "a", "problem": "p1"}, {"passage": "b\\'
        data, truncated = extract_json_with_salvage(raw)
        assert truncated is True
        assert data == {"flags": [{"passage": "a", "problem": "p1"}]}

    def test_nothing_recoverable_returns_none(self):
        from ci_core.llm.json_utils import (
            extract_json_with_salvage,
        )

        data, truncated = extract_json_with_salvage('{"flags": [{"passage"')
        assert data is None
        assert truncated is False

    def test_empty_content_returns_none(self):
        from ci_core.llm.json_utils import (
            extract_json_with_salvage,
        )

        assert extract_json_with_salvage("") == (None, False)
        assert extract_json_with_salvage(None) == (None, False)


class TestPerplexityExtractJson:
    def test_plain_json(self):
        from ci_core.llm.json_utils import extract_json as _extract_json

        assert _extract_json('{"flags": []}') == {"flags": []}

    def test_code_fence(self):
        from ci_core.llm.json_utils import extract_json as _extract_json

        assert _extract_json('```json\n{"flags": [1]}\n```') == {"flags": [1]}

    def test_think_preamble(self):
        # The observed failure mode: a reasoning block before the JSON.
        from ci_core.llm.json_utils import extract_json as _extract_json

        raw = '<think>\nLet me reason about this claim...\nstep two\n</think>\n{"verdict": "confirmed"}'
        assert _extract_json(raw) == {"verdict": "confirmed"}

    def test_prose_before_and_after(self):
        from ci_core.llm.json_utils import extract_json as _extract_json

        raw = 'Here is my analysis:\n{"a": 1, "b": [2, 3]}\nHope that helps!'
        assert _extract_json(raw) == {"a": 1, "b": [2, 3]}

    def test_unrecoverable_returns_none(self):
        from ci_core.llm.json_utils import extract_json as _extract_json

        assert _extract_json("no json here at all") is None
        assert _extract_json("") is None

    def test_multiple_think_blocks(self):
        # sonar-reasoning-pro can emit more than one <think> block.
        from ci_core.llm.json_utils import extract_json as _extract_json

        raw = (
            "<think>first pass</think>\n"
            "<think>second pass, reconsidering</think>\n"
            '{"verdict": "confirmed"}'
        )
        assert _extract_json(raw) == {"verdict": "confirmed"}

    def test_think_block_containing_braces_does_not_corrupt_span_match(self):
        # A reasoning block that itself mentions braces (e.g. discussing the
        # target JSON schema) must not widen the outermost-{...} span past the
        # real payload.
        from ci_core.llm.json_utils import extract_json as _extract_json

        raw = (
            '<think>The schema should look like {"verdict": ...} '
            "so I need to produce that.</think>\n"
            '{"verdict": "confirmed"}'
        )
        assert _extract_json(raw) == {"verdict": "confirmed"}

    def test_worst_case_think_fence_and_leading_prose(self):
        # The documented worst case: leading prose, a <think> block, AND a
        # markdown fence, all in the same response.
        from ci_core.llm.json_utils import extract_json as _extract_json

        raw = (
            "Sure, here is my analysis.\n"
            "<think>\nLet me think about the claim: {this is not json}\n</think>\n"
            "```json\n"
            '{"verdict": "confirmed", "citations": [1, 2]}\n'
            "```\n"
            "Let me know if you need anything else!"
        )
        assert _extract_json(raw) == {
            "verdict": "confirmed",
            "citations": [1, 2],
        }
