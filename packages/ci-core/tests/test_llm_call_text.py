"""Tests for the ci_core.llm.adapters dispatch layer and call_text.

``call_provider`` is a thin name-based dispatch over the six adapter modules.
``call_text`` sits on top of it for callers that want the assembled response
text rather than the adapters' JSON-parsing verdict — ci-style-profile parses
JSON itself downstream and feeds some model output into a later prompt
verbatim, so for it a prose reply is usable rather than fatal.
"""

import pytest

from ci_core.llm import adapters


class TestGetAdapter:
    def test_every_declared_adapter_imports(self):
        for name in adapters.ADAPTER_MODULES:
            mod = adapters.get_adapter(name)
            assert callable(mod.call), f"{name} has no call()"

    def test_unknown_adapter_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown adapter"):
            adapters.get_adapter("not-a-provider")

    def test_all_six_providers_declared(self):
        assert set(adapters.ADAPTER_MODULES) == {
            "openai",
            "gemini",
            "mistral",
            "grok",
            "claude",
            "perplexity",
        }


class TestCallProvider:
    def test_passes_arguments_through_to_adapter(self, monkeypatch):
        seen = {}

        def fake_call(system_prompt, user_prompt, api_key, **kwargs):
            seen.update(
                system=system_prompt, user=user_prompt, key=api_key, kwargs=kwargs
            )
            return {"failed": False, "data": {}, "raw": "{}"}

        monkeypatch.setattr(adapters.get_adapter("grok"), "call", fake_call)
        adapters.call_provider(
            "grok", "sys", "usr", "xai-key", model="grok-4.3", retry=False
        )

        assert seen["system"] == "sys"
        assert seen["user"] == "usr"
        assert seen["key"] == "xai-key"
        assert seen["kwargs"]["model"] == "grok-4.3"
        assert seen["kwargs"]["retry"] is False

    def test_returns_adapter_result_unchanged(self, monkeypatch):
        payload = {
            "failed": False,
            "data": {"a": 1},
            "raw": '{"a": 1}',
            "extra": "kept",
        }
        monkeypatch.setattr(
            adapters.get_adapter("claude"), "call", lambda *a, **kw: dict(payload)
        )
        assert adapters.call_provider("claude", "s", "u", "k") == payload


def _stub(monkeypatch, name, result):
    monkeypatch.setattr(adapters.get_adapter(name), "call", lambda *a, **kw: result)


class TestCallText:
    def test_success_returns_raw_text_not_parsed_data(self, monkeypatch):
        _stub(
            monkeypatch,
            "openai",
            {
                "failed": False,
                "data": {"style_profile": "terse"},
                "raw": '{"style_profile": "terse"}',
                "model": "gpt-5.4",
                "tokens": {"prompt": 10, "completion": 3},
                "elapsed_seconds": 1.5,
            },
        )
        out = adapters.call_text("openai", "s", "u", "k")

        assert out["failed"] is False
        assert out["content"] == '{"style_profile": "terse"}'
        assert out["model"] == "gpt-5.4"
        assert out["tokens"] == {"prompt": 10, "completion": 3}
        assert out["elapsed"] == 1.5

    def test_malformed_json_with_text_is_passed_through_as_success(self, monkeypatch):
        # The adapters require JSON because the review pipeline needs structured
        # findings. Text-mode callers parse it themselves, so prose is usable.
        _stub(
            monkeypatch,
            "mistral",
            {
                "failed": True,
                "error": "Malformed JSON response",
                "raw": "Here is my analysis, in prose.",
                "model": "mistral-medium-3-5",
                "tokens": {"prompt": 5, "completion": 7},
                "elapsed_seconds": 0.5,
            },
        )
        out = adapters.call_text("mistral", "s", "u", "k")

        assert out["failed"] is False
        assert out["content"] == "Here is my analysis, in prose."
        assert "error" not in out

    def test_malformed_json_with_no_text_stays_failed(self, monkeypatch):
        # An empty body is a real failure — there is nothing to hand downstream.
        _stub(
            monkeypatch,
            "gemini",
            {
                "failed": True,
                "error": "Malformed JSON response",
                "raw": "",
                "model": "gemini-2.5-flash",
                "tokens": {},
                "elapsed_seconds": 0.2,
            },
        )
        out = adapters.call_text("gemini", "s", "u", "k")

        assert out["failed"] is True
        assert out["content"] == ""

    def test_transport_failure_stays_failed_and_keeps_error_body(self, monkeypatch):
        _stub(
            monkeypatch,
            "perplexity",
            {
                "failed": True,
                "error": "401 Client Error: Unauthorized",
                "error_body": '{"error": "invalid api key"}',
                "raw": None,
                "model": "sonar-pro",
                "tokens": {},
                "elapsed_seconds": 0.1,
            },
        )
        out = adapters.call_text("perplexity", "s", "u", "k")

        assert out["failed"] is True
        assert out["error"] == "401 Client Error: Unauthorized"
        assert out["error_body"] == '{"error": "invalid api key"}'
        assert out["content"] == ""

    def test_empty_response_failure_stays_failed(self, monkeypatch):
        _stub(
            monkeypatch,
            "claude",
            {
                "failed": True,
                "error": "Empty text response (stop_reason='max_tokens')",
                "raw": None,
                "model": "claude-opus-4-8",
                "tokens": {},
                "elapsed_seconds": 3.0,
            },
        )
        out = adapters.call_text("claude", "s", "u", "k")
        assert out["failed"] is True

    def test_model_falls_back_to_adapter_name_when_absent(self, monkeypatch):
        _stub(monkeypatch, "grok", {"failed": False, "raw": "{}"})
        assert adapters.call_text("grok", "s", "u", "k")["model"] == "grok"


class TestAdaptersReturnRawOnSuccess:
    """Every adapter must include the assembled text as ``raw`` on success.

    ``call_text`` is built on that key; an adapter that stopped emitting it
    would silently start returning empty content instead of failing loudly.
    """

    @pytest.mark.parametrize("name", sorted(adapters.ADAPTER_MODULES))
    def test_success_return_includes_raw(self, name):
        import inspect

        src = inspect.getsource(adapters.get_adapter(name))
        successes = src.count('"failed": False,')
        raws = src.count('"raw": ')
        assert successes > 0, f"{name}: no success return found"
        # Each success return is immediately followed by a "raw" key; failure
        # paths may add more, so raws >= successes.
        assert raws >= successes, (
            f"{name}: {successes} success return(s) but only {raws} 'raw' key(s) — "
            f"a success path is missing the assembled text"
        )
