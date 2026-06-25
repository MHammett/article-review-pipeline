"""Tests for voice_consolidation.py."""

from __future__ import annotations

from unittest.mock import patch


def _make_result(model_name: str, items: list[str], key: str = "banned_words") -> dict:
    return {
        "content": "{}",
        "failed": False,
        "tokens": {},
        "elapsed": 0.1,
        "_parsed": {key: items},
    }


class TestConsolidateLists:
    def test_claude_openai_pass_threshold(self):
        """Claude (1.1) + OpenAI (1.2) = 2.3 ≥ 2.0 → item included."""
        from ci_web_intel.voice_consolidation import consolidate_lists

        results = {
            "claude": _make_result("claude", ["leverage"]),
            "openai": _make_result("openai", ["leverage"]),
        }
        output = consolidate_lists(results, "banned_words")
        assert "leverage" in output

    def test_mistral_only_below_threshold(self):
        """Mistral only (weight 1.0) < 2.0 → item excluded."""
        from ci_web_intel.voice_consolidation import consolidate_lists

        results = {
            "mistral": _make_result("mistral", ["synergy"]),
        }
        output = consolidate_lists(results, "banned_words")
        assert "synergy" not in output

    def test_four_models_agree(self):
        """All 4 models agree → item included regardless."""
        from ci_web_intel.voice_consolidation import consolidate_lists

        results = {
            "claude": _make_result("claude", ["utilize"]),
            "openai": _make_result("openai", ["utilize"]),
            "mistral": _make_result("mistral", ["utilize"]),
            "gemini": _make_result("gemini", ["utilize"]),
        }
        output = consolidate_lists(results, "banned_words")
        assert "utilize" in output

    def test_failed_model_not_counted(self):
        """Failed model results don't contribute to vote counts."""
        from ci_web_intel.voice_consolidation import consolidate_lists

        results = {
            "claude": {"failed": True, "error": "timeout", "_parsed": {}},
            "openai": _make_result("openai", ["basically"]),
        }
        output = consolidate_lists(results, "banned_words")
        assert "basically" not in output  # OpenAI alone (1.2) < 2.0

    def test_custom_threshold(self):
        """Custom threshold works."""
        from ci_web_intel.voice_consolidation import consolidate_lists

        results = {
            "openai": _make_result("openai", ["arguably"]),
        }
        output = consolidate_lists(results, "banned_words", threshold=1.0)
        assert "arguably" in output


class TestCollectProse:
    def test_sorted_by_weight_descending(self):
        """collect_prose returns entries sorted by weight descending (Claude and OpenAI first)."""
        from ci_web_intel.voice_consolidation import collect_prose

        results = {
            "mistral": {
                "failed": False,
                "_parsed": {"voice_profile": "mistral prose"},
                "tokens": {},
            },
            "claude": {
                "failed": False,
                "_parsed": {"voice_profile": "claude prose"},
                "tokens": {},
            },
            "openai": {
                "failed": False,
                "_parsed": {"voice_profile": "openai prose"},
                "tokens": {},
            },
        }
        entries = collect_prose(results, "voice_profile")
        assert entries[0]["model"] in ("openai", "claude")  # highest weight first
        assert entries[-1]["model"] == "mistral"

    def test_failed_excluded(self):
        """Failed model results are excluded from prose collection."""
        from ci_web_intel.voice_consolidation import collect_prose

        results = {
            "claude": {"failed": True, "_parsed": {}, "tokens": {}},
            "openai": {
                "failed": False,
                "_parsed": {"voice_profile": "openai result"},
                "tokens": {},
            },
        }
        entries = collect_prose(results, "voice_profile")
        assert len(entries) == 1
        assert entries[0]["model"] == "openai"

    def test_empty_key_excluded(self):
        """Models with missing key are excluded."""
        from ci_web_intel.voice_consolidation import collect_prose

        results = {
            "claude": {"failed": False, "_parsed": {}, "tokens": {}},
        }
        entries = collect_prose(results, "voice_profile")
        assert len(entries) == 0


class TestConsolidateDetection:
    def test_consolidate_calls_claude(self):
        """consolidate_detection makes one Claude API call with consolidate_detection.txt prompt."""
        from ci_web_intel.voice_consolidation import consolidate_detection

        detection_results = {
            "claude": {
                "content": '{"detected_voices": [{"label": "technical", "description": "Long form", "features": {}, "source_distribution": {}, "sample_ids": [], "confidence": "high"}], "overall_confidence": "high", "consolidation_notes": ""}',
                "failed": False,
            },
            "openai": {
                "content": '{"detected_voices": [{"label": "technical analysis", "description": "Analytical", "features": {}, "source_distribution": {}, "sample_ids": [], "confidence": "high"}], "overall_confidence": "high", "consolidation_notes": ""}',
                "failed": False,
            },
        }

        mock_response = {
            "content": '{"detected_voices": [{"label": "technical", "description": "Unified", "features": {}, "source_distribution": {}, "sample_ids": [], "confidence": "high"}], "overall_confidence": "high", "consolidation_notes": "agreed"}',
            "failed": False,
            "tokens": {"prompt": 100, "completion": 50},
            "elapsed": 1.0,
        }
        user_config = {
            "models": {
                "claude": {"provider": "anthropic", "model": "claude-sonnet-4-6"}
            },
            "api_keys": {"claude": {"api_key": "test-key"}},
        }

        with patch(
            "ci_web_intel.voice_consolidation.call_one", return_value=mock_response
        ):
            result = consolidate_detection(detection_results, user_config)

        assert result is not None
        raw_voices, overall_confidence = result
        assert len(raw_voices) >= 1
        assert raw_voices[0]["label"] == "technical"
