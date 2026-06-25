"""Tests for synthesize.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _make_doc(text: str, source: str = "wordpress", date: str = "2024-01-15"):
    from collectors.base import Document
    doc = Document.from_text(text=text, source=source, register="long_form", date=date, url_or_id="http://ex.com/1")
    doc.metrics = {"avg_sentence_words": 15.0, "hedging_ratio": 0.05, "first_person_ratio": 0.2}
    return doc


_CANONICAL_SYNTHESIS = """{
  "voice_profile": "This author writes with clarity and precision.",
  "audience_primary": "Business professionals",
  "audience_secondary": null,
  "banned_words": ["utilize", "leverage"],
  "banned_phrases": ["at the end of the day"],
  "positive_rules": ["Lead with the main claim", "Use concrete examples"]
}"""

_RECONCILE_CANONICAL = """{
  "canonical": {
    "voice_profile": "Reconciled voice profile text.",
    "audience_primary": "Business professionals",
    "audience_secondary": null,
    "banned_words": ["utilize", "leverage"],
    "banned_phrases": ["at the end of the day"],
    "positive_rules": ["Lead with the main claim"],
    "confidence": "high"
  },
  "detected_voices": {},
  "synthesis_notes": "Models agreed on key points."
}"""

_DETECT_RESPONSE = """{
  "detected_voices": [
    {
      "label": "technical analysis",
      "description": "Long analytical",
      "features": {"avg_sentence_words": [">", 18]},
      "source_distribution": {"wordpress": 1.0},
      "sample_ids": [],
      "confidence": "high"
    }
  ],
  "overall_confidence": "high",
  "detection_notes": ""
}"""

_PER_VOICE_SYNTHESIS = """{
  "voice_profile": "Technical voice.",
  "additional_banned_words": [],
  "additional_positive_rules": ["Use numbered lists"],
  "voice_notes": "Used for in-depth analysis",
  "source_distribution": {"wordpress": 1.0},
  "doc_count": 5,
  "confidence": "high"
}"""

_RECONCILE_DETECT = """{
  "canonical": {
    "voice_profile": "Overall reconciled profile.",
    "audience_primary": "Professionals",
    "audience_secondary": null,
    "banned_words": ["utilize"],
    "banned_phrases": [],
    "positive_rules": ["Be direct"],
    "confidence": "high"
  },
  "detected_voices": {
    "technical analysis": {
      "voice_profile": "Technical voice reconciled.",
      "additional_banned_words": [],
      "additional_positive_rules": ["Use numbered lists"],
      "voice_notes": "For deep analysis",
      "source_distribution": {"wordpress": 1.0},
      "doc_count": 5,
      "confidence": "high"
    }
  },
  "synthesis_notes": "Strong signal."
}"""


_USER_CONFIG = {
    "models": {
        "claude": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "openai": {"provider": "openai", "model": "gpt-5.4"},
    },
    "api_keys": {
        "claude": {"api_key": "test"},
        "openai": {"api_key": "test"},
    },
}


class TestCanonicalMode:
    def test_canonical_makes_correct_calls(self):
        """canonical mode: M synthesis + 1 reconciliation = M+1 total calls."""
        from synthesize import synthesize

        docs = [_make_doc("Long article text " * 100) for _ in range(5)]
        call_count = []

        def _fake_call_all(system_prompt, user_prompt, user_config, models=None, max_parallel=0, pass_name="", **kw):
            call_count.append(("call_all", models, pass_name))
            return {
                "claude": {"content": _CANONICAL_SYNTHESIS, "failed": False, "tokens": {}, "elapsed": 1.0, "_parsed": {"voice_profile": "test", "audience_primary": "test", "banned_words": ["utilize"], "banned_phrases": [], "positive_rules": ["rule1"]}},
                "openai": {"content": _CANONICAL_SYNTHESIS, "failed": False, "tokens": {}, "elapsed": 1.0, "_parsed": {"voice_profile": "test", "audience_primary": "test", "banned_words": ["utilize"], "banned_phrases": [], "positive_rules": ["rule1"]}},
            }

        def _fake_call_one(model_name, model_cfg, api_keys, system_prompt, user_prompt, pass_name=""):
            call_count.append(("call_one", model_name, pass_name))
            return {"content": _RECONCILE_CANONICAL, "failed": False, "tokens": {}, "elapsed": 1.0}

        with patch("synthesize.call_all", side_effect=_fake_call_all), \
             patch("synthesize.call_one", side_effect=_fake_call_one):
            result = synthesize(docs, _USER_CONFIG, mode="canonical")

        call_all_calls = [c for c in call_count if c[0] == "call_all"]
        call_one_calls = [c for c in call_count if c[0] == "call_one"]

        assert len(call_all_calls) == 1  # synthesis pass
        assert len(call_one_calls) == 1  # reconciliation
        assert "voice_profile" in result

    def test_canonical_all_models_fail_raises(self):
        """All models fail → SynthesisError raised."""
        from synthesize import synthesize, SynthesisError

        docs = [_make_doc("Test " * 100)]

        with patch("synthesize.call_all", return_value={
            "claude": {"failed": True, "error": "timeout", "tokens": {}},
            "openai": {"failed": True, "error": "timeout", "tokens": {}},
        }):
            with pytest.raises(SynthesisError):
                synthesize(docs, _USER_CONFIG, mode="canonical")


class TestDetectMode:
    def test_detect_mode_call_graph(self):
        """detect mode: detection + consolidation + N per-voice × M models + reconciliation."""
        from synthesize import synthesize

        docs = [_make_doc(f"Article {i} " * 100) for i in range(10)]
        calls = []

        def _fake_detect(docs, user_config, **kw):
            from detect import VoiceCluster
            c = VoiceCluster(
                label="technical analysis",
                description="analytical",
                features={"avg_sentence_words": [">", 16]},
                source_distribution={"wordpress": 1.0},
                sample_ids=[],
                assigned_docs=docs[:5],
                word_count=sum(d.word_count for d in docs[:5]),
                confidence="high",
            )
            calls.append("detect_voices")
            return [c]

        def _fake_classify(docs, clusters, ambiguity_threshold=0.2, per_voice_min_words=2000, **kw):
            calls.append("classify_documents")
            return {clusters[0].label: clusters[0].assigned_docs}, []

        def _fake_call_all(system_prompt, user_prompt, user_config, models=None, max_parallel=0, pass_name="", **kw):
            calls.append(f"call_all:{pass_name}")
            content = _PER_VOICE_SYNTHESIS if "voice_" in pass_name else _CANONICAL_SYNTHESIS
            return {
                "claude": {"content": content, "failed": False, "tokens": {}, "elapsed": 1.0, "_parsed": {
                    "voice_profile": "test", "audience_primary": "test",
                    "banned_words": ["utilize"], "banned_phrases": [], "positive_rules": ["rule1"],
                    "additional_banned_words": [], "additional_positive_rules": [],
                }},
                "openai": {"content": content, "failed": False, "tokens": {}, "elapsed": 1.0, "_parsed": {
                    "voice_profile": "test", "audience_primary": "test",
                    "banned_words": ["utilize"], "banned_phrases": [], "positive_rules": ["rule1"],
                    "additional_banned_words": [], "additional_positive_rules": [],
                }},
            }

        def _fake_call_one(model_name, model_cfg, api_keys, system_prompt, user_prompt, pass_name=""):
            calls.append(f"call_one:{pass_name}")
            return {"content": _RECONCILE_DETECT, "failed": False, "tokens": {}, "elapsed": 1.0}

        with patch("synthesize.detect_voices", side_effect=_fake_detect), \
             patch("synthesize.classify_documents", side_effect=_fake_classify), \
             patch("synthesize.call_all", side_effect=_fake_call_all), \
             patch("synthesize.call_one", side_effect=_fake_call_one):
            result = synthesize(docs, _USER_CONFIG, mode="detect")

        assert "canonical" in result
        assert "detected_voices" in result
        # detected_voices keyed by model-generated label, not register names
        assert "technical analysis" in result["detected_voices"]

    def test_detect_low_confidence_fallback(self):
        """CanonicalFallbackWarning during detection → falls back to canonical."""
        from synthesize import synthesize
        from detect import CanonicalFallbackWarning

        docs = [_make_doc("Text " * 100)]

        with patch("synthesize.detect_voices", side_effect=CanonicalFallbackWarning("low confidence")), \
             patch("synthesize.call_all", return_value={
                 "claude": {"content": _CANONICAL_SYNTHESIS, "failed": False, "tokens": {}, "elapsed": 1.0, "_parsed": {
                     "voice_profile": "test", "audience_primary": "test",
                     "banned_words": [], "banned_phrases": [], "positive_rules": [],
                 }},
             }), \
             patch("synthesize.call_one", return_value={"content": _RECONCILE_CANONICAL, "failed": False, "tokens": {}, "elapsed": 1.0}):
            result = synthesize(docs, _USER_CONFIG, mode="detect")

        assert "voice_profile" in result
        assert result.get("_fallback_reason")


class TestValidateSynthesisOutput:
    def test_canonical_missing_key_raises(self):
        """SynthesisError raised when canonical profile missing required keys."""
        from synthesize import validate_synthesis_output, SynthesisError

        with pytest.raises(SynthesisError, match="missing required keys"):
            validate_synthesis_output({"voice_profile": "test"}, "canonical")

    def test_detect_missing_canonical_raises(self):
        """SynthesisError raised when detect result missing canonical section."""
        from synthesize import validate_synthesis_output, SynthesisError

        with pytest.raises(SynthesisError):
            validate_synthesis_output({"detected_voices": {}}, "detect")

    def test_valid_canonical_passes(self):
        """Valid canonical profile passes validation."""
        from synthesize import validate_synthesis_output

        profile = {
            "voice_profile": "text",
            "audience_primary": "audience",
            "banned_words": ["word"],
            "banned_phrases": ["phrase"],
            "positive_rules": ["rule"],
        }
        result = validate_synthesis_output(profile, "canonical")
        assert result is profile
