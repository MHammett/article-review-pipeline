"""Tests for detect.py — voice detection, classification, cluster validation."""

from __future__ import annotations

from unittest.mock import patch

import pytest


def _make_doc(
    text: str,
    source: str = "wordpress",
    date: str = "2024-01-01",
    url: str = "http://ex.com/1",
    avg_sentence_words: float = 15.0,
    hedging_ratio: float = 0.1,
    first_person_ratio: float = 0.3,
):
    from ci_web_intel.collectors.base import Document

    doc = Document.from_text(
        text=text, source=source, register="long_form", date=date, url_or_id=url
    )
    doc.metrics = {
        "avg_sentence_words": avg_sentence_words,
        "hedging_ratio": hedging_ratio,
        "first_person_ratio": first_person_ratio,
        "vocab_richness": 0.5,
    }
    return doc


_DETECTION_RESPONSE = """{
  "detected_voices": [
    {
      "label": "technical analysis",
      "description": "Long analytical prose",
      "features": {"avg_sentence_words": [">", 18], "hedging_ratio": ["<", 0.05]},
      "source_distribution": {"wordpress": 1.0},
      "sample_ids": ["http://ex.com/1"],
      "confidence": "high"
    },
    {
      "label": "direct editorial",
      "description": "Short opinionated",
      "features": {"avg_sentence_words": ["<", 14], "first_person_ratio": [">", 0.4]},
      "source_distribution": {"wordpress": 1.0},
      "sample_ids": ["http://ex.com/2"],
      "confidence": "medium"
    }
  ],
  "overall_confidence": "high",
  "detection_notes": "clear separation"
}"""


class TestDetectVoices:
    def test_detect_voices_returns_clusters(self):
        """detect_voices: mock detection response → VoiceCluster list returned with correct fields."""
        from ci_web_intel.detect import detect_voices, VoiceCluster

        docs = [
            _make_doc(f"Document {i} " * 100, url=f"http://ex.com/{i}")
            for i in range(10)
        ]
        user_config = {
            "models": {
                "claude": {"provider": "anthropic", "model": "claude-sonnet-4-6"}
            },
            "api_keys": {"claude": {"api_key": "test"}},
        }

        mock_result = {
            "content": _DETECTION_RESPONSE,
            "failed": False,
            "tokens": {},
            "elapsed": 1.0,
        }

        with patch(
            "ci_web_intel.detect.call_all", return_value={"claude": mock_result}
        ):
            clusters = detect_voices(docs, user_config, max_voices=5)

        assert len(clusters) == 2
        labels = [c.label for c in clusters]
        assert "technical analysis" in labels
        assert "direct editorial" in labels
        assert all(isinstance(c, VoiceCluster) for c in clusters)

    def test_low_confidence_raises_warning(self):
        """overall_confidence: 'low' → CanonicalFallbackWarning raised."""
        from ci_web_intel.detect import detect_voices, CanonicalFallbackWarning

        low_conf_response = _DETECTION_RESPONSE.replace(
            '"overall_confidence": "high"', '"overall_confidence": "low"'
        )
        docs = [_make_doc("Text " * 50)]
        user_config = {
            "models": {
                "claude": {"provider": "anthropic", "model": "claude-sonnet-4-6"}
            },
            "api_keys": {"claude": {"api_key": "test"}},
        }

        mock_result = {
            "content": low_conf_response,
            "failed": False,
            "tokens": {},
            "elapsed": 1.0,
        }

        with patch(
            "ci_web_intel.detect.call_all", return_value={"claude": mock_result}
        ):
            with pytest.raises(CanonicalFallbackWarning):
                detect_voices(docs, user_config)

    def test_all_models_fail_returns_empty(self):
        """All detection models fail → empty list returned (caller falls back to canonical)."""
        from ci_web_intel.detect import detect_voices

        docs = [_make_doc("Text " * 50)]
        user_config = {
            "models": {
                "claude": {"provider": "anthropic", "model": "claude-sonnet-4-6"}
            },
            "api_keys": {"claude": {"api_key": "test"}},
        }

        with patch(
            "ci_web_intel.detect.call_all",
            return_value={"claude": {"failed": True, "error": "timeout", "tokens": {}}},
        ):
            result = detect_voices(docs, user_config)

        assert result == []


class TestClassifyDocuments:
    def test_correct_cluster_assignment(self):
        """classify_documents: docs with known metrics → correct cluster assignment."""
        from ci_web_intel.detect import VoiceCluster, classify_documents

        long_doc = _make_doc(
            "Long technical " * 100, avg_sentence_words=22.0, hedging_ratio=0.02
        )
        short_doc = _make_doc(
            "Short personal " * 50, avg_sentence_words=10.0, first_person_ratio=0.6
        )

        clusters = [
            VoiceCluster(
                label="technical",
                description="Long analytical",
                features={
                    "avg_sentence_words": [">", 18],
                    "hedging_ratio": ["<", 0.05],
                },
                source_distribution={},
                sample_ids=[],
            ),
            VoiceCluster(
                label="editorial",
                description="Short personal",
                features={
                    "avg_sentence_words": ["<", 14],
                    "first_person_ratio": [">", 0.4],
                },
                source_distribution={},
                sample_ids=[],
            ),
        ]

        assigned, ambiguous = classify_documents(
            [long_doc, short_doc],
            clusters,
            ambiguity_threshold=0.2,
            per_voice_min_words=0,
        )

        assert long_doc in assigned.get("technical", [])
        assert short_doc in assigned.get("editorial", [])

    def test_ambiguous_doc_placement(self):
        """Docs with scores within ambiguity_threshold → placed in ambiguous bucket."""
        from ci_web_intel.detect import VoiceCluster, classify_documents

        # A doc that matches BOTH clusters equally (no clear winner)
        ambig_doc = _make_doc(
            "Ambiguous text " * 50, avg_sentence_words=16.0, first_person_ratio=0.3
        )

        clusters = [
            VoiceCluster(
                label="technical",
                description="Long",
                features={"avg_sentence_words": [">", 15]},
                source_distribution={},
                sample_ids=[],
            ),
            VoiceCluster(
                label="editorial",
                description="Personal",
                features={"avg_sentence_words": [">", 14]},
                source_distribution={},
                sample_ids=[],
            ),
        ]

        assigned, ambiguous = classify_documents(
            [ambig_doc], clusters, ambiguity_threshold=0.5
        )

        # Both features match both clusters — should be ambiguous
        assert ambig_doc in ambiguous

    def test_undersized_cluster_merged(self):
        """Cluster below per_voice_min_words → merged into nearest cluster with warning."""
        from ci_web_intel.detect import VoiceCluster, classify_documents

        small_docs = [_make_doc("Short " * 20, avg_sentence_words=10.0)]  # ~20 words
        large_docs = [
            _make_doc("Long technical " * 200, avg_sentence_words=22.0)
            for _ in range(5)
        ]

        clusters = [
            VoiceCluster(
                label="small_voice",
                description="Tiny cluster",
                features={"avg_sentence_words": ["<", 12]},
                source_distribution={},
                sample_ids=[],
            ),
            VoiceCluster(
                label="large_voice",
                description="Big cluster",
                features={"avg_sentence_words": [">", 20]},
                source_distribution={},
                sample_ids=[],
            ),
        ]

        all_docs = small_docs + large_docs
        assigned, ambiguous = classify_documents(
            all_docs, clusters, per_voice_min_words=500
        )

        # small_voice had < 500 words → merged into large_voice
        assert "small_voice" not in {c.label for c in clusters}
