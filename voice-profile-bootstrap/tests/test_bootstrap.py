"""Integration tests for bootstrap.py CLI."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _make_doc(text: str = "This is test content. " * 60, source: str = "wordpress"):
    from collectors.base import Document
    doc = Document.from_text(text=text, source=source, register="long_form", date="2024-01-15", url_or_id="http://ex.com/1")
    doc.metrics = {"avg_sentence_words": 15.0, "hedging_ratio": 0.05, "first_person_ratio": 0.2}
    return doc


_MOCK_DOCS = [_make_doc(f"Article content {i}. " * 80) for i in range(10)]

_CANONICAL_RESULT = {
    "voice_profile": "Clear analytical voice.",
    "audience_primary": "Professionals",
    "audience_secondary": None,
    "banned_words": ["utilize"],
    "banned_phrases": [],
    "positive_rules": ["Be direct"],
}


def _run_bootstrap(*args):
    from bootstrap import main
    return main(list(args))


def _make_mock_registry(*source_names):
    """Return a REGISTRY dict with no-op mock collectors for the given source names."""
    from collectors.base import Collector

    registry = {}
    for name in source_names:
        _name = name

        class _MockCollector(Collector):
            SOURCE_NAME = _name

            @classmethod
            def validate_config(cls, config):
                pass

            def fetch(self, since=None):
                return iter([])

        _MockCollector.__name__ = f"MockCollector_{name}"
        registry[name] = _MockCollector
    return registry


class TestDryRun:
    def test_dry_run_exits_before_synthesis(self):
        """--dry-run exits before synthesis; no output file written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.yaml"

            with patch("bootstrap._load_sources_yaml", return_value={}), \
                 patch("bootstrap._load_user_config_lenient", return_value={}), \
                 patch("collectors.REGISTRY", _make_mock_registry("wordpress")), \
                 patch("bootstrap._collect_source", return_value=_MOCK_DOCS):
                rc = _run_bootstrap(
                    "--output-yaml", str(output_path),
                    "--sources", "wordpress",
                    "--voice", "canonical",
                    "--dry-run",
                )

            assert rc == 0
            assert not output_path.exists()


class TestContinueOnError:
    def test_continue_on_error_skips_failed_source(self):
        """--continue-on-error: one collector raises CollectorError; run completes."""
        from collectors.base import CollectorError

        def _fail_collect(source, *a, **kw):
            if source == "gmail":
                raise CollectorError("gmail", "Auth failed")
            return _MOCK_DOCS

        _RECONCILE = """{
          "canonical": {"voice_profile": "test", "audience_primary": "test", "banned_words": [], "banned_phrases": [], "positive_rules": [], "confidence": "high"},
          "detected_voices": {}, "synthesis_notes": ""
        }"""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.yaml"
            with patch("bootstrap._load_sources_yaml", return_value={}), \
                 patch("bootstrap._load_user_config_lenient", return_value={
                     "models": {"claude": {"provider": "anthropic", "model": "claude-sonnet-4-6"}},
                     "api_keys": {"claude": {"api_key": "test"}},
                 }), \
                 patch("collectors.REGISTRY", _make_mock_registry("wordpress", "gmail")), \
                 patch("bootstrap._collect_source", side_effect=_fail_collect), \
                 patch("synthesize.call_all", return_value={
                     "claude": {"content": '{"voice_profile": "t", "audience_primary": "t", "banned_words": [], "banned_phrases": [], "positive_rules": []}',
                                "failed": False, "tokens": {}, "elapsed": 1.0, "_parsed": {"voice_profile": "t", "audience_primary": "t", "banned_words": [], "banned_phrases": [], "positive_rules": []}},
                 }), \
                 patch("synthesize.call_one", return_value={"content": _RECONCILE, "failed": False, "tokens": {}, "elapsed": 1.0}):
                rc = _run_bootstrap(
                    "--output-yaml", str(output_path),
                    "--sources", "wordpress,gmail",
                    "--voice", "canonical",
                    "--continue-on-error",
                    "--overwrite",
                )

            # Should complete even though gmail failed
            assert rc == 0

    def test_stop_on_error_without_flag(self):
        """Without --continue-on-error: collector error causes exit code 1."""
        from collectors.base import CollectorError

        with patch("bootstrap._load_sources_yaml", return_value={}), \
             patch("bootstrap._load_user_config_lenient", return_value={}), \
             patch("collectors.REGISTRY", _make_mock_registry("wordpress")), \
             patch("bootstrap._collect_source", side_effect=CollectorError("wordpress", "Server error")):
            rc = _run_bootstrap(
                "--output-yaml", "/tmp/out.yaml",
                "--sources", "wordpress",
                "--voice", "canonical",
                "--dry-run",  # dry-run but error happens at collection
            )

        assert rc == 1


class TestPublicationFlag:
    def test_publication_resolves_to_configs(self):
        """--publication mikehammett resolves to configs/mikehammett.yaml."""
        from bootstrap import _resolve_output_path
        path = _resolve_output_path("mikehammett", None)
        assert str(path) == "configs/mikehammett.yaml"

    def test_output_yaml_explicit_path(self):
        """--output-yaml sets explicit output path."""
        from bootstrap import _resolve_output_path
        path = _resolve_output_path(None, "/tmp/my_profile.yaml")
        assert str(path) == "/tmp/my_profile.yaml"


class TestRefreshFlag:
    def test_refresh_clears_watermarks(self):
        """--refresh: watermarks cleared before collection."""
        with patch("bootstrap._load_sources_yaml", return_value={}), \
             patch("bootstrap._load_user_config_lenient", return_value={}), \
             patch("collectors.REGISTRY", _make_mock_registry("wordpress")), \
             patch("bootstrap._load_watermarks", return_value={"wordpress": "2024-01-01"}), \
             patch("bootstrap._collect_source", return_value=_MOCK_DOCS) as mock_collect:

            _run_bootstrap(
                "--output-yaml", "/tmp/out.yaml",
                "--sources", "wordpress",
                "--voice", "canonical",
                "--dry-run",
                "--refresh",
            )

            # When --refresh, watermarks should be {} (cleared) so _collect_source is called
            assert mock_collect.called
            call_kwargs = mock_collect.call_args
            # watermarks kwarg should be {} (cleared by --refresh)
            watermarks_arg = call_kwargs.kwargs.get("watermarks", call_kwargs.args[4] if len(call_kwargs.args) > 4 else None)
            if watermarks_arg is not None:
                assert watermarks_arg == {}


class TestCheckDraftNotImplemented:
    def test_check_draft_raises_not_implemented(self):
        """--check-draft raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            _run_bootstrap("--publication", "test", "--check-draft", "/tmp/draft.md")
