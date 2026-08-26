"""Tests for env_provenance — detecting when an OS env var shadows .env values.

This is the diagnostic that would have caught a real incident: a persistent
Windows User-scoped OPENAI_API_KEY silently overrode whatever was in .env
(python-dotenv's load_dotenv() defaults to override=False), and nothing said
so for days while the pipeline kept billing the wrong OpenAI org.
"""

import pytest

from ci_core.env_provenance import provenance, shadowed_mismatches, snapshot


@pytest.fixture
def dotenv_file(tmp_path):
    p = tmp_path / ".env"
    p.write_text("FOO=file_value\nBAR=another_value\n", encoding="utf-8")
    return str(p)


class TestSnapshot:
    def test_captures_pre_existing_os_keys(self, monkeypatch, dotenv_file):
        monkeypatch.setenv("ALREADY_SET", "x")
        snap = snapshot(dotenv_file)
        assert "ALREADY_SET" in snap["pre_existing_keys"]

    def test_reads_dotenv_file_values(self, dotenv_file):
        snap = snapshot(dotenv_file)
        assert snap["file_values"] == {"FOO": "file_value", "BAR": "another_value"}

    def test_no_dotenv_path_has_empty_file_values(self):
        snap = snapshot(None)
        assert snap["file_values"] == {}
        assert snap["dotenv_path"] is None

    def test_falsy_dotenv_path_normalizes_to_none(self):
        # find_dotenv() returns "" (not None) when nothing is found.
        snap = snapshot("")
        assert snap["dotenv_path"] is None
        assert snap["file_values"] == {}


class TestProvenance:
    def test_var_not_pre_existing_and_loaded_from_dotenv(
        self, monkeypatch, dotenv_file
    ):
        monkeypatch.delenv("FOO", raising=False)
        snap = snapshot(dotenv_file)
        monkeypatch.setenv("FOO", "file_value")  # simulate load_dotenv() applying it
        prov = provenance(snap, "FOO")
        assert prov["shadowed_by_os_env"] is False
        assert prov["mismatched"] is False
        assert prov["in_dotenv_file"] is True
        assert prov["active_value"] == "file_value"

    def test_pre_existing_and_matching_is_shadowed_but_not_mismatched(
        self, monkeypatch, dotenv_file
    ):
        monkeypatch.setenv("FOO", "file_value")  # OS already has the same value
        snap = snapshot(dotenv_file)
        prov = provenance(snap, "FOO")
        assert prov["shadowed_by_os_env"] is True
        assert prov["mismatched"] is False

    def test_pre_existing_and_different_is_mismatched(self, monkeypatch, dotenv_file):
        """The actual incident: the OS env wins silently, and its value
        differs from .env's -- rotating the key in .env has zero effect."""
        monkeypatch.setenv("FOO", "stale_os_value")
        snap = snapshot(dotenv_file)
        prov = provenance(snap, "FOO")
        assert prov["shadowed_by_os_env"] is True
        assert prov["mismatched"] is True
        assert prov["active_value"] == "stale_os_value"
        assert prov["dotenv_value"] == "file_value"

    def test_var_set_in_os_only_not_in_dotenv_at_all(self, monkeypatch, dotenv_file):
        monkeypatch.setenv("SHELL_ONLY", "shell_value")
        snap = snapshot(dotenv_file)
        prov = provenance(snap, "SHELL_ONLY")
        assert prov["in_dotenv_file"] is False
        assert prov["shadowed_by_os_env"] is True
        assert prov["mismatched"] is False  # nothing in .env to disagree with

    def test_var_unset_anywhere(self, dotenv_file):
        snap = snapshot(dotenv_file)
        prov = provenance(snap, "NEVER_SET_XYZ")
        assert prov["active_value"] is None
        assert prov["shadowed_by_os_env"] is False
        assert prov["in_dotenv_file"] is False
        assert prov["mismatched"] is False


class TestShadowedMismatches:
    def test_finds_only_the_mismatched_ones(self, monkeypatch, dotenv_file):
        monkeypatch.setenv("FOO", "stale_os_value")  # mismatched
        monkeypatch.setenv("BAR", "another_value")  # pre-existing but matches
        snap = snapshot(dotenv_file)
        names = {m["var_name"] for m in shadowed_mismatches(snap)}
        assert names == {"FOO"}

    def test_empty_when_nothing_shadowed(self, monkeypatch, dotenv_file):
        monkeypatch.delenv("FOO", raising=False)
        monkeypatch.delenv("BAR", raising=False)
        snap = snapshot(dotenv_file)
        assert shadowed_mismatches(snap) == []

    def test_empty_when_shadowed_but_matching(self, monkeypatch, dotenv_file):
        monkeypatch.setenv("FOO", "file_value")
        monkeypatch.setenv("BAR", "another_value")
        snap = snapshot(dotenv_file)
        assert shadowed_mismatches(snap) == []
