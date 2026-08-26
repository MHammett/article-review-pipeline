"""Tests for env_provenance — resolving .env vs OS env precedence and
flagging when they disagree.

This is the diagnostic that would have caught a real incident: a persistent
Windows User-scoped OPENAI_API_KEY silently overrode whatever was in .env
(python-dotenv's load_dotenv() defaults to override=False), and nothing said
so for days while the pipeline kept billing the wrong OpenAI org. The project
now enforces .env > bare OS env explicitly (see module docstring) rather than
just detecting the old failure mode after the fact.
"""

import pytest

from ci_core.env_provenance import (
    effective_env,
    provenance,
    shadowed_mismatches,
    snapshot,
)


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

    def test_pre_existing_and_different_is_mismatched_but_dotenv_still_wins(
        self, monkeypatch, dotenv_file
    ):
        """The actual incident, as it now resolves: the OS env variable is
        still there and still differs, but .env wins the effective value —
        rotating the key in .env now has the effect the person expects."""
        monkeypatch.setenv("FOO", "stale_os_value")
        snap = snapshot(dotenv_file)
        prov = provenance(snap, "FOO")
        assert prov["shadowed_by_os_env"] is True
        assert prov["mismatched"] is True
        assert prov["active_value"] == "file_value"
        assert prov["os_env_value"] == "stale_os_value"
        assert prov["dotenv_value"] == "file_value"

    def test_var_set_in_os_only_not_in_dotenv_at_all(self, monkeypatch, dotenv_file):
        """No .env entry to prefer, so the OS variable is the effective
        value — the lowest-priority source is still a real fallback."""
        monkeypatch.setenv("SHELL_ONLY", "shell_value")
        snap = snapshot(dotenv_file)
        prov = provenance(snap, "SHELL_ONLY")
        assert prov["in_dotenv_file"] is False
        assert prov["shadowed_by_os_env"] is True
        assert prov["mismatched"] is False  # nothing in .env to disagree with
        assert prov["active_value"] == "shell_value"

    def test_var_unset_anywhere(self, dotenv_file):
        snap = snapshot(dotenv_file)
        prov = provenance(snap, "NEVER_SET_XYZ")
        assert prov["active_value"] is None
        assert prov["shadowed_by_os_env"] is False
        assert prov["in_dotenv_file"] is False
        assert prov["mismatched"] is False


class TestEffectiveEnv:
    def test_dotenv_value_overrides_a_differing_os_value(
        self, monkeypatch, dotenv_file
    ):
        monkeypatch.setenv("FOO", "stale_os_value")
        snap = snapshot(dotenv_file)
        assert effective_env(snap)["FOO"] == "file_value"

    def test_os_only_variable_still_passes_through(self, monkeypatch, dotenv_file):
        monkeypatch.setenv("SHELL_ONLY", "shell_value")
        snap = snapshot(dotenv_file)
        assert effective_env(snap)["SHELL_ONLY"] == "shell_value"

    def test_dotenv_only_variable_is_present_even_if_load_dotenv_never_ran(
        self, dotenv_file
    ):
        """effective_env must not depend on load_dotenv() having already
        applied the file to os.environ — it reads the snapshot directly."""
        snap = snapshot(dotenv_file)
        assert effective_env(snap)["BAR"] == "another_value"


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
