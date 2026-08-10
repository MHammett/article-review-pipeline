"""Tests that fail when a pre-commit hook drifts away from the locked toolchain.

`.pre-commit-config.yaml` pins each hook to a `rev`, and `uv.lock` pins the same
tool for CI. Nothing keeps the two in step, so they drift — and the drift is
quiet. The local hook keeps passing; it just enforces a different version than
CI does, which shows up as a lint or type error that reproduces in one place and
not the other.

This has cost the repo twice. Commit 0a2b342 bumped ruff-pre-commit v0.4.4 →
v0.15.19 after the hook and CI disagreed. The mypy hook then sat at v1.9.0 while
the lockfile moved to 2.1.0, which produced a false `attr-defined` failure on
pipeline.py. Both were mechanically detectable, and this test detects them.

Like the docs-drift tests next door, the failure message names the file, the
hook, and the two versions, so whoever trips it can fix it in a minute.
"""

import tomllib
from pathlib import Path

import pytest
import yaml


def _find_repo_root():
    """Locate the workspace root by walking up from this file.

    Mirrors test_docs_current.py: these tests read repo-root files, so they must
    not depend on pytest's working directory.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "packages").is_dir() and (candidate / "uv.lock").is_file():
            return candidate
    raise RuntimeError(
        "Could not locate the content-intelligence repo root above "
        f"{Path(__file__).resolve()} — expected an ancestor containing both "
        "packages/ and uv.lock."
    )


REPO_ROOT = _find_repo_root()
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
UV_LOCK = REPO_ROOT / "uv.lock"


# Which locked distribution each hook repo ships. The hook's `rev` is that
# distribution's version with a leading "v", which is the convention both
# upstreams follow (ruff-pre-commit tags v0.15.19 for ruff 0.15.19;
# mirrors-mypy tags v2.1.0 for mypy 2.1.0).
#
# Map a repo to None when its hooks genuinely have no counterpart in uv.lock —
# pre-commit-hooks and similar utility repos ship no tool we install. Adding a
# new hook repo is then a one-line change here, and forgetting to add it fails
# test_every_hook_repo_is_classified rather than silently going unchecked.
HOOK_REPO_TO_LOCKED_PACKAGE = {
    "https://github.com/astral-sh/ruff-pre-commit": "ruff",
    "https://github.com/pre-commit/mirrors-mypy": "mypy",
}


def _pre_commit_repos():
    """The `repos:` entries, excluding pre-commit's built-in meta/local repos.

    `repo: local` and `repo: meta` carry no rev to check.
    """
    config = yaml.safe_load(PRE_COMMIT_CONFIG.read_text(encoding="utf-8"))
    return [
        entry
        for entry in config.get("repos", [])
        if entry.get("repo") not in {"local", "meta"}
    ]


def _locked_versions():
    """name -> version for every package in uv.lock."""
    lock = tomllib.loads(UV_LOCK.read_text(encoding="utf-8"))
    return {package["name"]: package["version"] for package in lock["package"]}


def test_every_hook_repo_is_classified():
    """A new hook repo must be added to the map, not silently skipped."""
    unclassified = sorted(
        entry["repo"]
        for entry in _pre_commit_repos()
        if entry["repo"] not in HOOK_REPO_TO_LOCKED_PACKAGE
    )
    assert not unclassified, (
        f"{PRE_COMMIT_CONFIG.name} has hook repos that this test does not know "
        f"about: {', '.join(unclassified)}. Add each to "
        "HOOK_REPO_TO_LOCKED_PACKAGE in this file — map it to the name of the "
        "distribution it ships in uv.lock, or to None if it ships nothing we "
        "install."
    )


def test_map_lists_no_repos_the_config_dropped():
    """A stale map entry would exempt nothing — or hide a repo that was removed."""
    configured = {entry["repo"] for entry in _pre_commit_repos()}
    stale = sorted(set(HOOK_REPO_TO_LOCKED_PACKAGE) - configured)
    assert not stale, (
        "HOOK_REPO_TO_LOCKED_PACKAGE in this file lists repos that "
        f"{PRE_COMMIT_CONFIG.name} no longer uses: {', '.join(stale)}. Remove "
        "them from the map."
    )


def _version_checked_repos():
    """(repo_url, rev, locked_package) for each hook repo pinned to a locked tool."""
    cases = []
    for entry in _pre_commit_repos():
        locked_package = HOOK_REPO_TO_LOCKED_PACKAGE.get(entry["repo"])
        if locked_package is None:
            continue
        cases.append((entry["repo"], entry["rev"], locked_package))
    return cases


@pytest.mark.parametrize(
    "repo_url, rev, locked_package",
    _version_checked_repos(),
    ids=lambda value: value.rsplit("/", 1)[-1] if isinstance(value, str) else value,
)
def test_hook_rev_matches_locked_version(repo_url, rev, locked_package):
    locked_versions = _locked_versions()
    assert locked_package in locked_versions, (
        f"HOOK_REPO_TO_LOCKED_PACKAGE maps {repo_url} to '{locked_package}', "
        f"which {UV_LOCK.name} does not contain. Fix the mapping in this file."
    )

    locked_version = locked_versions[locked_package]
    assert rev.startswith("v"), (
        f"{PRE_COMMIT_CONFIG.name} pins {repo_url} to rev '{rev}', which is not "
        f"a vN.N.N release tag. This test compares hook revs against "
        f"{UV_LOCK.name}; a branch or commit sha cannot be compared. Pin the "
        f"tag matching {locked_package} {locked_version}."
    )

    hook_version = rev[1:]
    assert hook_version == locked_version, (
        f"Toolchain skew: {PRE_COMMIT_CONFIG.name} pins {repo_url} at "
        f"{rev} ({locked_package} {hook_version}), but {UV_LOCK.name} resolves "
        f"{locked_package} to {locked_version} — and CI runs the locked "
        f"version. The hook and CI would enforce different rules.\n"
        f"Fix: set that repo's rev to 'v{locked_version}' in "
        f"{PRE_COMMIT_CONFIG.name}, then run "
        f"`uv run pre-commit run --all-files`."
    )
