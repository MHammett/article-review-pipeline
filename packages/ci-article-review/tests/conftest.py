"""Shared fixtures for the ci-article-review suite."""

import hashlib

import pytest

from ci_article_review import live_model_check
from ci_article_review.adapters.citation import wayback


@pytest.fixture(scope="session")
def _model_discovery_cache_dir(tmp_path_factory):
    """One scratch directory per session for ``isolate_model_discovery_cache``."""
    return tmp_path_factory.mktemp("model_discovery_caches")


@pytest.fixture(autouse=True)
def isolate_model_discovery_cache(_model_discovery_cache_dir, request, monkeypatch):
    """Point the model-discovery cache at a scratch path for every test.

    ``live_model_check.CACHE_PATH`` is relative to the working directory, and
    pytest does not chdir — so without this, any test that runs the pipeline
    reads (and ``ci-discover``'s tests write) the developer's real
    ``.cache/model_discovery.json`` in the repo root.

    That is exactly the kind of environment-dependent test this suite has been
    careful to avoid: whether the golden report matches would depend on whether
    whoever ran the suite had happened to run ``ci-discover`` recently, and it
    would pass in CI — which checks out a clean tree — every time.

    What isolates a test here is the *filename*, not a directory of its own.
    This asked for ``tmp_path``, which makes pytest create a fresh directory
    per test — 1,119 of them, 2.6s of the suite's runtime, for a file most of
    those tests never write. A digest of the node id gives the same guarantee
    (unique per test, stable across runs) inside one session directory; and
    ``save_cache`` mkdirs its own parent, so nothing depends on the path
    existing beforehand.
    """
    unique = hashlib.sha256(request.node.nodeid.encode("utf-8")).hexdigest()[:16]
    monkeypatch.setattr(
        live_model_check,
        "CACHE_PATH",
        _model_discovery_cache_dir / f"{unique}.json",
    )


@pytest.fixture(autouse=True)
def neutralise_wayback_pacing(monkeypatch):
    """Take the archive.org pacing clock out of every test's wall-clock cost.

    ``wayback`` paces its calls to archive.org at one every
    ``_MIN_INTERVAL_SECONDS`` (3.0) and backs off from a 429 on a shared clock.
    Both are process-wide by design — ``check()`` runs on a resolver thread
    pool, so per-thread pacing would not pace anything — which means the clock
    also outlives the test that moved it. The cost of that leaked across the
    suite: six tests in ``TestWaybackCheck`` alone spent 27 seconds sitting in
    ``_pace()``, none of them about pacing, several of them waiting out an
    interval a *previous* test's call had started.

    Zeroing the intervals removes no coverage: every test that is about the
    guard already patches these to 0.0 itself, so nothing here ever exercised
    the wait. The wait is now asserted directly, and cheaply, by
    ``TestPacingClock`` in ``test_wayback.py`` — a test that patches the
    interval back up and checks what ``_pace()`` asks to sleep for.

    The state reset on both sides is the same one ``run_draft_pipeline`` does
    per run, for the same reason: a breaker tripped by one test would otherwise
    skip every archive lookup in the next.
    """
    monkeypatch.setattr(wayback, "_MIN_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(wayback, "_BACKOFF_BASE_SECONDS", 0.0)
    wayback.reset_rate_limit_state()
    yield
    wayback.reset_rate_limit_state()
