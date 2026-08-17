"""Shared fixtures for the ci-article-review suite."""

import pytest

from ci_article_review import live_model_check


@pytest.fixture(autouse=True)
def isolate_model_discovery_cache(tmp_path, monkeypatch):
    """Point the model-discovery cache at a scratch path for every test.

    ``live_model_check.CACHE_PATH`` is relative to the working directory, and
    pytest does not chdir — so without this, any test that runs the pipeline
    reads (and ``ci-discover``'s tests write) the developer's real
    ``.cache/model_discovery.json`` in the repo root.

    That is exactly the kind of environment-dependent test this suite has been
    careful to avoid: whether the golden report matches would depend on whether
    whoever ran the suite had happened to run ``ci-discover`` recently, and it
    would pass in CI — which checks out a clean tree — every time.
    """
    monkeypatch.setattr(
        live_model_check, "CACHE_PATH", tmp_path / "model_discovery.json"
    )
