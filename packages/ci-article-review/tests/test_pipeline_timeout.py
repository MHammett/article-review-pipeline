"""Tests for the review pipeline's per-task wall-clock backstop.

Modelled on ci-style-profile's ``test_call_all_applies_wall_clock_backstop``:
the point of a backstop is that the run stops *waiting*, so these assert the
call returns before the slow work finishes, not merely that a TimeoutError is
eventually raised.
"""

from __future__ import annotations

import threading
import time

from unittest.mock import patch


import ci_article_review.pipeline as pipeline


class TestRunReviewsInParallel:
    """Exercises ``_run_reviews_in_parallel`` — the pipeline's real fan-out for
    the (model, domain) review calls, not a standalone reimplementation of the
    pattern. It delegates the daemon-thread mechanics to
    :func:`ci_core.concurrency.run_all_with_timeout`, which has its own
    thorough coverage (including the subprocess-level "does not hold the
    interpreter open" regression); these tests are about this function's own
    bookkeeping — result shape, per-model timeout lookup, and that a slow call
    doesn't stall its batch-mates.
    """

    def test_returns_result_when_under_budget(self):
        runners = [("claude:accuracy", lambda: {"ok": True})]
        results = pipeline._run_reviews_in_parallel(runners, {}, {}, task_timeout=5)
        assert results["claude:accuracy"] == {"ok": True}

    def test_propagates_exceptions_from_the_task(self):
        def _boom():
            raise ValueError("adapter blew up")

        runners = [("claude:accuracy", _boom)]
        results = pipeline._run_reviews_in_parallel(runners, {}, {}, task_timeout=5)
        assert results["claude:accuracy"]["failed"] is True
        assert "adapter blew up" in results["claude:accuracy"]["error"]

    def test_gives_up_before_the_slow_call_completes(self):
        """The backstop must bound wall-clock time, not just detect the overrun.

        Under streaming the socket timeout is only the inter-token read gap, so
        a model that keeps dribbling tokens has nothing else to stop it — the
        thread backstop has to be what returns control, not the call itself
        finishing.
        """
        finished = threading.Event()

        def _slow():
            time.sleep(2)
            finished.set()
            return {"failed": False}

        runners = [("claude:accuracy", _slow)]
        model_configs = {"claude": {"timeout_seconds": 0.2}}

        started = time.monotonic()
        results = pipeline._run_reviews_in_parallel(
            runners, {}, model_configs, task_timeout=5
        )
        gave_up_after = time.monotonic() - started

        assert gave_up_after < 1.5
        assert results["claude:accuracy"]["failed"] is True
        assert "timed out after 0.2s" in results["claude:accuracy"]["error"].lower()
        # The abandoned call is genuinely still running — that is the accepted
        # tradeoff (a running thread cannot be killed), not a leak. Because it
        # runs on a daemon thread, it cannot hold the interpreter open either
        # (see ci_core.concurrency for the regression test proving that).
        assert not finished.is_set()
        finished.wait(timeout=5)

    def test_one_slow_task_does_not_hold_up_the_batch(self):
        """Shape of the pipeline's parallel block: fast passes are not blocked.

        Previously the outer executor's own ``with`` exit joined its workers —
        including ones that had already been given up on — so a backstop that
        only *detected* the overrun would stall the whole batch until the
        slowest call finished.
        """

        def _slow():
            time.sleep(2)
            return {"failed": False}

        runners = [
            ("claude:accuracy", _slow),
            ("openai:structure", lambda: {"failed": False}),
        ]
        model_configs = {"claude": {"timeout_seconds": 0.2}}

        started = time.monotonic()
        results = pipeline._run_reviews_in_parallel(
            runners, {}, model_configs, task_timeout=5
        )
        batch_elapsed = time.monotonic() - started

        assert batch_elapsed < 1.5
        assert results["openai:structure"] == {"failed": False}
        assert results["claude:accuracy"]["failed"] is True
        assert "timed out" in results["claude:accuracy"]["error"].lower()

    def test_an_explicit_null_task_timeout_does_not_crash(self):
        """``task_timeout_seconds: null`` in user.yaml is a real, reachable
        value — config_loader's 180 default only fires when the key is
        *absent*, not when it is present and null, which is exactly the shape
        of the PR #105 regression. Computing this runner's own timeout must
        not raise trying to add a stagger offset to None; it falls back to
        being bound by the group's global ceiling instead (see
        ci_core.concurrency.run_all_with_timeout)."""
        runners = [("claude:accuracy", lambda: {"ok": True})]
        results = pipeline._run_reviews_in_parallel(runners, {}, {}, task_timeout=None)
        assert results["claude:accuracy"] == {"ok": True}


class TestSameProviderStagger:
    """A provider's own calls must not all fire in the same instant.

    Rate limits are per account, not per call, so five simultaneous Perplexity
    requests compete for one quota. Observed 2026-08-12: two returned HTTP 429
    within one second of each other and one failed outright after its retry also
    hit the limit, which then cost Section 9 all of its grounded-search URLs.
    """

    def test_calls_to_one_provider_are_spread(self):
        names = [
            f"perplexity:{d}"
            for d in (
                "fact_check",
                "voice_style",
                "completeness",
                "argument_integrity",
                "red_team",
            )
        ]
        offsets = pipeline._stagger_offsets(names, 3)
        assert sorted(offsets.values()) == [0, 3, 6, 9, 12]

    def test_different_providers_still_start_together(self):
        """The parallelism that matters is across providers, not within one."""
        names = ["perplexity:fact_check", "gemini:fact_check", "grok:fact_check"]
        assert set(pipeline._stagger_offsets(names, 3).values()) == {0}

    def test_stagger_of_zero_disables_it(self):
        names = [f"perplexity:{d}" for d in ("a", "b", "c")]
        assert set(pipeline._stagger_offsets(names, 0).values()) == {0}

    def test_delay_start_is_a_passthrough_at_zero(self):
        """No offset must mean no wrapper, so the common case pays nothing."""
        fn = lambda: "ran"  # noqa: E731
        assert pipeline._delay_start(fn, 0) is fn

    def test_delay_start_sleeps_then_runs(self):
        slept = []
        fn = pipeline._delay_start(lambda: "ran", 6)
        with patch.object(pipeline.time, "sleep", side_effect=slept.append):
            assert fn() == "ran"
        assert slept == [6]
