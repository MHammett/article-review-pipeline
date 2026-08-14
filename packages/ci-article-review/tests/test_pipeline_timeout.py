"""Tests for the review pipeline's per-task wall-clock backstop.

Modelled on ci-style-profile's ``test_call_all_applies_wall_clock_backstop``:
the point of a backstop is that the run stops *waiting*, so these assert the
call returns before the slow work finishes, not merely that a TimeoutError is
eventually raised.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time

from unittest.mock import patch

import pytest

import ci_article_review.pipeline as pipeline


class TestRunWithTimeout:
    def test_returns_result_when_under_budget(self):
        from ci_article_review.pipeline import _run_with_timeout

        assert _run_with_timeout(lambda: {"ok": True}, 5, "claude:accuracy") == {
            "ok": True
        }

    def test_propagates_exceptions_from_the_task(self):
        from ci_article_review.pipeline import _run_with_timeout

        def _boom():
            raise ValueError("adapter blew up")

        with pytest.raises(ValueError, match="adapter blew up"):
            _run_with_timeout(_boom, 5, "claude:accuracy")

    def test_gives_up_before_the_slow_call_completes(self):
        """The backstop must bound wall-clock time, not just detect the overrun.

        The context-manager form of ThreadPoolExecutor re-joins the worker on
        the way out, so the TimeoutError was previously raised only *after* the
        slow call had run to completion. Under streaming the socket timeout is
        only the inter-token read gap, so a model that keeps dribbling tokens
        has nothing else to stop it.
        """
        from ci_article_review.pipeline import _run_with_timeout

        finished = threading.Event()

        def _slow():
            time.sleep(2)
            finished.set()
            return {"failed": False}

        started = time.monotonic()
        with pytest.raises(TimeoutError, match="Timed out after 0.2s"):
            _run_with_timeout(_slow, 0.2, "claude:accuracy")
        gave_up_after = time.monotonic() - started

        assert gave_up_after < 1.5
        # The abandoned call is genuinely still running — that is the accepted
        # tradeoff (a running thread cannot be killed), not a leak.
        assert not finished.is_set()
        finished.wait(timeout=5)

    def test_one_slow_task_does_not_hold_up_the_batch(self):
        """Shape of the pipeline's parallel block: fast passes are not blocked.

        The outer executor's own ``with`` exit joins its workers, so a backstop
        that only *detects* the overrun would stall the whole batch until the
        slowest call finished.
        """
        from ci_article_review.pipeline import _run_with_timeout

        def _slow():
            time.sleep(2)
            return {"failed": False}

        runners = [
            ("claude:accuracy", _slow, 0.2),
            ("openai:structure", lambda: {"failed": False}, 5),
        ]

        started = time.monotonic()
        results: dict[str, dict] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_name = {
                executor.submit(_run_with_timeout, fn, timeout, name): name
                for name, fn, timeout in runners
            }
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = {"failed": True, "error": str(e)}
        batch_elapsed = time.monotonic() - started

        assert batch_elapsed < 1.5
        assert results["openai:structure"] == {"failed": False}
        assert results["claude:accuracy"]["failed"] is True
        assert "timed out" in results["claude:accuracy"]["error"].lower()


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
