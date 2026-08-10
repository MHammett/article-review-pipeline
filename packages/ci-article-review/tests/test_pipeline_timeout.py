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

import pytest


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
