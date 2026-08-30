"""Tests for ci_core.concurrency.run_with_timeout.

The important one here is `test_an_abandoned_call_does_not_hold_the_process_open`,
and it has to spawn a subprocess: the defect it guards is about *interpreter
exit*, which nothing running inside the interpreter can observe. Every
in-process assertion passed against the broken version — the pipeline ran, the
report was written, the summary printed, and only the shell never came back.
"""

import subprocess
import sys
import textwrap
import threading
import time

import pytest

from ci_core.concurrency import run_all_with_timeout, run_with_timeout


class TestContract:
    def test_returns_the_value(self):
        assert run_with_timeout(lambda: 21 * 2, timeout=5) == 42

    def test_reraises_what_the_call_raised(self):
        """Callers must see provider failures exactly as if called inline."""

        def _boom():
            raise ValueError("provider said no")

        with pytest.raises(ValueError, match="provider said no"):
            run_with_timeout(_boom, timeout=5)

    def test_raises_timeout_error_on_expiry(self):
        with pytest.raises(TimeoutError, match="Timed out after"):
            run_with_timeout(lambda: time.sleep(5), timeout=0.1)

    def test_expiry_returns_promptly_rather_than_waiting_out_the_call(self):
        """The point of a backstop: stop waiting now, not when the call ends."""
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            run_with_timeout(lambda: time.sleep(10), timeout=0.2)
        assert time.monotonic() - t0 < 3

    def test_a_none_timeout_waits_for_the_call(self):
        """compute_all can hand back None for a model with no budget."""
        assert run_with_timeout(lambda: "done", timeout=None) == "done"

    def test_the_worker_is_a_daemon(self):
        """The whole fix in one assertion."""
        seen = {}

        def _capture():
            seen["daemon"] = threading.current_thread().daemon

        run_with_timeout(_capture, timeout=5)
        assert seen["daemon"] is True


class TestProcessExit:
    """The regression that cost two days of a process sitting on a log file."""

    # Longer than the assert below, so a process that waits for the abandoned
    # call cannot pass by finishing it.
    #
    # Sized from measurement, not guesswork. The two exit times these tests
    # separate do not scale together:
    #
    #   * the fixed (daemon-thread) form exits in ~0.57s — interpreter start
    #     plus importing ci_core — *independently* of how long the abandoned
    #     call still has to run;
    #   * the broken (executor) form exits at ~abandoned + 0.1s, because the
    #     atexit join waits the call out.
    #
    # So the only job of ABANDONED_SECONDS is to put daylight between those
    # two, and MUST_EXIT_WITHIN is the line drawn between them. At 5s/3s the
    # margin is >5x on both sides (0.57s vs the 3s line, 3s vs 5.1s), which is
    # as decisive as the 20s/10s this used to spend and 15s cheaper per run.
    ABANDONED_SECONDS = 5
    MUST_EXIT_WITHIN = 3

    def _run(self, body):
        script = textwrap.dedent(body).format(
            abandoned=self.ABANDONED_SECONDS,
        )
        t0 = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=self.ABANDONED_SECONDS + 30,
        )
        return proc, time.monotonic() - t0

    def test_an_abandoned_call_does_not_hold_the_process_open(self):
        """A timed-out call keeps running; it must not keep the interpreter alive.

        Against the single-worker-executor version this took exactly as long as
        the abandoned call, because concurrent.futures' atexit hook joins worker
        threads with a bare `t.join()` and no timeout. In production that meant
        `ci-review` processes alive two days after writing their report.
        """
        proc, elapsed = self._run(
            """
            import time
            from ci_core.concurrency import run_with_timeout

            try:
                run_with_timeout(lambda: time.sleep({abandoned}), timeout=0.5)
            except TimeoutError:
                print("backstop fired")
            print("exiting")
            """
        )

        assert proc.returncode == 0, proc.stderr
        assert "backstop fired" in proc.stdout
        assert "exiting" in proc.stdout
        assert elapsed < self.MUST_EXIT_WITHIN, (
            f"process took {elapsed:.1f}s to exit with a {self.ABANDONED_SECONDS}s "
            f"abandoned call still running — the abandoned worker is holding the "
            f"interpreter open, which is the bug this guards"
        )

    @pytest.mark.slow
    def test_the_executor_form_really_did_hang(self):
        """Pins the diagnosis, so the fix cannot be argued away later.

        If this ever stops hanging, CPython changed its atexit behaviour and the
        comment in ci_core.concurrency explaining the fix needs revisiting.

        Marked ``slow`` because proving a hang means waiting one out: this is
        the only test in the suite whose cost is irreducible rather than
        accidental. It still runs by default; ``-m "not slow"`` deselects it for
        a fast inner loop.
        """
        proc, elapsed = self._run(
            """
            import concurrent.futures, time

            inner = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            fut = inner.submit(lambda: time.sleep({abandoned}))
            try:
                fut.result(timeout=0.5)
            except concurrent.futures.TimeoutError:
                print("backstop fired")
            finally:
                inner.shutdown(wait=False, cancel_futures=True)
            print("exiting")
            """
        )

        assert "exiting" in proc.stdout
        assert elapsed >= self.ABANDONED_SECONDS - 2, (
            f"the executor form exited in {elapsed:.1f}s — it used to be held "
            f"open for the full {self.ABANDONED_SECONDS}s by the atexit join"
        )


class TestRunAllWithTimeout:
    """The N-way sibling of run_with_timeout, for pipeline.py's parallel
    review-call fan-out and any other caller that used to hand a whole batch
    to a ``with ThreadPoolExecutor() as executor:`` block."""

    def test_returns_every_value_on_success(self):
        jobs = [("a", lambda: 1, 5), ("b", lambda: 2, 5)]
        results = run_all_with_timeout(jobs, global_timeout=5)
        assert results == {"a": (1, None), "b": (2, None)}

    def test_reraises_what_the_call_raised(self):
        def _boom():
            raise ValueError("provider said no")

        results = run_all_with_timeout([("a", _boom, 5)], global_timeout=5)
        value, error = results["a"]
        assert value is None
        assert isinstance(error, ValueError)
        assert "provider said no" in str(error)

    def test_own_timeout_fires_before_the_global_one(self):
        results = run_all_with_timeout(
            [("slow", lambda: time.sleep(5), 0.2)], global_timeout=10
        )
        value, error = results["slow"]
        assert value is None
        assert isinstance(error, TimeoutError)
        assert "Timed out after 0.2s" in str(error)

    def test_global_timeout_fires_before_an_individual_ones(self):
        """A call whose own budget hasn't run out yet still gets cut off once
        the group deadline arrives — the same shape as pipeline.py's global
        ceiling cancelling calls that are individually still within budget."""
        results = run_all_with_timeout(
            [("slow", lambda: time.sleep(5), 30)], global_timeout=0.2
        )
        value, error = results["slow"]
        assert value is None
        assert isinstance(error, TimeoutError)
        assert "Exceeded global timeout of 0.2s" in str(error)

    def test_a_none_own_timeout_still_respects_the_global_one(self):
        """A model with no calibrated per-call budget must not be able to
        stall the whole batch forever — the group deadline is the backstop."""
        results = run_all_with_timeout(
            [("uncapped", lambda: time.sleep(5), None)], global_timeout=0.2
        )
        value, error = results["uncapped"]
        assert value is None
        assert isinstance(error, TimeoutError)
        assert "Exceeded global timeout of 0.2s" in str(error)

    def test_one_slow_job_does_not_hold_up_the_others(self):
        """The whole point: fast jobs are not blocked waiting on a slow one."""
        finished_order = []

        def _slow():
            time.sleep(2)
            finished_order.append("slow")
            return "slow"

        def _fast():
            finished_order.append("fast")
            return "fast"

        started = time.monotonic()
        results = run_all_with_timeout(
            [("slow", _slow, 0.2), ("fast", _fast, 5)], global_timeout=5
        )
        elapsed = time.monotonic() - started

        assert elapsed < 1.5
        assert results["fast"] == ("fast", None)
        assert isinstance(results["slow"][1], TimeoutError)
        assert finished_order == ["fast"]

    def test_jobs_actually_run_concurrently(self):
        """All four jobs must be in flight at once, not run in a sequential loop.

        A barrier rather than a stopwatch. Every job has to arrive before any of
        them is released, so a sequential implementation cannot satisfy this at
        any speed: the first job would sit waiting for three callers that will
        not run until it returns, and the barrier would break. The previous
        form slept 1s in each of four jobs and asserted the total stayed under
        2s — the same claim, argued from wall-clock ratio, for 1s a run.

        `run_all_with_timeout` starts one daemon thread per job with no pool
        bound (see its implementation), so a four-party barrier is safe here.
        """
        gate = threading.Barrier(4, timeout=5)
        jobs = [(str(i), gate.wait, 5) for i in range(4)]

        results = run_all_with_timeout(jobs, global_timeout=5)

        errors = {name: err for name, (_, err) in results.items() if err is not None}
        assert not errors, (
            f"Jobs did not all reach the barrier together, so they were not "
            f"running concurrently: {errors}"
        )
        # Each waiter gets a distinct arrival index, so this also shows all four
        # really passed through rather than one running four times.
        assert sorted(value for value, _ in results.values()) == [0, 1, 2, 3]


class TestRunAllWithTimeoutProcessExit:
    """Same regression as TestProcessExit, for the N-way form: an abandoned
    job in the group must not hold the interpreter open either."""

    ABANDONED_SECONDS = 20
    MUST_EXIT_WITHIN = 10

    def test_an_abandoned_job_does_not_hold_the_process_open(self):
        script = textwrap.dedent(
            """
            import time
            from ci_core.concurrency import run_all_with_timeout

            results = run_all_with_timeout(
                [("stuck", lambda: time.sleep({abandoned}), 0.5)],
                global_timeout=0.5,
            )
            print("backstop fired" if results["stuck"][1] else "no backstop")
            print("exiting")
            """
        ).format(abandoned=self.ABANDONED_SECONDS)

        t0 = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=self.ABANDONED_SECONDS + 30,
        )
        elapsed = time.monotonic() - t0

        assert proc.returncode == 0, proc.stderr
        assert "backstop fired" in proc.stdout
        assert "exiting" in proc.stdout
        assert elapsed < self.MUST_EXIT_WITHIN, (
            f"process took {elapsed:.1f}s to exit with a {self.ABANDONED_SECONDS}s "
            f"abandoned job still running in the group"
        )
