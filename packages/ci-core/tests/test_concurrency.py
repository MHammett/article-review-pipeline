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

from ci_core.concurrency import run_with_timeout


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
    ABANDONED_SECONDS = 20
    MUST_EXIT_WITHIN = 10

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

    def test_the_executor_form_really_did_hang(self):
        """Pins the diagnosis, so the fix cannot be argued away later.

        If this ever stops hanging, CPython changed its atexit behaviour and the
        comment in ci_core.concurrency explaining the fix needs revisiting.
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
