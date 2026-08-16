"""Running work under a wall-clock backstop.

Both applications need the same thing: run one provider call, and stop *waiting*
on it after N seconds. A running thread cannot be killed, so "stop waiting" is
all that is on offer — the call itself carries on until its own socket timeouts
fire. That is fine, and is the whole design of the backstop.

What is not fine is what the abandoned thread then does to process exit.

Why not a ThreadPoolExecutor
----------------------------
Both callers used to spin up a single-worker ``ThreadPoolExecutor`` and shut it
down with ``wait=False``. That abandons the call correctly, but
``concurrent.futures.thread`` registers ``_python_exit`` through
``threading._register_atexit``, and that hook joins every worker thread with a
bare ``t.join()`` — no timeout. So an abandoned worker does not merely delay
interpreter exit, it blocks it for exactly as long as the abandoned call runs.

Measured 2026-08-16, and this is not theoretical: six ``ci-review`` processes
were found still alive, two of them **two days** after their run had written its
report and printed "REVIEW COMPLETE". Each held a file handle on its own log,
which is also what made ``git worktree remove`` fail. Killing them made three
waiting shells return instantly — the work had been done all along.

The code this replaces said the delay was "bounded by the adapter's read-gap
timeout, and the alternative (daemonizing the pool's threads) means reaching
into ThreadPoolExecutor internals". Two days is not bounded by any read-gap
timeout in the config, and no internals are needed: a plain daemon thread is
both the smaller mechanism and the correct one. The executor was only ever being
used for its ``result(timeout=...)``, which ``Thread.join(timeout)`` provides
directly.
"""

import threading

__all__ = ["run_with_timeout"]


def run_with_timeout(fn, timeout):
    """Call ``fn()`` and return its result, or raise ``TimeoutError``.

    The budget applies to this call alone, not to its position in any queue.

    On expiry the worker is abandoned rather than killed: it keeps running until
    the call it is making ends on its own. Because the thread is a daemon, that
    abandoned work can never hold the interpreter open — the process exits when
    its real work is done, and the orphan goes with it.

    Anything ``fn`` raises is re-raised here, so callers see provider failures
    exactly as if the call had been made inline.
    """
    outcome = {}

    def _target():
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — relayed to the caller
            outcome["error"] = exc

    worker = threading.Thread(target=_target, name="ci-backstopped-call", daemon=True)
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        raise TimeoutError(f"Timed out after {timeout}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")
