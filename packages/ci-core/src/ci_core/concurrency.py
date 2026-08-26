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
import time

__all__ = ["run_with_timeout", "run_all_with_timeout"]


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


def run_all_with_timeout(jobs, global_timeout):
    """Run every ``(name, fn, timeout)`` in ``jobs`` concurrently, one daemon
    thread each — never a ``ThreadPoolExecutor``, for the reason in this
    module's docstring.

    Two budgets apply to each call: its own ``timeout``, and the shared
    ``global_timeout`` for the whole group, measured from when this function
    is entered. Whichever is tighter governs how long this function waits for
    that particular call — the same abandon-rather-than-wait contract as
    :func:`run_with_timeout`, just applied to many calls sharing one deadline
    instead of one call on its own. A straggler still running when its budget
    runs out is left running; because every thread here is a daemon, none of
    them can ever hold the interpreter open, no matter how long they take —
    that guarantee is what a ``ThreadPoolExecutor`` cannot make (its
    ``__exit__`` — and the atexit hook it registers even without ``with`` —
    both join every worker with a bare, untimed ``t.join()``).

    Returns ``{name: (value, None)}`` for a call that finished in time and
    raised nothing, or ``{name: (None, exc)}`` otherwise — ``exc`` is whatever
    ``fn`` raised, or ``TimeoutError`` (message says which budget hit first)
    if neither finished before its deadline.
    """
    outcomes = {}
    threads = {}
    per_job_timeout = {}

    for name, fn, timeout in jobs:
        outcome = {}

        def _target(fn=fn, outcome=outcome):
            try:
                outcome["value"] = fn()
            except BaseException as exc:  # noqa: BLE001 — relayed to the caller
                outcome["error"] = exc

        worker = threading.Thread(
            target=_target, name=f"ci-group-call-{name}", daemon=True
        )
        outcomes[name] = outcome
        threads[name] = worker
        per_job_timeout[name] = timeout
        worker.start()

    deadline = time.monotonic() + global_timeout
    results = {}
    for name, worker in threads.items():
        own_timeout = per_job_timeout[name]
        remaining = max(0.0, deadline - time.monotonic())
        # own_timeout of None means "no per-call budget" (e.g. a model with no
        # calibrated timeout) — the group deadline is still the backstop.
        own_binds = own_timeout is not None and own_timeout <= remaining
        worker.join(own_timeout if own_binds else remaining)

        outcome = outcomes[name]
        if worker.is_alive():
            if own_binds:
                results[name] = (None, TimeoutError(f"Timed out after {own_timeout}s"))
            else:
                results[name] = (
                    None,
                    TimeoutError(f"Exceeded global timeout of {global_timeout}s"),
                )
        elif "error" in outcome:
            results[name] = (None, outcome["error"])
        else:
            results[name] = (outcome.get("value"), None)
    return results
