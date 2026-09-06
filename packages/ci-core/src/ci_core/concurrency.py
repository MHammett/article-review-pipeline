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

import logging
import os
import sys
import threading
import time

__all__ = [
    "run_with_timeout",
    "run_all_with_timeout",
    "exit_without_waiting_for_foreign_threads",
]


def exit_without_waiting_for_foreign_threads(code=0):
    """Flush, then leave, without waiting on threads nobody can reach.

    Everything above keeps *our* abandoned work off the shutdown path. It cannot
    help with a pool this codebase never creates, and litellm creates one per
    streaming call for its post-call success logging. Those workers are not
    daemons, so ``threading._shutdown()`` joins them and the process cannot
    leave until they return.

    Measured 2026-09-03: a ``ci-review`` run printed its whole summary, wrote
    every output file, then sat at 0% CPU with 19 threads — 18 of them blocked
    in ``accept()`` inside ``socket.socketpair()``, asyncio's Windows self-pipe,
    which had stopped returning. The work was finished; the process could not
    leave. Two of six such processes were found alive two days later.

    ``os._exit`` because nothing gentler reaches it
    ----------------------------------------------
    The obvious trick is to empty ``concurrent.futures.thread._threads_queues``
    so its ``_python_exit`` hook finds nothing to join. Tried, measured, and it
    is *worse*: that same hook is what puts a ``None`` sentinel in each work
    queue to wake the workers. Removed from the registry, they never receive it,
    block on ``work_queue.get()`` forever, and — still not being daemons —
    ``threading._shutdown()`` waits on them without end. A bounded hang becomes
    an unbounded one.

    Nothing else works either, because the thread is not merely idle, it is
    stuck inside a syscall that will not return. It cannot be woken, it cannot
    be killed, and its ``daemon`` flag cannot be changed after ``start()``.
    ``os._exit`` skips interpreter finalisation altogether, which is the only
    thing that does not depend on that thread cooperating.

    Safe here specifically because of *when* it is called: at the end of
    ``main``, after the report, the readable review and the capture have all
    been written and closed. Flushing explicitly first is what makes that true
    for the streams, since ``os._exit`` runs no buffers down.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # pragma: no cover - a closed pipe is not our problem
            pass
    logging.shutdown()
    os._exit(code)


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

    Both budgets are absolute deadlines stamped up front — the group's when this
    function is entered, each call's when that call starts — never durations
    handed out when the join loop arrives. The loop joins sequentially, so a
    duration would restart every straggler's budget from wherever the loop had
    got to: a call late in the list would be waited on for the sum of every
    budget ahead of it plus its own. Measured 2026-09-03 at four grok calls
    running 143-378s against a 120s budget, each recorded ``OK`` and each
    landing within a second of *(the moment its join began) + 120*.

    A call that outlived its budget but finished anyway still returns its value.
    The budget bounds how long this function *waits*; work that is already done
    has been paid for, and throwing it away would buy nothing.
    """
    group_deadline = time.monotonic() + global_timeout

    outcomes = {}
    threads = {}
    per_job_timeout = {}
    job_deadline = {}

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
        # timeout of None means "no per-call budget" (e.g. a model with no
        # calibrated timeout) — the group deadline is still the backstop.
        job_deadline[name] = None if timeout is None else time.monotonic() + timeout
        worker.start()

    results = {}
    for name, worker in threads.items():
        own_deadline = job_deadline[name]
        own_binds = own_deadline is not None and own_deadline <= group_deadline
        limit = own_deadline if own_binds else group_deadline
        worker.join(max(0.0, limit - time.monotonic()))

        outcome = outcomes[name]
        if worker.is_alive():
            if own_binds:
                results[name] = (
                    None,
                    TimeoutError(f"Timed out after {per_job_timeout[name]}s"),
                )
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
