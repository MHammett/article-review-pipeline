"""Detects when a pre-existing OS environment variable silently shadows a
``.env`` file value.

python-dotenv's ``load_dotenv()`` defaults to ``override=False``: when an OS
environment variable of the same name is already set, the ``.env`` file's
value for it is silently discarded -- no error, no warning, nothing in the
return value distinguishes this from a normal load. A persistent Windows
User-scoped ``OPENAI_API_KEY`` did exactly this for days: every edit to
``.env``'s key was ignored in favor of the stale OS variable, and nothing in
the pipeline's output said so.

Call :func:`snapshot` with the *same* dotenv path ``load_dotenv()`` is about
to use, before it runs -- while the OS environment is still in its
pre-``.env`` state. Everything else in this module reads from that snapshot.
"""

import os

from dotenv import dotenv_values


def snapshot(dotenv_path):
    """Capture pre-``load_dotenv()`` state for the given ``.env`` path.

    ``dotenv_path`` should be the exact path (or a falsy value, if none was
    found) the caller is about to pass to ``load_dotenv()`` -- resolve it
    once with ``dotenv.find_dotenv()`` and reuse that same value for both
    calls, so this snapshot and the actual load never disagree about which
    file, if any, is in play.
    """
    return {
        "dotenv_path": dotenv_path or None,
        "pre_existing_keys": frozenset(os.environ.keys()),
        "file_values": dict(dotenv_values(dotenv_path)) if dotenv_path else {},
    }


def provenance(snap, var_name):
    """Where ``var_name``'s active value came from, evaluated any time after
    ``load_dotenv()`` has run against ``snap``.
    """
    in_dotenv = var_name in snap["file_values"]
    dotenv_value = snap["file_values"].get(var_name)
    shadowed = var_name in snap["pre_existing_keys"]
    active_value = os.environ.get(var_name)
    return {
        "var_name": var_name,
        "active_value": active_value,
        "in_dotenv_file": in_dotenv,
        "dotenv_value": dotenv_value,
        # var_name was already set in the OS environment before load_dotenv()
        # ran, so python-dotenv's override=False left it untouched -- any
        # .env value for this name was never applied.
        "shadowed_by_os_env": shadowed,
        # The dangerous case: a .env entry exists and is being silently
        # ignored in favor of a *different* value nobody is looking at.
        "mismatched": shadowed and in_dotenv and dotenv_value != active_value,
    }


def shadowed_mismatches(snap):
    """Every .env-defined variable whose value differs from what's actually
    active because a pre-existing OS environment variable got there first.

    This is the passive-warning case -- inherently suspicious regardless of
    whether anyone thought to ask for it.
    """
    return [
        provenance(snap, name)
        for name in snap["file_values"]
        if name in snap["pre_existing_keys"]
        and snap["file_values"][name] != os.environ.get(name)
    ]
