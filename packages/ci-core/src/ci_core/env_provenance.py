"""Resolves the precedence between a ``.env`` file value and a same-named OS
environment variable, and detects when they disagree.

python-dotenv's ``load_dotenv()`` defaults to ``override=False``: when an OS
environment variable of the same name is already set, the ``.env`` file's
value for it is silently discarded at the ``os.environ`` level -- no error, no
warning, nothing in the return value distinguishes this from a normal load. A
persistent Windows User-scoped ``OPENAI_API_KEY`` did exactly this for days:
every edit to ``.env``'s key had no effect while the pipeline kept billing the
stale OS-level key, and nothing said so.

The project's precedence, most to least specific, is: CLI override >
publication config file > ``.env`` file > bare OS environment variable. This
module's :func:`provenance` computes the *effective* value under that
precedence directly -- callers doing ``${VAR}`` resolution should prefer a
``.env``-defined value over ``os.environ`` themselves (see
``ci_core.config_helpers.resolve_env``'s ``env`` parameter and
``ci_article_review.config_loader``'s ``_EFFECTIVE_ENV``); this module exists
to make that precedence inspectable and to flag the two sources disagreeing,
which is worth a heads-up even though the outcome is no longer ambiguous.

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


def effective_env(snap):
    """The OS environment, with every ``.env``-defined variable overriding
    its same-named OS entry -- the mapping ``${VAR}`` resolution should read
    from, so a ``.env`` edit always takes effect regardless of what else is
    set in the shell. A variable ``.env`` doesn't mention falls through to
    the OS environment unchanged; that's still the lowest-priority source,
    not an ignored one.
    """
    return {**os.environ, **snap["file_values"]}


def provenance(snap, var_name):
    """Where ``var_name``'s effective value comes from under this project's
    precedence, evaluated any time after ``load_dotenv()`` has run against
    ``snap``. ``active_value`` is the value that will actually be used by
    ``${VAR}`` resolution (see module docstring) -- a ``.env`` entry always
    wins over a same-named OS variable, not the reverse.
    """
    in_dotenv = var_name in snap["file_values"]
    dotenv_value = snap["file_values"].get(var_name)
    os_value = os.environ.get(var_name)
    shadowed = var_name in snap["pre_existing_keys"]
    return {
        "var_name": var_name,
        "active_value": dotenv_value if in_dotenv else os_value,
        "in_dotenv_file": in_dotenv,
        "dotenv_value": dotenv_value,
        "os_env_value": os_value,
        # var_name was already set in the OS environment before load_dotenv()
        # ran. Under override=False that would have silently discarded a
        # differing .env value at the os.environ level -- this module (and
        # config_loader's ${VAR} resolution) route around that by preferring
        # snap["file_values"] directly, so this flag is informational, not a
        # warning about which value is "really" in effect.
        "shadowed_by_os_env": shadowed,
        # No longer the dangerous case it once was -- .env always wins now
        # when it defines the variable. Still worth surfacing: an OS-level
        # value that differs is being ignored, which the person who set it
        # may not expect.
        "mismatched": shadowed and in_dotenv and dotenv_value != os_value,
    }


def shadowed_mismatches(snap):
    """Every .env-defined variable whose value differs from a same-named,
    pre-existing OS environment variable.

    The .env value wins under this project's precedence (see module
    docstring), so this is no longer the dangerous silent-override case it
    once was -- but two config sources disagreeing is still worth a heads-up,
    so callers can confirm the winner is the one they intended.
    """
    return [
        provenance(snap, name)
        for name in snap["file_values"]
        if name in snap["pre_existing_keys"]
        and snap["file_values"][name] != os.environ.get(name)
    ]
