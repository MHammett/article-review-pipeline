"""Console I/O setup shared by every CLI entry point.

Deliberately dependency-free (stdlib ``sys`` only) and written to import on any
Python a user might have lying around: the entry points call this *before* their
own "Python 3.10+ required" guard, so a syntax error here would replace that
friendly message with a traceback.
"""

import sys


def force_utf8_stdio() -> None:
    """Re-encode stdout/stderr as UTF-8. Call once, first thing, at CLI entry.

    Windows is this project's documented primary platform and its consoles
    default to cp1252, which cannot encode a large part of what these tools
    print: the status markers and arrows in the reports themselves, and — the
    half that stripping decorative characters would never fix — article titles,
    flagged passages, and provider error bodies that arrive as data. Printing
    any of it raises ``UnicodeEncodeError`` and kills the command partway
    through its own report, which is how ``ci-discover`` came to die on its
    first provider row.

    ``errors="replace"`` covers the residue UTF-8 cannot represent either —
    chiefly lone surrogates, which reach us from Windows filenames read with
    ``surrogateescape`` and from malformed provider responses. A run should not
    end because one character in a link's title is unpaired.

    Streams that cannot be reconfigured are left alone: ``sys.stdout`` is
    ``None`` under ``pythonw.exe``, and test harnesses substitute objects of
    their own. Each stream is checked separately because they are replaced
    independently.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            # A detached or already-closed stream. Nothing to fix, and failing
            # here would take down a CLI over its own output settings.
            pass
