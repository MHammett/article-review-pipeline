r"""Repair provider text whose punctuation arrived narrowed to control bytes.

Perplexity returns prose in which characters from the Unicode General
Punctuation block (U+2000-U+201F) have been narrowed to their low byte, so the
codepoint lands in the C0 control range instead. The mapping is exactly
``cp & 0xFF``, and two cases pin it down against text the same run recorded
intact elsewhere:

    "Lawrence Berkeley National Laboratory\x19s"  <- U+2019 RIGHT SINGLE QUOTE
    "counterexample\x14a published piece"         <- U+2014 EM DASH

Left alone this reaches three places that matter: the fact-check ``claim`` and
``note`` fields a human reads, ``run_N_*_review.md`` (which the documented
workflow tells the author to paste into a chat model), and the next review
round's prompt. A raw C0 byte in prose is never legitimate, so repairing it
cannot be worse than passing it through.

What each range becomes, and why:

* ``0x10-0x1F`` -> ``U+2010-U+201F``. Dashes and quotation marks -- visible
  punctuation with a real meaning in prose. This is the evidence-backed case.
* ``0x00-0x08`` -> a plain space. ``U+2000-U+2008`` are all space variants
  (en quad, em quad, figure space...); collapsing them to ASCII space keeps the
  word break without inventing a specific width.
* ``0x0B, 0x0C, 0x0E, 0x0F`` -> dropped. ``U+200B/C/E/F`` are zero-width and
  bidi format controls, invisible by definition, so removing them restores the
  text as the narrowing found it rather than deleting content. This cannot be
  told apart from a character lost even further upstream; what is certain is
  that we cannot recover one, and a bare control byte is not a better guess.
* Tab, newline and carriage return are untouched. They are real whitespace.

Only the block that lands in C0 is detectable at all. The same narrowing
applied to, say, U+4E2D yields ``0x2D`` ("-"), indistinguishable from a hyphen
the model meant to write. This repairs what can be identified; it does not
claim to make a narrowed response whole.
"""

import re

#: Built once. Keys are the C0 code points we rewrite; anything absent (tab,
#: newline, carriage return) passes through untouched. A ``None`` value is
#: str.translate's spelling of "delete this character".
_REPAIR: dict[int, str | None] = {}
for _c in range(0x00, 0x20):
    if _c in (0x09, 0x0A, 0x0D):  # real whitespace, not damage
        continue
    elif 0x10 <= _c <= 0x1F:  # dashes and quotes: restore the punctuation
        _REPAIR[_c] = chr(0x2000 + _c)
    elif _c in (0x0B, 0x0C, 0x0E, 0x0F):  # zero-width / bidi: invisible anyway
        _REPAIR[_c] = None
    else:  # 0x00-0x08 -> U+2000-U+2008, all space variants
        _REPAIR[_c] = " "
del _c

#: Derived from ``_REPAIR`` so the two can never drift apart. Lets a clean
#: response -- which is nearly all of them -- be returned without copying:
#: bodies here run to hundreds of kilobytes and this sees every one.
_DAMAGED = re.compile("[" + "".join(re.escape(chr(c)) for c in _REPAIR) + "]")


def repair_narrowed_punctuation(text):
    """Undo low-byte narrowing of General Punctuation in ``text``.

    Returns ``text`` itself when it holds none of the affected control
    characters, which is the overwhelmingly common case.
    """
    if not text or not _DAMAGED.search(text):
        return text
    return text.translate(_REPAIR)


def repair_tree(value):
    """Apply :func:`repair_narrowed_punctuation` to every string in ``value``.

    Walks dicts, lists and tuples so a response's nested payload -- Perplexity's
    ``search_results`` snippets as much as its answer text -- is repaired by the
    same pass. Values of any other type (usage objects, numbers, None) are
    returned as they are.
    """
    if isinstance(value, str):
        return repair_narrowed_punctuation(value)
    if isinstance(value, dict):
        return {k: repair_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [repair_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(repair_tree(v) for v in value)
    return value
