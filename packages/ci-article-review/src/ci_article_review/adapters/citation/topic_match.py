"""Shared keyword-gating helper for pointer-only citation adapters.

Pointer-only adapters (EPA, FHWA, PJM, ICC, ILGA, FERC) gate on whether a
claim's text contains any of a category's keywords, via plain substring
containment. That's not enough on its own: a keyword phrase can appear in a
claim that isn't actually *about* that regulatory topic — e.g. "He does not
hold credentials in environmental engineering or air quality analysis" is a
claim about a person's academic background, but it genuinely contains the
word-boundary phrase "air quality" (inside "air quality analysis"), so raw
containment resolves it to the EPA Air Quality System portal.

``topic_match`` filters out keyword hits that occur in the same sentence as
a qualification/credential phrase ("credentials in", "degree in", "expertise
in", etc.) — those mark the keyword as describing a person's background, not
the claim's actual subject. This is deliberately conservative: it operates
at sentence granularity (not just immediate adjacency) so constructions like
"credentials in X or Y" still catch the tail item of the list, and it
suppresses the whole sentence rather than trying to prove the negative. The
goal, per this adapter family's accuracy requirements, is to err toward not
resolving a claim rather than resolving it to the wrong topic.
"""

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_DISQUALIFYING_PHRASES = (
    "credentials in",
    "credential in",
    "expertise in",
    "background in",
    "degree in",
    "degrees in",
    "doctorate in",
    "phd in",
    "ph.d. in",
    "training in",
    "certification in",
    "certifications in",
    "certified in",
    "qualifications in",
    "qualification in",
    "experience in",
    "specializes in",
    "specialize in",
    "specialty in",
    "major in",
    "minor in",
    "coursework in",
)


def topic_match(claim_lower, keywords):
    """Return the first keyword from ``keywords`` that genuinely appears in
    ``claim_lower``, or ``None``.

    A keyword hit is discarded if the sentence it occurs in also contains a
    disqualifying credential/qualification phrase — see module docstring.
    ``claim_lower`` may be pre-padded with spaces by the caller (some
    adapters do this for their own word-boundary checks); that's harmless
    here since sentences are split on terminal punctuation, not padding.
    """
    sentences = _SENTENCE_SPLIT_RE.split(claim_lower)
    for kw in keywords:
        for sentence in sentences:
            if kw not in sentence:
                continue
            if any(p in sentence for p in _DISQUALIFYING_PHRASES):
                continue
            return kw
    return None
