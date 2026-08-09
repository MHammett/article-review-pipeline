"""Tests for the shared keyword-gating helper used by pointer-only citation
adapters (EPA, FHWA, PJM, ICC, ILGA, FERC)."""

from ci_article_review.adapters.citation.topic_match import topic_match


def test_matches_genuine_topical_mention():
    assert topic_match("the facility exceeded naaqs thresholds.", ["naaqs"]) == "naaqs"


def test_no_match_when_keyword_absent():
    assert topic_match("the county approved the tax levy.", ["naaqs"]) is None


def test_rejects_keyword_in_credentials_clause():
    claim = (
        "cork holds a doctorate in statistics. he does not hold credentials "
        "in environmental engineering or air quality analysis."
    )
    assert topic_match(claim, ["air quality"]) is None


def test_accepts_keyword_in_unrelated_sentence_of_same_claim():
    """A credentials clause in one sentence shouldn't suppress a genuine
    topical mention in a different sentence of the same claim."""
    claim = (
        "she holds a degree in journalism. separately, pm2.5 levels in the "
        "county exceeded federal limits last month."
    )
    assert topic_match(claim, ["pm2.5"]) == "pm2.5"


def test_returns_first_keyword_in_priority_order():
    claim = "new pfas limits were announced alongside naaqs updates."
    assert topic_match(claim, ["pfas", "naaqs"]) == "pfas"
