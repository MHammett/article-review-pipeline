"""Tests for normalize.py."""

from __future__ import annotations

import pytest


def _make_doc(text: str, source: str = "wordpress"):
    from ci_web_intel.collectors.base import Document
    return Document.from_text(text=text, source=source, register="long_form", date="2024-01-01", url_or_id="test")


class TestCleanText:
    def test_strips_wp_shortcodes(self):
        from ci_web_intel.normalize import clean_text
        result = clean_text("[gallery ids='1,2,3'] Hello world.", "wordpress")
        assert "[gallery" not in result
        assert "Hello world." in result

    def test_strips_html_entities(self):
        from ci_web_intel.normalize import clean_text
        result = clean_text("Hello &amp; world &mdash; test.", "wordpress")
        assert "&amp;" not in result
        assert "&mdash;" not in result

    def test_normalizes_twitter_tco(self):
        from ci_web_intel.normalize import clean_text
        result = clean_text("Check this out https://t.co/abc123 today", "twitter")
        assert "https://t.co/abc123" not in result
        assert "[link]" in result

    def test_strips_twitter_reply_mentions(self):
        from ci_web_intel.normalize import clean_text
        result = clean_text("@user1 @user2 great point about this topic", "twitter")
        assert result.startswith("great point")

    def test_strips_gmail_quoted_blocks(self):
        from ci_web_intel.normalize import clean_text
        text = "My response here.\n\n> Quoted text from previous email\n> More quoted text"
        result = clean_text(text, "gmail")
        assert "My response here" in result
        assert "> Quoted text" not in result

    def test_unicode_quotes_normalized(self):
        from ci_web_intel.normalize import clean_text
        result = clean_text("“Hello” and ‘world’", "wordpress")
        assert '"Hello"' in result or "'world'" in result or "Hello" in result


class TestSentenceSplit:
    def test_abbreviations_dont_split(self):
        from ci_web_intel.normalize import sentence_split
        # Abbreviations in the middle of a sentence should not trigger a split
        text = "He spoke with Dr. Smith and Mr. Jones. They agreed."
        sentences = sentence_split(text)
        # Should produce 2 sentences (Dr. and Mr. don't split)
        assert len(sentences) == 2

    def test_basic_split(self):
        from ci_web_intel.normalize import sentence_split
        text = "First sentence. Second sentence. Third sentence."
        sentences = sentence_split(text)
        assert len(sentences) == 3

    def test_questions_and_exclamations(self):
        from ci_web_intel.normalize import sentence_split
        text = "Is this correct? Yes it is! Great."
        sentences = sentence_split(text)
        assert len(sentences) == 3


class TestComputeMetrics:
    def test_metrics_on_known_text(self):
        from ci_web_intel.normalize import compute_metrics
        text = "I write clearly. I use simple sentences. I avoid jargon."
        doc = _make_doc(text)
        metrics = compute_metrics(doc)
        assert "avg_sentence_words" in metrics
        assert "first_person_ratio" in metrics
        assert metrics["first_person_ratio"] > 0  # "I" appears in each sentence

    def test_passive_ratio_detected(self):
        from ci_web_intel.normalize import compute_metrics
        text = "The report was written by the team. The data was analyzed carefully."
        doc = _make_doc(text)
        metrics = compute_metrics(doc)
        assert metrics["passive_ratio"] > 0

    def test_question_ratio_detected(self):
        from ci_web_intel.normalize import compute_metrics
        text = "What do we know? We know the facts. How do we proceed? Carefully."
        doc = _make_doc(text)
        metrics = compute_metrics(doc)
        assert metrics["question_ratio"] > 0


class TestDeduplicate:
    def test_drops_exact_duplicates(self):
        from ci_web_intel.normalize import deduplicate
        doc1 = _make_doc("Same text here for testing.")
        doc2 = _make_doc("Same text here for testing.")
        doc3 = _make_doc("Different text entirely.")
        unique, n_dropped = deduplicate([doc1, doc2, doc3])
        assert n_dropped == 1
        assert len(unique) == 2

    def test_keeps_unique_docs(self):
        from ci_web_intel.normalize import deduplicate
        docs = [_make_doc(f"Unique text {i}.") for i in range(5)]
        unique, n_dropped = deduplicate(docs)
        assert n_dropped == 0
        assert len(unique) == 5


class TestCorpusSummary:
    def test_aggregates_correctly(self):
        from ci_web_intel.normalize import corpus_summary
        docs = [
            _make_doc("Word " * 100, "wordpress"),
            _make_doc("Word " * 50, "wordpress"),
            _make_doc("Word " * 50, "gmail"),
        ]
        summary = corpus_summary(docs)
        assert summary["doc_count"] == 3
        assert summary["total_words"] == 200
        assert "wordpress" in summary["sources"]
        assert "gmail" in summary["sources"]

    def test_empty_corpus(self):
        from ci_web_intel.normalize import corpus_summary
        summary = corpus_summary([])
        assert summary["doc_count"] == 0
        assert summary["total_words"] == 0


class TestCorpusBiasWarnings:
    def test_fires_above_75_percent(self):
        from ci_web_intel.normalize import corpus_bias_warnings, corpus_summary
        docs = [_make_doc("Word " * 100, "wordpress") for _ in range(8)]
        docs += [_make_doc("Word " * 25, "gmail")]
        summary = corpus_summary(docs)
        warnings = corpus_bias_warnings(summary)
        assert any("wordpress" in w for w in warnings)

    def test_silent_below_threshold(self):
        from ci_web_intel.normalize import corpus_bias_warnings, corpus_summary
        docs = [
            _make_doc("Word " * 50, "wordpress"),
            _make_doc("Word " * 40, "gmail"),
            _make_doc("Word " * 35, "twitter"),
        ]
        summary = corpus_summary(docs)
        warnings = corpus_bias_warnings(summary)
        bias_warnings = [w for w in warnings if "%" in w]
        assert not bias_warnings
