"""Tests for analysis.readability."""

from ci_article_review.analysis.readability import analyze, _count_syllables


class TestCountSyllables:
    def test_monosyllabic(self):
        assert _count_syllables("cat") == 1

    def test_two_syllables(self):
        assert _count_syllables("hello") == 2

    def test_silent_e_reduces_count(self):
        # "make" → ma-ke → silent e removes one count
        assert _count_syllables("make") == 1

    def test_empty_returns_zero(self):
        assert _count_syllables("") == 0

    def test_minimum_one(self):
        # Short words with silent-e correction shouldn't go below 1
        assert _count_syllables("the") >= 1


class TestReadabilityAnalyze:
    _SIMPLE = "The cat sat on the mat. It was a fat cat. The hat was flat."
    _COMPLEX = (
        "The extraordinary metamorphosis of epistemological frameworks "
        "necessitates a comprehensive reconfiguration of philosophical paradigms "
        "within the contemporary academic discourse."
    )

    def test_returns_all_keys(self):
        result = analyze(self._SIMPLE)
        for key in (
            "word_count",
            "sentence_count",
            "paragraph_count",
            "avg_sentence_length",
            "avg_syllables_per_word",
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "reading_level",
            "longest_paragraph_words",
        ):
            assert key in result, f"missing key: {key}"

    def test_word_count(self):
        result = analyze(self._SIMPLE)
        assert result["word_count"] == 15

    def test_simple_text_high_fre(self):
        result = analyze(self._SIMPLE)
        assert result["flesch_reading_ease"] >= 70, "simple text should be easy to read"

    def test_complex_text_lower_fre(self):
        simple = analyze(self._SIMPLE)
        complex_ = analyze(self._COMPLEX)
        assert complex_["flesch_reading_ease"] < simple["flesch_reading_ease"]

    def test_reading_level_string_present(self):
        result = analyze(self._SIMPLE)
        assert result["reading_level"] in (
            "Very Easy",
            "Easy",
            "Fairly Easy",
            "Standard",
            "Fairly Difficult",
            "Difficult",
            "Very Confusing",
        )

    def test_grade_level_nonnegative(self):
        assert analyze(self._SIMPLE)["flesch_kincaid_grade"] >= 0

    def test_paragraph_count(self):
        two_para = "First paragraph here.\n\nSecond paragraph here."
        result = analyze(two_para)
        assert result["paragraph_count"] == 2

    def test_empty_text_no_crash(self):
        result = analyze("")
        assert result["word_count"] == 0
