"""Readability metrics — Flesch-Kincaid and supporting stats. No external dependencies."""
import re


def _count_syllables(word):
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in "aeiouy"
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _split_sentences(text):
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _split_words(text):
    return re.findall(r"\b[a-zA-Z']+\b", text)


def analyze(text):
    """Return readability metrics dict for the given text."""
    sentences = _split_sentences(text)
    words = _split_words(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    n_sentences = max(len(sentences), 1)
    n_words = max(len(words), 1)
    n_syllables = sum(_count_syllables(w) for w in words)
    n_paragraphs = max(len(paragraphs), 1)

    avg_sentence_length = round(n_words / n_sentences, 1)
    avg_syllables_per_word = round(n_syllables / n_words, 2)

    # Flesch Reading Ease — higher is easier to read
    fre = round(206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words), 1)
    fre = max(0.0, min(100.0, fre))

    # Flesch-Kincaid Grade Level
    fkgl = round(0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59, 1)
    fkgl = max(0.0, fkgl)

    reading_level = (
        "Very Easy" if fre >= 90 else
        "Easy" if fre >= 80 else
        "Fairly Easy" if fre >= 70 else
        "Standard" if fre >= 60 else
        "Fairly Difficult" if fre >= 50 else
        "Difficult" if fre >= 30 else
        "Very Confusing"
    )

    longest_para_words = max(
        (len(_split_words(p)) for p in paragraphs),
        default=0,
    )

    return {
        "word_count": len(words),
        "sentence_count": n_sentences,
        "paragraph_count": n_paragraphs,
        "avg_sentence_length": avg_sentence_length,
        "avg_syllables_per_word": avg_syllables_per_word,
        "flesch_reading_ease": fre,
        "flesch_kincaid_grade": fkgl,
        "reading_level": reading_level,
        "longest_paragraph_words": longest_para_words,
    }
