"""Corpus cleaning, metrics, and deduplication."""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from .collectors.base import Document

log = logging.getLogger(__name__)

_ABBREVS = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|i\.e|e\.g|U\.S|U\.K|Rev|Gen|Col|Lt|Sgt|Cpl|St)\.")
_SENTENCE_END = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+")

# WP shortcode pattern
_WP_SHORTCODE = re.compile(r"\[[a-zA-Z_-]+[^\]]*\]")
# HTML entity cleanup
_HTML_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")
# Twitter t.co links
_TCO_LINK = re.compile(r"https://t\.co/\S+")
# Gmail quoted blocks
_GMAIL_QUOTE = re.compile(r"^>.*", re.MULTILINE)
_GMAIL_ON_WROTE = re.compile(r"\nOn .+wrote:\s*\n", re.DOTALL)
# Signature separator
_SIG_SEP = re.compile(r"\n--\s*\n")
# Unicode smart quotes/dashes → ASCII
_UNICODE_SUBS = [
    ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
    ("–", "-"), ("—", "--"), ("…", "..."),
]

_HEDGING = re.compile(r"\b(may|might|perhaps|could be argued|arguably|possibly|it is possible)\b", re.I)
_PASSIVE = re.compile(r"\b(was|were|is|are|been)\s+\w+ed\b", re.I)
_FIRST_PERSON = re.compile(r"\b(I|me|my|we|our|us|myself|ourselves)\b")
_QUESTION_END = re.compile(r"\?\s*$")


def clean_text(raw: str, source: str) -> str:
    text = raw
    for old, new in _UNICODE_SUBS:
        text = text.replace(old, new)

    if source == "wordpress":
        text = _WP_SHORTCODE.sub("", text)
        text = _HTML_ENTITY.sub(" ", text)
    elif source == "twitter":
        text = _TCO_LINK.sub("[link]", text)
        text = re.sub(r"^(@\w+\s+)+", "", text)
    elif source in ("gmail", "outlook365"):
        text = _GMAIL_QUOTE.sub("", text)
        text = _GMAIL_ON_WROTE.sub("\n", text)
        text = _SIG_SEP.split(text)[0]

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentence_split(text: str) -> list[str]:
    # Protect known abbreviations by temporarily replacing the dot
    protected = _ABBREVS.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    sentences = _SENTENCE_END.split(protected)
    return [s.replace("\x00", ".").strip() for s in sentences if s.strip()]


def compute_metrics(doc: Document) -> dict:
    text = doc.text
    if not text:
        return {}

    sentences = sentence_split(text)
    if not sentences:
        return {}

    words_per_sentence = [len(s.split()) for s in sentences]
    n = len(words_per_sentence)

    avg_sentence_words = sum(words_per_sentence) / n
    sorted_wps = sorted(words_per_sentence)
    mid = n // 2
    median_sentence_words = sorted_wps[mid] if n % 2 else (sorted_wps[mid - 1] + sorted_wps[mid]) / 2
    p90_idx = int(n * 0.9)
    p90_sentence_words = sorted_wps[min(p90_idx, n - 1)]

    passive_count = sum(1 for s in sentences if _PASSIVE.search(s))
    passive_ratio = passive_count / n

    fp_count = sum(1 for s in sentences if _FIRST_PERSON.search(s))
    first_person_ratio = fp_count / n

    hedge_count = sum(1 for s in sentences if _HEDGING.search(s))
    hedging_ratio = hedge_count / n

    q_count = sum(1 for s in sentences if _QUESTION_END.search(s))
    question_ratio = q_count / n

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        para_sentence_counts = []
        for para in paragraphs:
            para_sents = sentence_split(para)
            para_sentence_counts.append(len(para_sents))
        avg_paragraph_sentences = sum(para_sentence_counts) / len(para_sentence_counts)
    else:
        avg_paragraph_sentences = n

    all_words = re.findall(r"\b\w+\b", text.lower())
    if all_words:
        vocab_richness = len(set(all_words)) / len(all_words)
    else:
        vocab_richness = 0.0

    # Readability metrics from analysis module
    fk_grade = None
    fk_ease = None
    try:
        from analysis.readability import analyze as readability_analyze
        ra = readability_analyze(text)
        fk_grade = ra.get("flesch_kincaid_grade")
        fk_ease = ra.get("flesch_reading_ease")
    except Exception:
        pass

    metrics = {
        "avg_sentence_words": round(avg_sentence_words, 2),
        "median_sentence_words": round(median_sentence_words, 2),
        "p90_sentence_words": round(p90_sentence_words, 2),
        "passive_ratio": round(passive_ratio, 4),
        "first_person_ratio": round(first_person_ratio, 4),
        "hedging_ratio": round(hedging_ratio, 4),
        "question_ratio": round(question_ratio, 4),
        "avg_paragraph_sentences": round(avg_paragraph_sentences, 2),
        "vocab_richness": round(vocab_richness, 4),
    }
    if fk_grade is not None:
        metrics["flesch_kincaid_grade"] = round(fk_grade, 2)
    if fk_ease is not None:
        metrics["flesch_reading_ease"] = round(fk_ease, 2)
    return metrics


def deduplicate(docs: list[Document]) -> tuple[list[Document], int]:
    seen: dict[str, Document] = {}
    duplicates_by_source: dict[str, int] = defaultdict(int)

    for doc in docs:
        if doc.content_hash in seen:
            duplicates_by_source[doc.source] += 1
        else:
            seen[doc.content_hash] = doc

    n_dropped = len(docs) - len(seen)
    if n_dropped:
        summary = ", ".join(f"{src}:{n}" for src, n in duplicates_by_source.items())
        log.info("Deduplication: dropped %d duplicates (%s)", n_dropped, summary)

    return list(seen.values()), n_dropped


def corpus_summary(docs: list[Document]) -> dict:
    if not docs:
        return {"total_words": 0, "doc_count": 0, "date_range": None, "sources": {}, "source_word_pct": {}}

    total_words = sum(d.word_count for d in docs)
    dates = [d.date for d in docs if d.date]
    date_range = (min(dates), max(dates)) if dates else None

    per_source: dict[str, dict] = defaultdict(lambda: {"doc_count": 0, "word_count": 0})
    for doc in docs:
        per_source[doc.source]["doc_count"] += 1
        per_source[doc.source]["word_count"] += doc.word_count

    source_word_pct = {}
    if total_words:
        for src, stats in per_source.items():
            source_word_pct[src] = round(stats["word_count"] / total_words * 100, 1)

    return {
        "total_words": total_words,
        "doc_count": len(docs),
        "date_range": date_range,
        "sources": dict(per_source),
        "source_word_pct": source_word_pct,
    }


def corpus_bias_warnings(summary: dict) -> list[str]:
    warnings = []
    source_pct = summary.get("source_word_pct", {})
    for src, pct in source_pct.items():
        if pct > 75:
            warnings.append(
                f"WARNING: {src!r} accounts for {pct:.1f}% of total word count. "
                "The synthesized profile will be strongly biased toward this source. "
                "Consider adding more content from other sources."
            )
    total_words = summary.get("total_words", 0)
    if total_words < 1000:
        warnings.append(
            f"ERROR: Total corpus has only {total_words} words. Minimum is 1,000. "
            "Add more content before running synthesis."
        )
    elif total_words < 5000:
        long_form_words = 0
        for src, stats in summary.get("sources", {}).items():
            if src in ("wordpress", "textfiles"):
                long_form_words += stats.get("word_count", 0)
        if long_form_words < 5000:
            warnings.append(
                f"WARNING: Long-form content ({long_form_words} words) is below the recommended "
                "5,000-word minimum. Voice profile accuracy may be limited."
            )
    return warnings
