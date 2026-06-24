"""
Parse Template A (draft submission) and Template C (publication handoff) documents.
"""

import re
import logging

log = logging.getLogger(__name__)


def _extract_section(text, header, next_headers=None):
    """Extract content between a header and the next known header."""
    if next_headers:
        boundary = (
            r"(?=\n(?:" + "|".join(re.escape(h) for h in next_headers) + r")\s*\n)"
        )
    else:
        boundary = r"\Z"
    pattern = rf"^{re.escape(header)}\s*\n(.*?){boundary}"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


DRAFT_HEADERS = [
    "DRAFT SUBMISSION HANDOFF",
    "PRIMARY CLAIM",
    "TARGET AUDIENCE",
    "PRE-DRAFT ANALYSIS SUMMARY",
    "SOURCES ALREADY CITED",
    "UNCERTAIN SECTIONS",
    "KNOWN GAPS",
    "ADDITIONAL CONTEXT FOR REVIEW MODELS",
    "DRAFT",
]

PUB_HEADERS = [
    "PUBLICATION HANDOFF",
    "PUBLICATION PARAMETERS",
    "SEO METADATA",
    "EMBEDS AND SPECIAL ELEMENTS",
    "DISPOSITION LOG",
    "FINAL DRAFT",
]


def parse_draft_submission(text):
    def section(header):
        idx = DRAFT_HEADERS.index(header)
        next_h = DRAFT_HEADERS[idx + 1 :] if idx + 1 < len(DRAFT_HEADERS) else []
        return _extract_section(text, header, next_h or None)

    title = _extract_field(text, "Article:")
    publication = _extract_field(text, "Publication:")
    run_number = _extract_field(text, "Pipeline run:")

    # Fields that directly fill prompt template variables — warn if missing
    # so the user knows before a model call produces oddly generic output.
    _REQUIRED_FIELDS = {
        "title": ("Article:", title),
        "primary_claim": ("PRIMARY CLAIM", section("PRIMARY CLAIM")),
        "draft": ("DRAFT", section("DRAFT")),
    }
    for field, (label, value) in _REQUIRED_FIELDS.items():
        if not value:
            log.warning(
                f"Handoff document is missing '{label}'. "
                f"The review models will receive an empty {field} — results may be generic or misdirected."
            )

    # Advisory fields — missing is common and acceptable, but worth noting at debug level
    _ADVISORY_FIELDS = {
        "pre_draft_analysis": (
            "PRE-DRAFT ANALYSIS SUMMARY",
            section("PRE-DRAFT ANALYSIS SUMMARY"),
        ),
    }

    results = {
        "title": title,
        "publication": publication,
        "run_number": int(run_number) if run_number and run_number.isdigit() else 1,
        "primary_claim": _REQUIRED_FIELDS["primary_claim"][1],
        "target_audience": section("TARGET AUDIENCE"),
        "pre_draft_analysis": _ADVISORY_FIELDS["pre_draft_analysis"][1],
        "sources_cited": section("SOURCES ALREADY CITED"),
        "uncertain_sections": section("UNCERTAIN SECTIONS"),
        "known_gaps": section("KNOWN GAPS"),
        "additional_context": section("ADDITIONAL CONTEXT FOR REVIEW MODELS"),
        "draft": _REQUIRED_FIELDS["draft"][1],
    }

    if not results["pre_draft_analysis"]:
        log.debug(
            "No PRE-DRAFT ANALYSIS SUMMARY found. "
            "Argument and completeness models will have less context — consider adding one."
        )

    return results


def parse_publication_handoff(text):
    def section(header):
        idx = PUB_HEADERS.index(header)
        next_h = PUB_HEADERS[idx + 1 :] if idx + 1 < len(PUB_HEADERS) else []
        return _extract_section(text, header, next_h or None)

    title = _extract_field(text, "Article:")
    publication = _extract_field(text, "Publication:")

    pub_params_raw = section("PUBLICATION PARAMETERS")
    seo_raw = section("SEO METADATA")

    return {
        "title": title,
        "publication": publication,
        "publication_parameters": _parse_key_value_block(pub_params_raw),
        "seo": _parse_seo_block(seo_raw),
        "embeds": section("EMBEDS AND SPECIAL ELEMENTS"),
        "disposition_log": section("DISPOSITION LOG"),
        "final_draft": section("FINAL DRAFT"),
    }


def _extract_field(text, label):
    match = re.search(rf"^{re.escape(label)}\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _parse_key_value_block(text):
    result = {}
    for line in text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip().lower().replace(" ", "_")] = value.strip()
    return result


def _parse_seo_block(text):
    seo = {}
    field_map = {
        "Focus keyword": "focus_keyword",
        "Meta description": "meta_description",
        "OG title": "og_title",
        "OG description": "og_description",
        "Schema type": "schema_type",
    }
    for label, key in field_map.items():
        match = re.search(rf"^{re.escape(label)}:\s*(.+)$", text, re.MULTILINE)
        if match:
            value = match.group(1).strip()
            if not value.startswith("derive") and not value.startswith("use "):
                seo[key] = value
    return seo
