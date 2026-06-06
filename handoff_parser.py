"""
Parse Template A (draft submission) and Template C (publication handoff) documents.
"""
import re


def _extract_section(text, header, next_headers=None):
    """Extract content between a header and the next known header."""
    pattern = rf"^{re.escape(header)}\s*\n(.*?)(?=\n(?:{'|'.join(re.escape(h) for h in next_headers)})\s*\n|\Z)"
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
        next_h = DRAFT_HEADERS[idx + 1:] if idx + 1 < len(DRAFT_HEADERS) else []
        return _extract_section(text, header, next_h)

    # Extract metadata from header block
    title = _extract_field(text, "Article:")
    publication = _extract_field(text, "Publication:")
    run_number = _extract_field(text, "Pipeline run:")

    return {
        "title": title,
        "publication": publication,
        "run_number": int(run_number) if run_number and run_number.isdigit() else 1,
        "primary_claim": section("PRIMARY CLAIM"),
        "target_audience": section("TARGET AUDIENCE"),
        "pre_draft_analysis": section("PRE-DRAFT ANALYSIS SUMMARY"),
        "sources_cited": section("SOURCES ALREADY CITED"),
        "uncertain_sections": section("UNCERTAIN SECTIONS"),
        "known_gaps": section("KNOWN GAPS"),
        "additional_context": section("ADDITIONAL CONTEXT FOR REVIEW MODELS"),
        "draft": section("DRAFT"),
    }


def parse_publication_handoff(text):
    def section(header):
        idx = PUB_HEADERS.index(header)
        next_h = PUB_HEADERS[idx + 1:] if idx + 1 < len(PUB_HEADERS) else []
        return _extract_section(text, header, next_h)

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
