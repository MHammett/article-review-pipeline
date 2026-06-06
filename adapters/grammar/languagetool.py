import requests
import time
import logging

LANGUAGETOOL_API_URL = "https://api.languagetool.org/v2/check"

log = logging.getLogger(__name__)


def check_text(text, username, api_key, language="en-US"):
    payload = {
        "text": text,
        "language": language,
        "username": username,
        "apiKey": api_key,
    }
    resp = requests.post(LANGUAGETOOL_API_URL, data=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def apply_corrections(text, matches, auto_apply_categories, suppress_categories):
    """
    Apply corrections for matches whose rule category is in auto_apply_categories,
    skipping any in suppress_categories.

    Returns (corrected_text, change_log).
    """
    change_log = []
    # Process matches in reverse order so offsets stay valid
    sorted_matches = sorted(matches, key=lambda m: m["offset"], reverse=True)

    for match in sorted_matches:
        rule_id = match.get("rule", {}).get("id", "")
        category_id = match.get("rule", {}).get("category", {}).get("id", "")

        if category_id in suppress_categories:
            continue

        if not match.get("replacements"):
            continue

        if category_id not in auto_apply_categories:
            continue

        offset = match["offset"]
        length = match["length"]
        replacement = match["replacements"][0]["value"]
        original = text[offset: offset + length]

        if original == replacement:
            continue

        text = text[:offset] + replacement + text[offset + length:]
        change_log.append({
            "rule_id": rule_id,
            "category": category_id,
            "original": original,
            "replacement": replacement,
            "offset": offset,
            "message": match.get("message", ""),
        })

    change_log.reverse()  # back to document order
    return text, change_log


def run(text, lt_config, username, api_key, retry=True, retry_delay=10):
    """
    Main entry point. Returns dict with keys:
      corrected_text, change_log, flagged_matches, failed (bool)
    """
    auto_apply = set(lt_config.get("auto_apply", []))
    flag_for_review = set(lt_config.get("flag_for_review", []))
    suppress = set(lt_config.get("suppress", []))

    try:
        result = check_text(text, username, api_key)
    except Exception as e:
        if retry:
            log.warning(f"LanguageTool first attempt failed: {e}. Retrying in {retry_delay}s.")
            time.sleep(retry_delay)
            try:
                result = check_text(text, username, api_key)
            except Exception as e2:
                log.error(f"LanguageTool failed after retry: {e2}")
                return {
                    "corrected_text": text,
                    "change_log": [],
                    "flagged_matches": [],
                    "failed": True,
                    "error": str(e2),
                }
        else:
            return {
                "corrected_text": text,
                "change_log": [],
                "flagged_matches": [],
                "failed": True,
                "error": str(e),
            }

    matches = result.get("matches", [])
    corrected_text, change_log = apply_corrections(text, matches, auto_apply, suppress)

    flagged = []
    for match in matches:
        category_id = match.get("rule", {}).get("category", {}).get("id", "")
        if category_id in flag_for_review:
            flagged.append({
                "rule_id": match.get("rule", {}).get("id", ""),
                "category": category_id,
                "message": match.get("message", ""),
                "context": match.get("context", {}).get("text", ""),
                "offset": match["offset"],
                "length": match["length"],
                "replacements": [r["value"] for r in match.get("replacements", [])[:3]],
            })

    return {
        "corrected_text": corrected_text,
        "change_log": change_log,
        "flagged_matches": flagged,
        "failed": False,
    }
