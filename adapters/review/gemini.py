import json
import time
import logging
import requests

MODEL = "gemini-2.0-flash"  # gemini-flash-latest

log = logging.getLogger(__name__)


def _api_url(api_key):
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent?key={api_key}"
    )


def call(system_prompt, user_prompt, api_key, retry=True, retry_delay=10):
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }

    def _post():
        resp = requests.post(_api_url(api_key), json=payload, timeout=120)
        if resp.status_code == 429 and retry:
            log.warning(f"Gemini rate-limited. Waiting {retry_delay}s.")
            time.sleep(retry_delay)
            resp = requests.post(_api_url(api_key), json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    grounding_available = True
    try:
        raw = _post()
    except Exception as e:
        log.warning(f"Gemini with search grounding failed: {e}. Retrying without grounding.")
        grounding_available = False
        payload_no_ground = {k: v for k, v in payload.items() if k != "tools"}
        try:
            resp = requests.post(_api_url(api_key), json=payload_no_ground, timeout=120)
            resp.raise_for_status()
            raw = resp.json()
        except Exception as e2:
            log.error(f"Gemini call failed entirely: {e2}")
            return {
                "failed": True,
                "error": str(e2),
                "raw": None,
                "model": MODEL,
                "tokens": {},
                "grounding_available": False,
            }

    candidates = raw.get("candidates", [])
    if not candidates:
        return {
            "failed": True,
            "error": "No candidates in Gemini response",
            "raw": raw,
            "model": MODEL,
            "tokens": {},
            "grounding_available": grounding_available,
        }

    content_parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in content_parts)
    usage = raw.get("usageMetadata", {})

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning("Gemini returned non-JSON content")
            return {
                "failed": True,
                "error": "Malformed JSON response",
                "raw": text,
                "model": MODEL,
                "tokens": usage,
                "grounding_available": grounding_available,
            }

    return {
        "failed": False,
        "data": parsed,
        "model": MODEL,
        "tokens": {
            "prompt": usage.get("promptTokenCount"),
            "completion": usage.get("candidatesTokenCount"),
        },
        "grounding_available": grounding_available,
    }
