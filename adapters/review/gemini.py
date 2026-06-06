import json
import time
import logging
import requests

DEFAULT_MODEL = "gemini-2.0-flash"

log = logging.getLogger(__name__)


def _api_url(model, api_key):
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )


def _redact_key(text, api_key):
    if api_key and api_key in str(text):
        return str(text).replace(api_key, "[REDACTED]")
    return str(text)


def call(system_prompt, user_prompt, api_key, retry=True, retry_delay=10, model=None):
    model = model or DEFAULT_MODEL
    url = _api_url(model, api_key)

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }

    session = requests.Session()
    t0 = time.monotonic()

    def _post(use_grounding=True):
        p = payload if use_grounding else {k: v for k, v in payload.items() if k != "tools"}
        resp = session.post(url, json=p, timeout=120)
        if resp.status_code in (429, 500, 502, 503, 504) and retry:
            log.warning(f"Gemini HTTP {resp.status_code}. Waiting {retry_delay}s before retry.")
            time.sleep(retry_delay)
            resp = session.post(url, json=p, timeout=120)
        resp.raise_for_status()
        return resp.json()

    grounding_available = True
    try:
        raw = _post(use_grounding=True)
    except Exception as e:
        safe_err = _redact_key(e, api_key)
        log.warning(f"Gemini with search grounding failed: {safe_err}. Retrying without grounding.")
        grounding_available = False
        try:
            raw = _post(use_grounding=False)
        except Exception as e2:
            elapsed = round(time.monotonic() - t0, 2)
            safe_err2 = _redact_key(e2, api_key)
            log.error(f"Gemini call failed entirely after {elapsed}s: {safe_err2}")
            session.close()
            return {
                "failed": True,
                "error": safe_err2,
                "raw": None,
                "model": model,
                "tokens": {},
                "grounding_available": False,
                "elapsed_seconds": elapsed,
            }

    elapsed = round(time.monotonic() - t0, 2)
    session.close()

    candidates = raw.get("candidates", [])
    if not candidates:
        return {
            "failed": True,
            "error": "No candidates in Gemini response",
            "raw": raw,
            "model": model,
            "tokens": {},
            "grounding_available": grounding_available,
            "elapsed_seconds": elapsed,
        }

    content_parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in content_parts)
    usage = raw.get("usageMetadata", {})

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning(f"Gemini returned non-JSON content after {elapsed}s")
            return {
                "failed": True,
                "error": "Malformed JSON response",
                "raw": text,
                "model": model,
                "tokens": usage,
                "grounding_available": grounding_available,
                "elapsed_seconds": elapsed,
            }

    log.debug(f"Gemini call succeeded in {elapsed}s (grounding={grounding_available})")
    return {
        "failed": False,
        "data": parsed,
        "model": model,
        "tokens": {
            "prompt": usage.get("promptTokenCount"),
            "completion": usage.get("candidatesTokenCount"),
        },
        "grounding_available": grounding_available,
        "elapsed_seconds": elapsed,
    }
