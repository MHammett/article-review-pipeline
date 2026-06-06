import json
import time
import logging
import requests

DEFAULT_MODEL = "mistral-large-latest"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

log = logging.getLogger(__name__)


def call(system_prompt, user_prompt, api_key, retry=True, retry_delay=10, model=None):
    model = model or DEFAULT_MODEL
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }

    session = requests.Session()
    t0 = time.monotonic()

    def _post():
        resp = session.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=120)
        if resp.status_code in (429, 500, 502, 503, 504) and retry:
            log.warning(f"Mistral HTTP {resp.status_code}. Waiting {retry_delay}s before retry.")
            time.sleep(retry_delay)
            resp = session.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    try:
        raw = _post()
    except Exception as e:
        elapsed = round(time.monotonic() - t0, 2)
        log.error(f"Mistral call failed after {elapsed}s: {e}")
        session.close()
        return {"failed": True, "error": str(e), "raw": None, "model": model, "tokens": {}, "elapsed_seconds": elapsed}
    finally:
        session.close()

    elapsed = round(time.monotonic() - t0, 2)
    content = raw["choices"][0]["message"]["content"]
    usage = raw.get("usage", {})

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        log.warning(f"Mistral returned non-JSON content after {elapsed}s")
        return {
            "failed": True,
            "error": "Malformed JSON response",
            "raw": content,
            "model": model,
            "tokens": usage,
            "elapsed_seconds": elapsed,
        }

    log.debug(f"Mistral call succeeded in {elapsed}s ({usage.get('total_tokens', '?')} tokens)")
    return {
        "failed": False,
        "data": parsed,
        "model": model,
        "tokens": {
            "prompt": usage.get("prompt_tokens"),
            "completion": usage.get("completion_tokens"),
        },
        "elapsed_seconds": elapsed,
    }
