import json
import time
import logging
import requests

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4o"  # chat-latest alias; update when OpenAI changes their default

log = logging.getLogger(__name__)


def call(system_prompt, user_prompt, api_key, retry=True, retry_delay=10):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }

    def _post():
        resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=120)
        if resp.status_code == 429 and retry:
            log.warning(f"OpenAI rate-limited. Waiting {retry_delay}s.")
            time.sleep(retry_delay)
            resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    try:
        raw = _post()
    except Exception as e:
        log.error(f"OpenAI call failed: {e}")
        return {"failed": True, "error": str(e), "raw": None, "model": MODEL, "tokens": {}}

    content = raw["choices"][0]["message"]["content"]
    usage = raw.get("usage", {})
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        log.warning("OpenAI returned non-JSON content")
        return {
            "failed": True,
            "error": "Malformed JSON response",
            "raw": content,
            "model": MODEL,
            "tokens": usage,
        }

    return {
        "failed": False,
        "data": parsed,
        "model": MODEL,
        "tokens": {"prompt": usage.get("prompt_tokens"), "completion": usage.get("completion_tokens")},
    }
