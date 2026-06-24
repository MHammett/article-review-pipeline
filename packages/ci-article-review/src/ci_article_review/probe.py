#!/usr/bin/env python3
"""
Probe — empirically verifies every model ID and parameter assumption baked
into the cost presets.  Makes minimal API calls (one tiny prompt each) and
reports exactly what each provider accepted, rejected, or returned.

Usage:
  python probe.py                # test all providers
  python probe.py openai        # test one provider only
  python probe.py openai mistral gemini

This is intentionally separate from check.py, which only tests the model
currently in user.yaml.  Probe tests the specific model IDs that live in
the balanced/thorough/maximum presets so we know they'll work before we pay
for a full article run.
"""

import os
import time
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()

OK = "\033[32m OK  \033[0m"
FAIL = "\033[31m FAIL\033[0m"
WARN = "\033[33m WARN\033[0m"
SKIP = "\033[33m SKIP\033[0m"

MINI_PROMPT = "Reply with the single word: ok"

# ---------------------------------------------------------------------------
# Models to probe — mirrors the cost-preset assumptions
# ---------------------------------------------------------------------------

OPENAI_MODELS = [
    # (model_id, reasoning_effort_or_None, label)
    ("gpt-5.4-mini", None, "economy/standard non-reasoning"),
    ("gpt-5.4", None, "standard non-reasoning"),
    ("o4-mini", "low", "balanced/thorough reasoning"),
    ("o3", "high", "maximum reasoning"),
]

MISTRAL_MODELS = [
    ("mistral-small-latest", None, "economy non-reasoning"),
    ("mistral-large-latest", None, "standard non-reasoning"),
    (
        "mistral-medium-3-5",
        "high",
        "balanced+ reasoning (replaces magistral-medium-latest; only high/none supported)",
    ),
]

GEMINI_MODELS = [
    "gemini-2.5-flash",  # standard / thorough
    "gemini-3.5-flash",  # maximum preset
]

GROK_MODELS = [
    "grok-4.3",  # standard — also supports reasoning_effort for balanced+ (no separate model needed)
    "grok-build-0.1",  # capacity fallback
]

PERPLEXITY_MODELS = [
    "sonar",  # economy
    "sonar-pro",  # standard
    "sonar-reasoning-pro",  # balanced+
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(ok, label, detail=""):
    tag = OK if ok else FAIL
    line = f"  {tag}  {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def _warn(label, detail=""):
    line = f"  {WARN}  {label}"
    if detail:
        line += f" — {detail}"
    print(line)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def probe_openai():
    print("\n-- OpenAI ------------------------------------------------------")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print(f"  {SKIP}  OPENAI_API_KEY not set — skipping")
        return

    for model, reasoning_effort, label in OPENAI_MODELS:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": MINI_PROMPT}],
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
            # Reasoning mode: no response_format or temperature
        else:
            payload["response_format"] = {"type": "text"}
            payload["max_completion_tokens"] = 20

        t0 = time.monotonic()
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            elapsed = round(time.monotonic() - t0, 1)
            if resp.status_code == 200:
                data = resp.json()
                actual_model = data.get("model", "?")
                # Reasoning models may return content as a list of typed chunks.
                raw_content = data["choices"][0]["message"]["content"]
                if isinstance(raw_content, list):
                    reply = "".join(
                        c.get("text", "")
                        for c in raw_content
                        if c.get("type") == "text"
                    ).strip()[:60]
                else:
                    reply = raw_content.strip()[:60]
                _result(
                    True,
                    f"{model} ({label})",
                    f"model_in_response={actual_model!r} reply={reply!r} {elapsed}s",
                )
            else:
                body = resp.text[:300]
                _result(False, f"{model} ({label})", f"HTTP {resp.status_code}: {body}")
        except Exception as e:
            elapsed = round(time.monotonic() - t0, 1)
            _result(False, f"{model} ({label})", f"{e} ({elapsed}s)")


# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------


def probe_mistral():
    print("\n-- Mistral -----------------------------------------------------")
    api_key = os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        print(f"  {SKIP}  MISTRAL_API_KEY not set — skipping")
        return

    for model, reasoning_effort, label in MISTRAL_MODELS:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": MINI_PROMPT}],
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        else:
            payload["response_format"] = {"type": "text"}
            payload["max_tokens"] = 20

        t0 = time.monotonic()
        try:
            resp = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=90,
            )
            elapsed = round(time.monotonic() - t0, 1)
            if resp.status_code == 200:
                data = resp.json()
                actual_model = data.get("model", "?")
                # Reasoning models may return content as a list of typed chunks.
                raw_content = data["choices"][0]["message"]["content"]
                if isinstance(raw_content, list):
                    reply = "".join(
                        c.get("text", "")
                        for c in raw_content
                        if c.get("type") == "text"
                    ).strip()[:60]
                else:
                    reply = raw_content.strip()[:60]
                _result(
                    True,
                    f"{model} ({label})",
                    f"model_in_response={actual_model!r} reply={reply!r} {elapsed}s",
                )
            else:
                body = resp.text[:300]
                _result(False, f"{model} ({label})", f"HTTP {resp.status_code}: {body}")
        except Exception as e:
            elapsed = round(time.monotonic() - t0, 1)
            _result(False, f"{model} ({label})", f"{e} ({elapsed}s)")


# ---------------------------------------------------------------------------
# Gemini (Vertex AI — uses your configured credentials)
# ---------------------------------------------------------------------------


def probe_gemini_vertex():
    print("\n-- Gemini (Vertex AI) ------------------------------------------")

    try:
        import google.auth
        import google.auth.transport.requests
        from google.oauth2 import service_account
    except ImportError:
        print(
            f"  {SKIP}  google-auth not installed — run: pip install 'google-auth>=2.22.0,<3.0'"
        )
        return

    credentials_file = (
        r"C:\Users\mhammett.HAMMETT\gcp-keys\mikehammett-317d1e506314.json"
    )
    project = "mikehammett"
    location = "us-central1"

    if not os.path.exists(credentials_file):
        print(f"  {SKIP}  credentials file not found: {credentials_file}")
        return

    try:
        creds = service_account.Credentials.from_service_account_file(
            credentials_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
    except Exception as e:
        print(f"  {FAIL}  credentials refresh failed — {e}")
        return

    for model in GEMINI_MODELS:
        url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{location}/publishers/google/models/{model}:generateContent"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": MINI_PROMPT}]}],
            "generationConfig": {"maxOutputTokens": 64},
        }
        t0 = time.monotonic()
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {creds.token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=45,
            )
            elapsed = round(time.monotonic() - t0, 1)
            if resp.status_code == 200:
                candidates = resp.json().get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text_parts = [
                        p for p in parts if not p.get("thought") and "text" in p
                    ]
                    reply = (
                        (text_parts[0]["text"].strip()[:60])
                        if text_parts
                        else "(no text part)"
                    )
                else:
                    reply = "(no candidates)"
                _result(True, f"{model}", f"replied={reply!r} {elapsed}s")
            else:
                body = resp.text[:300]
                _result(False, f"{model}", f"HTTP {resp.status_code}: {body}")
        except Exception as e:
            elapsed = round(time.monotonic() - t0, 1)
            _result(False, f"{model}", f"{e} ({elapsed}s)")


# ---------------------------------------------------------------------------
# Grok
# ---------------------------------------------------------------------------


def probe_grok():
    print("\n-- Grok --------------------------------------------------------")
    api_key = os.getenv("GROK_API_KEY", "")
    if not api_key:
        print(f"  {SKIP}  GROK_API_KEY not set — skipping")
        return

    for model in GROK_MODELS:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": MINI_PROMPT}],
            "max_tokens": 20,
        }
        t0 = time.monotonic()
        try:
            resp = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            elapsed = round(time.monotonic() - t0, 1)
            if resp.status_code == 200:
                data = resp.json()
                actual_model = data.get("model", "?")
                reply = data["choices"][0]["message"]["content"].strip()[:60]
                _result(
                    True,
                    f"{model}",
                    f"model_in_response={actual_model!r} reply={reply!r} {elapsed}s",
                )
            else:
                body = resp.text[:300]
                _result(False, f"{model}", f"HTTP {resp.status_code}: {body}")
        except Exception as e:
            elapsed = round(time.monotonic() - t0, 1)
            _result(False, f"{model}", f"{e} ({elapsed}s)")


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


def probe_perplexity():
    print("\n-- Perplexity --------------------------------------------------")
    api_key = os.getenv("PERPLEXITY_API_KEY", "")
    if not api_key:
        print(f"  {SKIP}  PERPLEXITY_API_KEY not set — skipping")
        return

    for model in PERPLEXITY_MODELS:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": MINI_PROMPT}],
            "max_tokens": 20,
        }
        t0 = time.monotonic()
        try:
            resp = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            elapsed = round(time.monotonic() - t0, 1)
            if resp.status_code == 200:
                data = resp.json()
                actual_model = data.get("model", "?")
                reply = data["choices"][0]["message"]["content"].strip()[:60]
                _result(
                    True,
                    f"{model}",
                    f"model_in_response={actual_model!r} reply={reply!r} {elapsed}s",
                )
            else:
                body = resp.text[:300]
                _result(False, f"{model}", f"HTTP {resp.status_code}: {body}")
        except Exception as e:
            elapsed = round(time.monotonic() - t0, 1)
            _result(False, f"{model}", f"{e} ({elapsed}s)")


# ---------------------------------------------------------------------------
# OpenAI — list available models (confirms IDs without making a chat call)
# ---------------------------------------------------------------------------


def list_openai_models():
    print("\n-- OpenAI available models (filtered for relevant families) ----")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print(f"  {SKIP}  OPENAI_API_KEY not set")
        return

    try:
        resp = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        models = sorted(m["id"] for m in resp.json().get("data", []))
        relevant = [
            m
            for m in models
            if any(
                m.startswith(prefix) for prefix in ("gpt-5", "gpt-4o", "o3", "o4", "o1")
            )
        ]
        for m in relevant:
            print(f"    {m}")
        if not relevant:
            print("    (no gpt-5/o3/o4 models found in your account)")
    except Exception as e:
        print(f"  {FAIL}  {e}")


# ---------------------------------------------------------------------------
# Mistral — list available models
# ---------------------------------------------------------------------------


def list_mistral_models():
    print("\n-- Mistral available models (filtered for relevant families) ---")
    api_key = os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        print(f"  {SKIP}  MISTRAL_API_KEY not set")
        return

    try:
        resp = requests.get(
            "https://api.mistral.ai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        models = sorted(m["id"] for m in resp.json().get("data", []))
        relevant = [
            m
            for m in models
            if any(kw in m for kw in ("mistral", "magistral", "codestral", "pixtral"))
        ]
        for m in relevant:
            print(f"    {m}")
        if not relevant:
            print("    (no models returned)")
    except Exception as e:
        print(f"  {FAIL}  {e}")


# ---------------------------------------------------------------------------
# Grok — list available models
# ---------------------------------------------------------------------------


def list_grok_models():
    print("\n-- Grok available models ---------------------------------------")
    api_key = os.getenv("GROK_API_KEY", "")
    if not api_key:
        print(f"  {SKIP}  GROK_API_KEY not set")
        return

    try:
        resp = requests.get(
            "https://api.x.ai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        models = sorted(m["id"] for m in resp.json().get("data", []))
        for m in models:
            print(f"    {m}")
        if not models:
            print("    (no models returned)")
    except Exception as e:
        print(f"  {FAIL}  {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

PROVIDER_MAP = {
    "openai": (probe_openai, list_openai_models),
    "mistral": (probe_mistral, list_mistral_models),
    "gemini": (probe_gemini_vertex, None),
    "grok": (probe_grok, list_grok_models),
    "perplexity": (probe_perplexity, None),
}


def main():
    parser = argparse.ArgumentParser(
        description="Probe preset model IDs and parameters."
    )
    parser.add_argument(
        "providers",
        nargs="*",
        choices=list(PROVIDER_MAP) + ["all"],
        default=["all"],
        help="Providers to probe (default: all)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Also fetch the provider's models list to confirm available IDs",
    )
    args = parser.parse_args()

    providers = list(PROVIDER_MAP) if "all" in args.providers else args.providers

    print("Probing preset model assumptions...")
    print(
        "Each call sends a one-word prompt.  Results show model ID from response body."
    )

    for name in providers:
        probe_fn, list_fn = PROVIDER_MAP[name]
        probe_fn()
        if args.list_models and list_fn:
            list_fn()

    print()


if __name__ == "__main__":
    main()
