#!/usr/bin/env python3
"""
Connectivity check — verifies every configured API key works before
you submit a real article. Makes one minimal call to each service
and reports pass/fail with actionable error messages.

Usage:
  python check.py --publication your_publication_name
"""
import sys

if sys.version_info < (3, 10):
    print(f"ERROR: Python 3.10+ required. You are running {sys.version}.")
    sys.exit(1)

import importlib.util
_REQUIRED = {"requests": "requests", "yaml": "pyyaml", "dotenv": "python-dotenv"}
_missing = [pkg for mod, pkg in _REQUIRED.items() if not importlib.util.find_spec(mod)]
if _missing:
    print(f"ERROR: Missing packages. Run: pip install {' '.join(_missing)}")
    sys.exit(1)

import argparse
import json
import requests
import base64

from config_loader import load_user_config, load_publication_config, merge_configs, validate_publication_name

PASS = "\033[32m PASS\033[0m"
FAIL = "\033[31m FAIL\033[0m"
SKIP = "\033[33m SKIP\033[0m"


def check(label, fn):
    try:
        msg = fn()
        print(f"  {PASS}  {label}{(' — ' + msg) if msg else ''}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {label} — {e}")
        return False


def check_openai(api_key, model):
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": "Reply with the single word: ok"}], "max_tokens": 5},
        timeout=30,
    )
    resp.raise_for_status()
    reply = resp.json()["choices"][0]["message"]["content"].strip()
    return f"model={model}, replied: {reply!r}"


def check_gemini(api_key, model):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    resp = requests.post(
        url,
        json={"contents": [{"parts": [{"text": "Reply with the single word: ok"}]}],
              "generationConfig": {"maxOutputTokens": 5}},
        timeout=30,
    )
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
    candidates = resp.json().get("candidates", [])
    if not candidates:
        raise Exception("No candidates returned")
    reply = candidates[0]["content"]["parts"][0]["text"].strip()
    return f"model={model}, replied: {reply!r}"


def check_mistral(api_key, model):
    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": "Reply with the single word: ok"}], "max_tokens": 5},
        timeout=30,
    )
    resp.raise_for_status()
    reply = resp.json()["choices"][0]["message"]["content"].strip()
    return f"model={model}, replied: {reply!r}"


def check_grok(api_key, model):
    resp = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": "Reply with the single word: ok"}], "max_tokens": 5},
        timeout=30,
    )
    resp.raise_for_status()
    reply = resp.json()["choices"][0]["message"]["content"].strip()
    return f"model={model}, replied: {reply!r}"


def check_languagetool(username, api_key):
    resp = requests.post(
        "https://api.languagetool.org/v2/check",
        data={"text": "This are a test.", "language": "en-US", "username": username, "apiKey": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    matches = resp.json().get("matches", [])
    return f"API responded, {len(matches)} match(es) on test sentence"


def check_wordpress(site_url, username, app_password):
    site_url = site_url.rstrip("/")
    api_base = f"{site_url}/wp-json/wp/v2"

    # First check: REST API is reachable at all
    resp = requests.get(f"{site_url}/wp-json/wp/v2", timeout=15)
    if resp.status_code != 200:
        raise Exception(f"REST API returned HTTP {resp.status_code}. Check Settings → Permalinks in your WP admin.")

    # Second check: credentials are valid
    token = base64.b64encode(f"{username}:{app_password}".encode()).decode()
    resp = requests.get(
        f"{api_base}/users/me",
        headers={"Authorization": f"Basic {token}"},
        timeout=15,
    )
    if resp.status_code == 401:
        raise Exception("Authentication failed. Check your username and application password.")
    resp.raise_for_status()
    data = resp.json()
    wp_user = data.get("name") or data.get("slug") or "unknown"
    return f"authenticated as '{wp_user}' at {site_url}"


def main():
    parser = argparse.ArgumentParser(description="Article Review Pipeline — connectivity check")
    parser.add_argument("--publication", required=True, help="Publication config name (without .yaml)")
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()

    try:
        validate_publication_name(args.publication)
        user_config = load_user_config(args.config_dir)
        pub_config_raw = load_publication_config(args.publication, args.config_dir)
        config = merge_configs(user_config, pub_config_raw)
    except (FileNotFoundError, ValueError) as e:
        print(f"Config error: {e}")
        sys.exit(1)

    api_keys = config["api_keys"]
    models = config.get("models", {})
    pub = config["publication"]
    wp = pub.get("wordpress", {})
    pipeline_cfg = config.get("pipeline", {})

    print(f"\nConnectivity check — publication: {args.publication}\n")

    results = []

    # --- AI review models (required) ---
    print("Review models (required):")
    results.append(check(
        "OpenAI",
        lambda: check_openai(api_keys["openai"]["api_key"], models.get("openai", "gpt-4o"))
    ))
    results.append(check(
        "Gemini",
        lambda: check_gemini(api_keys["gemini"]["api_key"], models.get("gemini", "gemini-2.0-flash"))
    ))
    results.append(check(
        "Mistral",
        lambda: check_mistral(api_keys["mistral"]["api_key"], models.get("mistral", "mistral-large-latest"))
    ))

    # --- Grok (optional) ---
    print("\nReview models (optional):")
    grok_creds = api_keys.get("grok", {})
    if grok_creds.get("api_key"):
        results.append(check(
            "Grok",
            lambda: check_grok(grok_creds["api_key"], models.get("grok", "grok-3-latest"))
        ))
    else:
        print(f"  {SKIP}  Grok — no api_key configured (optional)")

    # --- LanguageTool (optional) ---
    lt_creds = api_keys.get("languagetool", {})
    grammar_enabled = pipeline_cfg.get("grammar_pass", True)
    if not grammar_enabled:
        print(f"  {SKIP}  LanguageTool — grammar_pass: false")
    elif lt_creds.get("username") and lt_creds.get("api_key"):
        results.append(check(
            "LanguageTool",
            lambda: check_languagetool(lt_creds["username"], lt_creds["api_key"])
        ))
    else:
        print(f"  {SKIP}  LanguageTool — no credentials configured (optional)")

    # --- WordPress ---
    print("\nWordPress:")
    if wp.get("site_url") and wp.get("username") and wp.get("application_password"):
        results.append(check(
            "WordPress REST API + credentials",
            lambda: check_wordpress(wp["site_url"], wp["username"], wp["application_password"])
        ))
    else:
        print(f"  {SKIP}  WordPress — missing site_url, username, or application_password in publication config")

    # --- Summary ---
    failures = results.count(False)
    print(f"\n{'All checks passed.' if failures == 0 else f'{failures} check(s) failed — fix the errors above before running the pipeline.'}\n")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
