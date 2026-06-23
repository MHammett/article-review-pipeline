#!/usr/bin/env python3
"""
Model discovery — queries each provider's live models API and reports what's
currently available, so you always know when newer models exist without reading
every provider's changelog yourself.

For each configured provider the script:
  1. Calls the provider's models list endpoint using your existing API key
  2. Filters to chat/completion models (strips embeddings, TTS, fine-tunes, etc.)
  3. Sorts by creation date newest-first
  4. Marks your currently configured model
  5. Flags models that are newer than what you have configured
  6. Notes models from the built-in superseded registry

Run this any time you want to check whether you're on the latest available model —
especially after a provider announces a new release.

Usage:
  python discover.py
  python discover.py --provider openai
  python discover.py --provider gemini --provider claude
  python discover.py --config-dir configs
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
import datetime
import requests

from config_loader import load_user_config, _normalize_model_configs
from model_registry import _SUPERSEDED, _NEWER_AVAILABLE, REGISTRY_DATE
from redact import redact_url_keys

NEW    = "\033[32m NEW\033[0m"
ACTIVE = "\033[36m  ✓\033[0m"
WARN   = "\033[33m  ⚠\033[0m"
INFO   = "    "
ERR    = "\033[31m ERR\033[0m"
SKIP   = "\033[33mSKIP\033[0m"

_UNKNOWN_DATE = datetime.date.min


def _iso(ts):
    """Convert a Unix timestamp (int) or ISO string to a datetime.date, or None."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.datetime.utcfromtimestamp(ts).date()
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(ts, str):
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"):
            try:
                return datetime.datetime.strptime(ts[:26], fmt[:len(ts)]).date()
            except ValueError:
                continue
    return None


def _days_ago(d):
    if d is None or d == _UNKNOWN_DATE:
        return "unknown date"
    days = (datetime.date.today() - d).days
    if days == 0:
        return "today"
    if days < 30:
        return f"{days}d ago"
    if days < 365:
        return f"{days // 30}mo ago"
    return f"{days / 365:.1f}yr ago"


def _print_model_row(model_id, date, configured_id, configured_date, prefix=""):
    marker_raw = INFO
    note = ""

    is_configured = model_id == configured_id
    is_newer = (
        date is not None
        and configured_date is not None
        and date > configured_date
        and not is_configured
    )
    in_superseded = model_id in _SUPERSEDED
    in_newer_available = model_id in _NEWER_AVAILABLE

    if is_configured:
        marker_raw = ACTIVE
        note = "  ← configured"
    elif is_newer:
        marker_raw = NEW
        note = "  ← newer than configured"
    elif in_superseded:
        marker_raw = WARN
        repl = _SUPERSEDED[model_id]["replacement"]
        note = f"  ⚠ superseded → {repl}"
    elif in_newer_available:
        note = f"  (soft upgrade: see model_registry.py)"

    date_str = f"  {date.isoformat()}  ({_days_ago(date)})" if date else "  (no date)"
    print(f"  {marker_raw}  {prefix}{model_id}{date_str}{note}")


# ---------------------------------------------------------------------------
# Per-provider discovery functions
# ---------------------------------------------------------------------------

def _discover_openai(api_key, configured_id):
    resp = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20,
    )
    resp.raise_for_status()
    all_models = resp.json().get("data", [])

    # Keep chat/reasoning models; drop embeddings, fine-tunes, audio, images, moderation.
    _CHAT_PREFIXES = ("gpt-5", "gpt-4", "gpt-3.5", "o1", "o3", "o4", "chatgpt-4o")
    _SKIP_PREFIXES = (
        "text-embedding", "text-moderation", "text-davinci", "text-curie",
        "text-babbage", "text-ada", "tts-", "whisper-", "dall-e-", "babbage-",
        "davinci-", "curie-", "ada-",
    )

    def _keep(m):
        mid = m["id"]
        if ":" in mid:
            return False  # fine-tuned (ft:gpt-4:...)
        if any(mid.startswith(p) for p in _SKIP_PREFIXES):
            return False
        return any(mid.startswith(p) for p in _CHAT_PREFIXES)

    models = [m for m in all_models if _keep(m)]
    models.sort(key=lambda m: m.get("created", 0), reverse=True)
    return [(m["id"], _iso(m.get("created"))) for m in models]


def _discover_gemini_aistudio(api_key, configured_id):
    resp = requests.get(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
        timeout=20,
    )
    resp.raise_for_status()
    all_models = resp.json().get("models", [])

    # Keep models that support generateContent; drop embedding-only models.
    def _keep(m):
        methods = m.get("supportedGenerationMethods", [])
        return "generateContent" in methods

    # Model name is like "models/gemini-2.5-flash" — strip the prefix
    models = [(m["name"].removeprefix("models/"), None) for m in all_models if _keep(m)]
    # Sort: newer-looking model IDs last in natural sort, so reverse for newest-first
    models.sort(key=lambda x: x[0], reverse=True)
    return models


def _discover_mistral(api_key, configured_id):
    resp = requests.get(
        "https://api.mistral.ai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])

    def _keep(m):
        return "embed" not in m["id"].lower()

    models = [m for m in data if _keep(m)]
    models.sort(key=lambda m: m.get("created", 0), reverse=True)
    return [(m["id"], _iso(m.get("created"))) for m in models]


def _discover_claude(api_key, configured_id):
    resp = requests.get(
        "https://api.anthropic.com/v1/models",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    models = [(m["id"], _iso(m.get("created_at"))) for m in data]
    models.sort(key=lambda x: x[1] or _UNKNOWN_DATE, reverse=True)
    return models


def _discover_grok(api_key, configured_id):
    resp = requests.get(
        "https://api.x.ai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])

    def _keep(m):
        return m["id"].startswith("grok-")

    models = [m for m in data if _keep(m)]
    models.sort(key=lambda m: m.get("created", 0), reverse=True)
    return [(m["id"], _iso(m.get("created"))) for m in models]


def _discover_perplexity(_api_key, _configured_id):
    # Perplexity does not publish a models list endpoint.
    # Return the documented set from June 2026 as a static fallback.
    static = [
        ("sonar-deep-research", None),
        ("sonar-reasoning-pro", None),
        ("sonar-pro",           None),
        ("sonar",               None),
    ]
    return static  # caller will note this is static


# ---------------------------------------------------------------------------
# Provider dispatch table
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "openai":     (_discover_openai,     "OpenAI"),
    "gemini":     (_discover_gemini_aistudio, "Gemini (AI Studio)"),
    "mistral":    (_discover_mistral,    "Mistral"),
    "claude":     (_discover_claude,     "Anthropic / Claude"),
    "grok":       (_discover_grok,       "Grok / xAI"),
    "perplexity": (_discover_perplexity, "Perplexity"),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Article Review Pipeline — live model discovery"
    )
    parser.add_argument(
        "--provider", action="append", dest="providers", metavar="NAME",
        help="Run discovery for this provider only. Repeat for multiple. "
             f"Valid: {', '.join(_PROVIDERS)}. Default: all configured providers.",
    )
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()

    try:
        user_config = load_user_config(args.config_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Config error: {e}")
        sys.exit(1)

    api_keys     = user_config.get("api_keys", {})
    model_configs = _normalize_model_configs(user_config.get("models", {}))

    # Apply cost_preset to get effective models (same as pipeline does at runtime)
    try:
        from config_loader import _apply_cost_preset, _apply_preset_overrides
        pipeline_cfg = user_config.get("pipeline", {})
        pipeline_cfg, models_raw = _apply_cost_preset(pipeline_cfg, user_config.get("models", {}))
        models_raw = _apply_preset_overrides(pipeline_cfg, models_raw)
        model_configs = _normalize_model_configs(models_raw)
    except Exception:
        pass  # fall back to raw model_configs on any error

    providers_to_check = args.providers or list(_PROVIDERS.keys())
    invalid = [p for p in providers_to_check if p not in _PROVIDERS]
    if invalid:
        print(f"Unknown provider(s): {', '.join(invalid)}. Valid: {', '.join(_PROVIDERS)}")
        sys.exit(1)

    today = datetime.date.today()
    registry_age = (today - REGISTRY_DATE).days
    print(f"\nModel Discovery Report — {today.isoformat()}")
    print(f"Built-in registry last updated: {REGISTRY_DATE.isoformat()} ({registry_age} days ago)")
    print("=" * 70)

    for provider_key in providers_to_check:
        fn, label = _PROVIDERS[provider_key]

        cfg         = model_configs.get(provider_key, {})
        prov_type   = cfg.get("provider", "")
        configured_id = cfg.get("model", "")
        enabled       = cfg.get("enabled", True)
        creds         = api_keys.get(provider_key, {})
        api_key       = creds.get("api_key", "") or ""

        print(f"\n{label}  (configured: {configured_id or '—'}{'' if enabled else ', disabled'})")

        # --- Vertex AI: no AI Studio key, listing requires separate auth ---
        if provider_key == "gemini" and prov_type == "vertex_ai":
            project  = cfg.get("project", "?")
            location = cfg.get("location", "?")
            print(
                f"  {SKIP}  Gemini is configured via Vertex AI (project={project} "
                f"location={location}).\n"
                "       Model listing against Vertex AI requires the gcloud SDK and is "
                "not supported here.\n"
                "       Check https://ai.google.dev/models for available Gemini models.\n"
                f"       Configured model: {configured_id!r}"
            )
            continue

        if not api_key:
            print(f"  {SKIP}  No API key configured — skipping")
            continue

        if not enabled:
            print(f"  {SKIP}  Provider disabled (enabled: false)")
            continue

        # --- Call the provider API ---
        is_static = provider_key == "perplexity"
        if is_static:
            print("  (No models endpoint — showing documented set from June 2026)")

        try:
            models = fn(api_key, configured_id)
        except requests.exceptions.HTTPError as e:
            print(f"  {ERR}  HTTP {e.response.status_code}: {redact_url_keys(e.response.text[:200])}")
            continue
        except Exception as e:
            print(f"  {ERR}  {redact_url_keys(e)}")
            continue

        if not models:
            print("  (No models returned)")
            continue

        # Find configured model's date for "newer than" comparison
        configured_date = next(
            (d for mid, d in models if mid == configured_id),
            None
        )

        for model_id, date in models:
            _print_model_row(model_id, date, configured_id, configured_date)

    print("\n" + "=" * 70)
    print(
        "Legend:\n"
        f"  {ACTIVE}  currently configured model\n"
        f"  {NEW}  available and newer than configured (by creation date)\n"
        f"  {WARN}  listed in built-in superseded registry\n"
        f"  {INFO}  available; not configured, not superseded\n"
    )
    print(
        "To update your configured model, edit models: in configs/user.yaml\n"
        "and run: python check.py --publication your_publication_name\n"
        "\nTo update the built-in registry after a model audit:\n"
        "  1. Edit model_registry.py: _SUPERSEDED, _NEWER_AVAILABLE\n"
        "  2. Bump REGISTRY_DATE to today's date\n"
    )


if __name__ == "__main__":
    main()
