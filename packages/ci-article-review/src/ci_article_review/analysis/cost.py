"""Cost estimation from API token counts.

Prices are per-million tokens (input, output).
Perplexity pricing is per-request-based in practice but approximated here
using token counts so the table stays consistent.

Pricing is loaded from configs/pricing.yaml at import time so model additions
and price changes require only a YAML edit, not a code change.  The hardcoded
_PRICING_FALLBACK is used only when the YAML file cannot be loaded.
"""

import logging
from pathlib import Path

import yaml as _yaml

log = logging.getLogger(__name__)

# Hardcoded fallback — used only when configs/pricing.yaml is missing or unreadable.
_PRICING_FALLBACK: dict[str, tuple[float, float]] = {
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "mistral-large-latest": (3.00, 9.00),
    "mistral-medium-3-5": (2.00, 6.00),
    "mistral-small-latest": (0.10, 0.30),
    "sonar-reasoning-pro": (2.00, 8.00),
    "sonar-pro": (1.00, 8.00),
    "sonar": (0.50, 1.50),
    "sonar-deep-research": (2.00, 8.00),
    "grok-4.20-0309-reasoning": (1.25, 2.50),
    "grok-4.20-0309-non-reasoning": (1.25, 2.50),
    "grok-4.3": (1.25, 2.50),
    "grok-build-0.1": (0.30, 1.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}
_UNKNOWN_PRICE_FALLBACK = (2.50, 10.00)


def _load_pricing():
    """Load pricing from configs/pricing.yaml; fall back to hardcoded table."""
    yaml_path = Path(__file__).parent.parent / "configs" / "pricing.yaml"
    if not yaml_path.exists():
        return _PRICING_FALLBACK, _UNKNOWN_PRICE_FALLBACK
    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = _yaml.safe_load(f) or {}
        raw = data.get("models", {})
        pricing = {}
        for model_id, pair in raw.items():
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                pricing[str(model_id)] = (float(pair[0]), float(pair[1]))
        unknown_raw = data.get("unknown_price", list(_UNKNOWN_PRICE_FALLBACK))
        unknown = (float(unknown_raw[0]), float(unknown_raw[1]))
        return pricing, unknown
    except Exception as exc:
        log.warning(
            "Could not load configs/pricing.yaml (%s) — using built-in defaults", exc
        )
        return _PRICING_FALLBACK, _UNKNOWN_PRICE_FALLBACK


_PRICING, _UNKNOWN_PRICE = _load_pricing()


def _price_for_model(model_id):
    if not model_id:
        return _UNKNOWN_PRICE
    # Exact match first
    if model_id in _PRICING:
        return _PRICING[model_id]
    # Prefix match (handles versioned IDs like "gemini-2.5-flash-0520").
    # Sort longest key first so "gpt-5.4-mini" is tested before "gpt-5.4".
    for key in sorted(_PRICING, key=len, reverse=True):
        if model_id.startswith(key):
            return _PRICING[key]
    return _UNKNOWN_PRICE


def _entry_cost(entry):
    """Return (input_cost_usd, output_cost_usd) for one api_call_log entry."""
    if entry.get("failed"):
        return 0.0, 0.0
    tokens = entry.get("tokens") or {}
    prompt_tok = tokens.get("prompt", 0) or 0
    completion_tok = tokens.get("completion", 0) or 0
    model_id = entry.get("model", "")
    # Strip fallback annotations like " [FALLBACK from gpt-5.4]"
    model_id = model_id.split(" ")[0] if model_id else ""
    in_price, out_price = _price_for_model(model_id)
    return (prompt_tok / 1_000_000 * in_price, completion_tok / 1_000_000 * out_price)


def calculate(api_call_log):
    """Return a cost summary dict from the api_call_log list.

    Keys in the returned dict:
      total_usd         float  — grand total across all calls
      total_input_usd   float
      total_output_usd  float
      by_pass           list   — [{pass, model, input_usd, output_usd, total_usd}]
      pricing_known     bool   — False when any model fell back to unknown pricing
    """
    by_pass = []
    total_in = 0.0
    total_out = 0.0
    pricing_known = True

    for entry in api_call_log or []:
        in_usd, out_usd = _entry_cost(entry)
        model_id = (entry.get("model") or "").split(" ")[0]
        if model_id and model_id not in _PRICING:
            # Check prefix match
            if not any(model_id.startswith(k) for k in _PRICING):
                pricing_known = False
        by_pass.append(
            {
                "pass": entry.get("pass", ""),
                "model": entry.get("model", ""),
                "input_usd": round(in_usd, 6),
                "output_usd": round(out_usd, 6),
                "total_usd": round(in_usd + out_usd, 6),
            }
        )
        total_in += in_usd
        total_out += out_usd

    return {
        "total_usd": round(total_in + total_out, 4),
        "total_input_usd": round(total_in, 4),
        "total_output_usd": round(total_out, 4),
        "by_pass": by_pass,
        "pricing_known": pricing_known,
    }
