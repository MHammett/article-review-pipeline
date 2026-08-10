"""Cost estimation from API token counts.

Prices are per-million tokens (input, output).
Perplexity pricing is per-request-based in practice but approximated here
using token counts so the table stays consistent.

Pricing is loaded from configs/pricing.yaml at import time so model additions
and price changes require only a YAML edit, not a code change.  The hardcoded
The YAML is the single source of truth; there is no duplicate table in Python.
"""

import logging
from pathlib import Path

import yaml as _yaml  # noqa: F401

from ci_core.config_helpers import PackagedConfigError, load_packaged_yaml

log = logging.getLogger(__name__)

# Hardcoded fallback — used only when configs/pricing.yaml is missing or unreadable.


def _load_pricing():
    """Load pricing from the packaged configs/pricing.yaml.

    Raises PackagedConfigError if the file is missing, unparseable, or does not
    contain a usable pricing table. There is deliberately no hardcoded fallback:
    the duplicate table this replaced had to be edited in lockstep with the YAML
    on the config surface that changes most often, and in the only state it
    could ever have fired — a broken install — silently serving stale prices is
    worse than saying so.
    """
    yaml_path = Path(__file__).parent.parent / "configs" / "pricing.yaml"
    data = load_packaged_yaml(yaml_path)

    pricing = {}
    for model_id, pair in (data.get("models") or {}).items():
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise PackagedConfigError(
                f"{yaml_path}: models.{model_id} must be a [prompt, completion] "
                f"pair, got {pair!r}"
            )
        pricing[str(model_id)] = (float(pair[0]), float(pair[1]))
    if not pricing:
        raise PackagedConfigError(f"{yaml_path}: 'models' is empty or missing")

    unknown_raw = data.get("unknown_price")
    if not (isinstance(unknown_raw, (list, tuple)) and len(unknown_raw) == 2):
        raise PackagedConfigError(
            f"{yaml_path}: 'unknown_price' must be a [prompt, completion] pair"
        )
    return pricing, (float(unknown_raw[0]), float(unknown_raw[1]))


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
