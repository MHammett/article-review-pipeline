import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

REQUIRED_USER_KEYS = [
    # LanguageTool is intentionally absent — grammar_pass is optional.
    ("api_keys", "openai", "api_key"),
    ("api_keys", "gemini", "api_key"),
    ("api_keys", "mistral", "api_key"),
]

REQUIRED_PUB_KEYS = [
    ("publication_name",),
    ("wordpress", "site_url"),
    ("wordpress", "username"),
    ("wordpress", "application_password"),
]

# Publication names must be simple identifiers — no path components allowed.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_publication_name(name):
    if not _SAFE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid publication name {name!r}. "
            "Use only letters, numbers, hyphens, and underscores."
        )


def _get_nested(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _resolve_env(value):
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_key = value[2:-1]
        resolved = os.getenv(env_key)
        if resolved is None:
            raise ValueError(
                f"Environment variable {env_key!r} is not set. "
                f"Add it to your .env file or set it in the shell."
            )
        return resolved
    return value


def _resolve_env_recursive(obj):
    if isinstance(obj, dict):
        return {k: _resolve_env_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_recursive(i) for i in obj]
    if isinstance(obj, str):
        return _resolve_env(obj)
    return obj


def _load_yaml(path):
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Config file {path} contains invalid YAML:\n  {e}") from e


def load_user_config(config_dir="configs"):
    path = Path(config_dir) / "user.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"User config not found at {path}.\n"
            "Copy configs/user.example.yaml to configs/user.yaml and fill in your API keys.\n"
            "API keys can also be set as environment variables — see .env.example."
        )
    config = _load_yaml(path)
    if not isinstance(config, dict):
        raise ValueError(f"{path} is empty or not a valid YAML mapping.")
    config = _resolve_env_recursive(config)
    _validate_user_config(config)
    return config


def load_publication_config(publication_name, config_dir="configs"):
    validate_publication_name(publication_name)
    path = Path(config_dir) / f"{publication_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Publication config not found at {path}.\n"
            f"Create configs/{publication_name}.yaml based on configs/publication.example.yaml.\n"
            "Example configs are in configs/examples/."
        )
    config = _load_yaml(path)
    if not isinstance(config, dict):
        raise ValueError(f"{path} is empty or not a valid YAML mapping.")
    config = _resolve_env_recursive(config)
    _validate_publication_config(config, publication_name)
    return config


def _validate_user_config(config):
    missing = []
    for key_path in REQUIRED_USER_KEYS:
        # Gemini API key is not required when using Vertex AI (which uses google-auth instead).
        if key_path == ("api_keys", "gemini", "api_key"):
            gemini_model_raw = config.get("models", {}).get("gemini", {})
            if (
                isinstance(gemini_model_raw, dict)
                and gemini_model_raw.get("provider") == "vertex_ai"
            ):
                continue
        if _get_nested(config, *key_path) is None:
            missing.append(" -> ".join(key_path))
    if missing:
        raise ValueError(
            "User config is missing required fields:\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\nSee configs/user.example.yaml for the expected structure."
        )


def _validate_publication_config(config, publication_name):
    missing = []
    for key_path in REQUIRED_PUB_KEYS:
        if _get_nested(config, *key_path) is None:
            missing.append(" -> ".join(key_path))
    if missing:
        raise ValueError(
            f"Publication config '{publication_name}' is missing required fields:\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\nSee configs/publication.example.yaml for the expected structure."
        )


# ---------------------------------------------------------------------------
# Cost presets
# ---------------------------------------------------------------------------
#
# Presets select model variants, reasoning flags, and thoroughness as a bundle
# so users can dial cost vs coverage with a single pipeline.cost_preset value.
#
# Preset definitions are loaded from configs/presets.yaml at runtime so model
# names can be updated without editing Python.  The _COST_PRESETS dict below
# is used only as a fallback when the YAML file is missing or unreadable.
#
# Behavior when cost_preset is set:
#   - Sets thoroughness (unless the user also set thoroughness explicitly).
#   - Overrides model name and reasoning flags for each configured provider.
#   - Preserves user's infrastructure settings: provider, project, location,
#     credentials_file, endpoint, deployment, api_version, prompts.
#   - Skips providers the user has not configured (no API key / no models entry).
#   - Respects enabled: false set by the user.
#
_COST_PRESETS = {
    # economy: fastest and cheapest — mini/small models, no reasoning, one model per domain.
    # Good for workflow validation and quick structural checks.
    "economy": {
        "thoroughness": "standard",
        "models": {
            "openai": {"model": "gpt-5.4-mini"},
            "gemini": {"model": "gemini-2.5-flash"},
            "mistral": {"model": "mistral-small-latest"},
            "perplexity": {"model": "sonar"},
            "grok": {"enabled": False},
            "claude": {"enabled": False},
        },
    },
    # standard: solid quality, no reasoning overhead — flagship non-reasoning models.
    # Good for first-pass review of clean drafts.
    "standard": {
        "thoroughness": "standard",
        "models": {
            "openai": {"model": "gpt-5.4"},
            "gemini": {"model": "gemini-2.5-flash"},
            "mistral": {"model": "mistral-large-latest"},
            "perplexity": {"model": "sonar-pro"},
            "grok": {"model": "grok-4.3"},
            "claude": {"model": "claude-haiku-4-5-20251001"},
        },
    },
    # balanced: thorough coverage with light reasoning.
    # OpenAI: gpt-5.4 with low reasoning_effort — mid-tier model, light CoT.
    # Mistral: mistral-medium-3-5 replaced magistral-medium-latest (deprecated 5/22/2026,
    #   retires 7/31/2026). Only "high"/"none" supported — no effort flag at balanced tier.
    # Grok: reasoning is model-selection based, not parameter based. Use grok-4.20-0309-reasoning
    #   for CoT at balanced and above; reasoning_effort is not a supported Grok parameter.
    "balanced": {
        "thoroughness": "thorough",
        "models": {
            "openai": {"model": "gpt-5.4", "reasoning_effort": "low"},
            "gemini": {"model": "gemini-2.5-flash"},
            "mistral": {"model": "mistral-medium-3-5"},
            "perplexity": {"model": "sonar-reasoning-pro"},
            "grok": {"model": "grok-4.20-0309-reasoning"},
            "claude": {"model": "claude-sonnet-4-6", "effort": "medium"},
        },
    },
    # thorough: deep reasoning, thorough coverage.
    # OpenAI: gpt-5.4 with high reasoning_effort — mid-tier model, deep CoT.
    # Gemini: gemini-2.5-pro for more thorough grounded analysis.
    "thorough": {
        "thoroughness": "thorough",
        "models": {
            "openai": {
                "model": "gpt-5.4",
                "reasoning_effort": "high",
                "stream_read_timeout": 200,
            },
            "gemini": {"model": "gemini-2.5-pro"},
            "mistral": {
                "model": "mistral-medium-3-5",
                "reasoning_effort": "high",
                "stream_read_timeout": 200,
            },
            "perplexity": {"model": "sonar-reasoning-pro"},
            "grok": {"model": "grok-4.20-0309-reasoning"},
            "claude": {"model": "claude-opus-4-8", "effort": "high"},
        },
    },
    # maximum: highest capability, all domains, max reasoning.
    # OpenAI: gpt-5.5 is the highest capability model as of June 2026; xhigh reasoning.
    # Gemini: gemini-2.5-pro confirmed available in Vertex AI (gemini-3.x models return 404 there).
    # Grok: reasoning is model-selection based — grok-4.20-0309-reasoning is the CoT variant.
    "maximum": {
        "thoroughness": "maximum",
        "models": {
            "openai": {
                "model": "gpt-5.5",
                "reasoning_effort": "xhigh",
                "stream_read_timeout": 300,
            },
            "gemini": {"model": "gemini-2.5-pro", "thinking_budget": 16000},
            "mistral": {
                "model": "mistral-medium-3-5",
                "reasoning_effort": "high",
                "stream_read_timeout": 200,
            },
            "perplexity": {"model": "sonar-reasoning-pro"},
            "grok": {"model": "grok-4.20-0309-reasoning"},
            "claude": {"model": "claude-opus-4-8", "effort": "high"},
        },
    },
}

# Keys that identify provider infrastructure or per-model tuning that the user
# controls independently of which cost_preset is active.  These are preserved from
# user config even when cost_preset overrides model names and reasoning flags.
_INFRA_KEYS = frozenset(
    {
        "provider",
        "endpoint",
        "deployment",
        "api_version",
        "project",
        "location",
        "credentials_file",
        "prompts",
        "timeout_seconds",  # user-tuned HTTP timeout; preserved so long articles don't time out
    }
)


def _load_presets_from_yaml(config_dir=None):
    """Load cost presets from configs/presets.yaml; return None if missing.

    Resolves the path relative to this module (not the CWD) so presets load
    correctly regardless of where the pipeline is invoked from — matching the
    behavior of analysis/cost.py and model_registry.py.
    """
    import logging

    if config_dir is None:
        config_dir = Path(__file__).parent / "configs"
    yaml_path = Path(config_dir) / "presets.yaml"
    if not yaml_path.exists():
        return None
    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Validate top-level structure: each key must be a dict with a 'models' sub-dict.
        result = {}
        for preset_name, preset_body in data.items():
            if not isinstance(preset_body, dict):
                continue
            models_raw = preset_body.get("models", {})
            # Normalise: each provider entry must be a dict (YAML may give it as a mapping)
            models_normalised = {}
            for provider, cfg in models_raw.items():
                if isinstance(cfg, dict):
                    models_normalised[provider] = cfg
                elif cfg is None:
                    models_normalised[provider] = {}
            result[preset_name] = {
                "thoroughness": preset_body.get("thoroughness", "standard"),
                "models": models_normalised,
            }
        return result or None
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Could not load %s (%s) — using built-in preset defaults", yaml_path, exc
        )
        return None


def _apply_cost_preset(pipeline_cfg, models_raw):
    """Apply a cost_preset to pipeline and models config.

    Called in merge_configs() before model normalization.  Returns updated
    (pipeline_cfg, models_raw) dicts; the originals are not mutated.
    """
    preset_name = pipeline_cfg.get("cost_preset")
    if not preset_name:
        return pipeline_cfg, models_raw

    presets = _load_presets_from_yaml() or _COST_PRESETS
    preset = presets.get(preset_name)
    if not preset:
        valid = ", ".join(f"'{k}'" for k in presets)
        raise ValueError(f"Unknown cost_preset {preset_name!r}. Valid values: {valid}")

    new_pipeline = dict(pipeline_cfg)
    # Set thoroughness from preset only if user has not already set it explicitly.
    if "thoroughness" not in pipeline_cfg:
        new_pipeline["thoroughness"] = preset["thoroughness"]

    merged_models = dict(models_raw or {})
    preset_models = preset.get("models", {})

    for provider, preset_cfg in preset_models.items():
        user_val = merged_models.get(provider)
        # Skip providers the user has not configured at all.
        if user_val is None:
            continue

        user_dict = {"model": user_val} if isinstance(user_val, str) else dict(user_val)

        # Respect user's explicit enabled: false.
        if user_dict.get("enabled") is False:
            continue

        # Preset disabling a provider in this cost tier.
        if preset_cfg.get("enabled") is False:
            merged_models[provider] = {**user_dict, "enabled": False}
            continue

        # New config: start from preset values; overlay user's infra settings on top.
        new_cfg = dict(preset_cfg)
        for key in _INFRA_KEYS:
            if key in user_dict:
                new_cfg[key] = user_dict[key]

        merged_models[provider] = new_cfg

    return new_pipeline, merged_models


def _apply_preset_overrides(pipeline_cfg, models_raw):
    """Apply user's selective preset_overrides on top of the already-preset models.

    Called after ``_apply_cost_preset()``.  Lets users tweak individual settings
    from a preset without rebuilding the entire config:

    .. code-block:: yaml

        pipeline:
          cost_preset: balanced
          preset_overrides:
            openai:
              reasoning_effort: high    # bump from preset's "low"
            claude:
              model: claude-opus-4-8    # use opus instead of preset's sonnet
              effort: high

    Rules:
    - Only providers already in ``models_raw`` are affected — overrides for
      unconfigured providers are ignored.
    - Infrastructure keys (provider, project, location, …) can be set here
      but are usually already correct from user.yaml.
    - An override of ``enabled: false`` disables the provider for this run.
    """
    overrides = (pipeline_cfg or {}).get("preset_overrides")
    if not overrides:
        return models_raw

    if not isinstance(overrides, dict):
        raise ValueError(
            "pipeline.preset_overrides must be a mapping of provider names to "
            "their override settings.  See configs/user.example.yaml for examples."
        )

    merged = dict(models_raw or {})
    for provider, override_cfg in overrides.items():
        if provider not in merged:
            continue  # provider not configured; skip silently
        current = (
            dict(merged[provider])
            if isinstance(merged[provider], dict)
            else {"model": merged[provider]}
        )
        current.update(override_cfg)
        merged[provider] = current

    return merged


# Default provider for each adapter when user.yaml uses the simple string form.
_DEFAULT_PROVIDERS = {
    "gemini": "ai_studio",
    "openai": "openai",
    "mistral": "mistral",
    "grok": "grok",
    "claude": "anthropic",
    "perplexity": "perplexity",
}


def _normalize_model_configs(models_raw):
    """Normalize the ``models`` section of user.yaml to always be dicts.

    Two forms are accepted — both are valid, and simple strings remain the
    default so existing user.yaml files need no changes:

    Simple form (backward-compatible)::

        models:
          gemini: gemini-2.5-flash
          openai: gpt-4o

    Extended form (enables provider switching)::

        models:
          gemini:
            provider: vertex_ai
            model: gemini-2.5-flash
            project: my-gcp-project
            location: us-central1
          openai:
            provider: azure
            model: gpt-4o
            endpoint: https://my-resource.openai.azure.com
            deployment: my-gpt4o-deployment
            api_version: "2024-02-01"
          mistral:
            provider: azure
            model: mistral-large-latest
            endpoint: https://Mistral-Large-abc.eastus2.inference.ai.azure.com

    After normalization every entry is a dict with at least ``provider`` and
    ``model`` keys.  The rest of the system only has to deal with dicts.
    """
    if not models_raw or not isinstance(models_raw, dict):
        return {}
    result = {}
    for adapter, value in models_raw.items():
        if isinstance(value, str):
            result[adapter] = {
                "provider": _DEFAULT_PROVIDERS.get(adapter, adapter),
                "model": value,
            }
        elif isinstance(value, dict):
            normalized = dict(value)
            if "provider" not in normalized:
                normalized["provider"] = _DEFAULT_PROVIDERS.get(adapter, adapter)
            result[adapter] = normalized
        # None or unexpected type — omit silently; adapter will use its built-in default.
    return result


def merge_configs(user_config, pub_config):
    pipeline = user_config.get(
        "pipeline",
        {
            "parallel_review_calls": True,
            "retry_on_failure": True,
            "retry_delay_seconds": 10,
            "abort_if_all_provider_calls_fail": False,
            "task_timeout_seconds": 180,
            "thoroughness": "standard",
        },
    )
    models_raw = user_config.get("models", {})

    # Apply cost_preset when present — overrides model variants, reasoning flags,
    # and thoroughness while preserving user's provider infrastructure settings.
    pipeline, models_raw = _apply_cost_preset(pipeline, models_raw)

    # Apply selective preset_overrides on top — lets users adjust individual
    # fields of a preset without specifying the full config.
    models_raw = _apply_preset_overrides(pipeline, models_raw)

    return {
        "api_keys": user_config.get("api_keys", {}),
        "pipeline": pipeline,
        "delta": user_config.get(
            "delta",
            {
                "word_change_threshold_pct": 15,
                "claim_change_triggers_rerun": True,
                "structure_change_triggers_rerun": True,
            },
        ),
        "ensemble": user_config.get("ensemble", {}),
        "models": _normalize_model_configs(models_raw),
        "publication": pub_config,
    }
