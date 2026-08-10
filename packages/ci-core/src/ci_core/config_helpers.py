"""Shared YAML/env config helpers.

These three functions are the supported cross-package contract for loading the
YAML config files both ``ci-article-review`` and ``ci-style-profile`` read.
They used to live as private helpers in ``ci_article_review.config_loader``
(``_load_yaml``, ``_resolve_env_recursive``, ``_normalize_model_configs``) and
were imported across the package boundary by their private names; moving them
here gives them a real public API.

The app-specific parts of config loading — required-key validation, publication
configs, cost presets — deliberately stay in ``ci_article_review.config_loader``.
"""

import os

import yaml

# Default provider for each adapter when the models section of user.yaml uses
# the simple string form.
DEFAULT_PROVIDERS = {
    "gemini": "ai_studio",
    "openai": "openai",
    "mistral": "mistral",
    "grok": "grok",
    "claude": "anthropic",
    "perplexity": "perplexity",
}


def resolve_env(value):
    """Resolve a single ``${ENV_VAR}`` placeholder to its environment value.

    Non-placeholder values pass through untouched. Raises ``ValueError`` when
    the referenced variable is not set, so a missing credential fails loudly at
    config-load time rather than as an opaque 401 mid-run.
    """
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


def resolve_env_recursive(obj):
    """Apply :func:`resolve_env` to every string in a nested dict/list tree."""
    if isinstance(obj, dict):
        return {k: resolve_env_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_env_recursive(i) for i in obj]
    if isinstance(obj, str):
        return resolve_env(obj)
    return obj


def load_yaml(path):
    """Read a YAML file, raising ``ValueError`` with the path on a parse error."""
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Config file {path} contains invalid YAML:\n  {e}") from e


def normalize_model_configs(models_raw):
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
                "provider": DEFAULT_PROVIDERS.get(adapter, adapter),
                "model": value,
            }
        elif isinstance(value, dict):
            normalized = dict(value)
            if "provider" not in normalized:
                normalized["provider"] = DEFAULT_PROVIDERS.get(adapter, adapter)
            result[adapter] = normalized
        # None or unexpected type — omit silently; adapter will use its built-in default.
    return result
