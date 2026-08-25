import re
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from ci_core.config_helpers import (
    PackagedConfigError,
    load_packaged_yaml,
    load_yaml as _load_yaml,
    normalize_model_configs as _normalize_model_configs,
    resolve_env_recursive as _resolve_env_recursive,
)
from ci_core.env_provenance import (
    provenance as _env_var_provenance,
    shadowed_mismatches as _shadowed_mismatches,
    snapshot as _snapshot_dotenv,
)
from ci_core.redact import mask_secret

# Resolved once so the snapshot below and the actual load agree on exactly
# which file (if any) is in play -- see ci_core.env_provenance for why this
# matters: the snapshot has to be taken before load_dotenv() touches the OS
# environment.
_DOTENV_PATH = find_dotenv()
_ENV_SNAPSHOT = _snapshot_dotenv(_DOTENV_PATH)
load_dotenv(_DOTENV_PATH or None)


def warn_env_shadowing(env_snapshot=None):
    """Print a loud, unprompted warning for every OS env var that's silently
    overriding a *different* value from the .env file.

    No flag needed -- a silent, load-bearing config mismatch like this costs
    real money if nobody notices (see this module's incident history in
    ci_core.env_provenance). Called once below against the real environment;
    tests pass ``env_snapshot`` to exercise specific scenarios without
    touching process state.
    """
    snap = _ENV_SNAPSHOT if env_snapshot is None else env_snapshot
    for m in _shadowed_mismatches(snap):
        print(
            f"WARNING: {m['var_name']} is set in the OS environment, and its "
            "value differs from what's in your .env file. The .env value is "
            "being silently ignored (python-dotenv's load_dotenv() never "
            "overrides an existing OS environment variable). If you edited "
            ".env expecting it to take effect, it will not until the OS-level "
            "variable is unset. Run `uv run ci-check --publication <name> "
            "--show-keys` to see exactly which value each provider will use."
        )


warn_env_shadowing()

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


def _iter_leaf_strings(node, path=()):
    """Yield (field_path_tuple, value) for every leaf under a nested dict."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _iter_leaf_strings(v, path + (k,))
    elif node is not None:
        yield path, node


def describe_api_key_sources(config_dir="configs", env_snapshot=None):
    """Masked, provenance-annotated view of every ``api_keys.*`` entry in
    ``user.yaml`` -- what a run would actually send as credentials, and where
    each value came from.

    Reads the file directly rather than through ``load_user_config()``, so it
    still works when required keys are missing -- the point is diagnosing a
    key that looks wrong *before* validation gets a chance to complain about
    something else. Returns a list of dicts in file order, one per leaf under
    ``api_keys``:

      provider      -- e.g. "openai"
      field         -- e.g. "api_key" ("username" for languagetool, etc.)
      var_name      -- the ``${VAR_NAME}`` this entry references, or None if
                        the value is written literally in user.yaml
      masked_value  -- the resolved value, masked for display (see
                        ci_core.redact.mask_secret)
      source        -- "env" (came from the .env file, unshadowed), "os_env"
                        (a pre-existing OS environment variable -- either
                        there's no conflicting .env entry, or there is and it
                        happens to match), "os_env_shadowing_env_file" (the
                        loud case: a pre-existing OS env var is silently
                        overriding a *different* .env value), "literal"
                        (hardcoded directly in user.yaml, not from an env
                        var), or "unset" (references an env var that isn't
                        set anywhere).
    """
    snap = _ENV_SNAPSHOT if env_snapshot is None else env_snapshot
    path = Path(config_dir) / "user.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"User config not found at {path}.\n"
            "Copy configs/user.example.yaml to configs/user.yaml and fill in your API keys.\n"
            "API keys can also be set as environment variables — see .env.example."
        )
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is empty or not a valid YAML mapping.")

    entries = []
    for field_path, raw_value in _iter_leaf_strings(raw.get("api_keys") or {}):
        provider, field = field_path[0], ".".join(field_path[1:])
        if (
            isinstance(raw_value, str)
            and raw_value.startswith("${")
            and raw_value.endswith("}")
        ):
            var_name = raw_value[2:-1]
            prov = _env_var_provenance(snap, var_name)
            active = prov["active_value"]
            if active is None:
                source = "unset"
            elif prov["mismatched"]:
                source = "os_env_shadowing_env_file"
            elif prov["shadowed_by_os_env"]:
                source = "os_env"
            elif prov["in_dotenv_file"]:
                source = "env"
            else:
                source = "os_env"
            entries.append(
                {
                    "provider": provider,
                    "field": field,
                    "var_name": var_name,
                    "masked_value": mask_secret(active) if active else "(NOT SET)",
                    "source": source,
                }
            )
        else:
            entries.append(
                {
                    "provider": provider,
                    "field": field,
                    "var_name": None,
                    "masked_value": mask_secret(raw_value),
                    "source": "literal",
                }
            )
    return entries


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
# names can be updated without editing Python. The YAML is the single source of
# truth — there is no duplicate table here (audit finding 14).
#
# Behavior when cost_preset is set:
#   - Sets thoroughness (unless the user also set thoroughness explicitly).
#   - Overrides model name and reasoning flags for each configured provider.
#   - Preserves user's infrastructure settings: provider, project, location,
#     credentials_file, endpoint, deployment, api_version, prompts, web_search.
#   - Skips providers the user has not configured (no API key / no models entry).
#   - Respects enabled: false set by the user.
#
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
        # Which domains may run a live web search. Same category as `prompts`:
        # the user decides what a model is allowed to do, the preset decides how
        # expensive a variant it runs. Without this the preset rebuilt the model
        # config and dropped it, so `web_search` set alongside any cost_preset —
        # which is every non-default configuration — silently never took effect.
        "web_search",
    }
)


def _load_presets_from_yaml(config_dir=None):
    """Load cost presets from the packaged configs/presets.yaml.

    Resolves the path relative to this module (not the CWD) so presets load
    correctly regardless of where the pipeline is invoked from — matching the
    behavior of ci_core/llm/cost.py and model_registry.py.

    Raises PackagedConfigError if the file is missing or malformed. There is no
    hardcoded fallback: the duplicate it replaced had to be edited in lockstep
    with the YAML, and in the only state it could have fired — a broken install
    — quietly running a stale preset is worse than saying so.
    """

    if config_dir is None:
        config_dir = Path(__file__).parent / "configs"
    yaml_path = Path(config_dir) / "presets.yaml"
    try:
        data = load_packaged_yaml(yaml_path)
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
        if not result:
            raise PackagedConfigError(
                f"{yaml_path}: contains no usable preset definitions"
            )
        return result
    except PackagedConfigError:
        # A missing or malformed packaged file is a broken install, not a
        # runtime condition to survive — quietly running a stale preset gives
        # the user wrong models and wrong costs with no indication why.
        raise
    except Exception as exc:
        raise PackagedConfigError(
            f"{yaml_path}: could not be parsed as cost presets ({exc})"
        ) from exc


def _apply_cost_preset(pipeline_cfg, models_raw, user_set=None):
    """Apply a cost_preset to pipeline and models config.

    Called in merge_configs() before model normalization.  Returns updated
    (pipeline_cfg, models_raw) dicts; the originals are not mutated.
    """
    preset_name = pipeline_cfg.get("cost_preset")
    if not preset_name:
        return pipeline_cfg, models_raw

    presets = _load_presets_from_yaml()
    preset = presets.get(preset_name)
    if not preset:
        valid = ", ".join(f"'{k}'" for k in presets)
        raise ValueError(f"Unknown cost_preset {preset_name!r}. Valid values: {valid}")

    new_pipeline = dict(pipeline_cfg)
    # Set thoroughness from the preset unless the user asked for one by name.
    #
    # "Did the user set it" cannot be answered from ``pipeline_cfg``, because by
    # the time this runs the defaults have been merged in and `thoroughness` is
    # always present. Checking the merged dict is what broke `cost_preset`
    # between #105 and here: every preset picked its models correctly and then
    # ran a `standard`-sized ensemble, so `maximum` made 7 calls where it should
    # make 30 — maximum-model prices for a fraction of the ensemble, silently.
    # ``user_set`` is the raw pipeline block, before defaults.
    asked_for = pipeline_cfg if user_set is None else user_set
    if "thoroughness" not in asked_for:
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


#: Pipeline defaults, applied key by key rather than as a whole-block fallback.
#:
#: These used to be the second argument to ``user_config.get("pipeline", ...)``,
#: which only applies when the key is absent entirely. Any *partial* block
#: therefore discarded all of them: ``pipeline: {cost_preset: maximum}`` — an
#: entirely reasonable thing to write — left ``task_timeout_seconds`` as ``None``,
#: which is a ``TypeError`` inside ``timeout_model.compute_timeout``, and
#: ``retry_on_failure`` as ``None``, silently disabling retries.
#:
#: ``task_timeout_seconds`` is 180 here to preserve the previous no-config
#: behaviour. Note that ``user.example.yaml`` ships **1100**, and anyone starting
#: from ``ci-setup`` gets that; this value only governs a config that omits the
#: key, where failing fast is the safer default.
PIPELINE_DEFAULTS = {
    "parallel_review_calls": True,
    "retry_on_failure": True,
    "retry_delay_seconds": 10,
    "abort_if_all_provider_calls_fail": False,
    "task_timeout_seconds": 180,
    "thoroughness": "standard",
}


def merge_configs(user_config, pub_config):
    # Per-key, so a partial block keeps the defaults it did not mention.
    user_pipeline = user_config.get("pipeline") or {}
    pipeline = {**PIPELINE_DEFAULTS, **user_pipeline}
    models_raw = user_config.get("models", {})

    # Apply cost_preset when present — overrides model variants, reasoning flags,
    # and thoroughness while preserving user's provider infrastructure settings.
    pipeline, models_raw = _apply_cost_preset(
        pipeline, models_raw, user_set=user_pipeline
    )

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
