import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

REQUIRED_USER_KEYS = [
    ("api_keys", "languagetool", "username"),
    ("api_keys", "languagetool", "api_key"),
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
        if _get_nested(config, *key_path) is None:
            missing.append(" -> ".join(key_path))
    if missing:
        raise ValueError(
            "User config is missing required fields:\n" +
            "\n".join(f"  {m}" for m in missing) +
            "\nSee configs/user.example.yaml for the expected structure."
        )


def _validate_publication_config(config, publication_name):
    missing = []
    for key_path in REQUIRED_PUB_KEYS:
        if _get_nested(config, *key_path) is None:
            missing.append(" -> ".join(key_path))
    if missing:
        raise ValueError(
            f"Publication config '{publication_name}' is missing required fields:\n" +
            "\n".join(f"  {m}" for m in missing) +
            "\nSee configs/publication.example.yaml for the expected structure."
        )


def merge_configs(user_config, pub_config):
    return {
        "api_keys": user_config.get("api_keys", {}),
        "pipeline": user_config.get("pipeline", {
            "parallel_review_calls": True,
            "retry_on_failure": True,
            "retry_delay_seconds": 10,
            "abort_if_all_provider_calls_fail": False,
            "task_timeout_seconds": 180,
        }),
        "delta": user_config.get("delta", {
            "word_change_threshold_pct": 15,
            "claim_change_triggers_rerun": True,
            "structure_change_triggers_rerun": True,
        }),
        "models": user_config.get("models", {}),
        "publication": pub_config,
    }
