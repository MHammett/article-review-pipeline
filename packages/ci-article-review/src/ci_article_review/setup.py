"""
First-run setup: create configs/ directory and copy example templates.

Usage:
    uv run python -m ci_article_review.setup
    uv run python -m ci_article_review.setup --publication dnacom
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


_EXAMPLE_DIR = Path(__file__).parent / "configs"
_ENV_EXAMPLE = Path(__file__).parents[5] / ".env.example"  # repo root


def _repo_root() -> Path:
    """Walk up from this file to find the repo root (contains pyproject.toml + uv.lock)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "uv.lock").exists():
            return parent
    return Path.cwd()


def _check_python() -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        print(f"ERROR: Python 3.10+ required (found {major}.{minor})")
        sys.exit(1)
    print(f"  Python {major}.{minor}  OK")


def _check_uv() -> bool:
    if shutil.which("uv") is None:
        print("  uv         NOT FOUND — install from https://docs.astral.sh/uv/")
        return False
    result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
    print(f"  uv         {result.stdout.strip()}  OK")
    return True


def _sync_deps(repo_root: Path) -> None:
    print("\nInstalling / verifying dependencies (uv sync)...")
    result = subprocess.run(["uv", "sync"], cwd=repo_root)
    if result.returncode != 0:
        print("ERROR: uv sync failed — see output above.")
        sys.exit(1)
    print("  Dependencies OK")


def _ensure_configs_dir(configs_dir: Path) -> None:
    if not configs_dir.exists():
        configs_dir.mkdir(parents=True)
        print(f"  Created {configs_dir}/")
    else:
        print(f"  {configs_dir}/  already exists")


def _copy_if_missing(src: Path, dst: Path, label: str) -> bool:
    """Copy src → dst if dst does not exist. Returns True if copied."""
    if dst.exists():
        print(f"  {label}  already exists — skipped")
        return False
    shutil.copy2(src, dst)
    print(f"  Created {dst}")
    return True


def _validate_publication_name(name: str) -> bool:
    return bool(re.match(r"^[a-z0-9][a-z0-9_-]*$", name))


def _prompt_publication_name(provided: str | None) -> str:
    if provided:
        if not _validate_publication_name(provided):
            print(f"ERROR: '{provided}' is not a valid publication name.")
            print("  Use lowercase letters, digits, hyphens, and underscores only.")
            sys.exit(1)
        return provided

    print("\nEnter a short identifier for your publication.")
    print("  Examples: dnacom, myblog, tech-review")
    print("  Rules: lowercase letters, digits, hyphens, underscores; no spaces.")
    while True:
        name = input("  Publication name: ").strip()
        if _validate_publication_name(name):
            return name
        print("  Invalid — try again (lowercase, no spaces).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="First-run setup: scaffold configs/ and verify dependencies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--publication",
        metavar="NAME",
        help="Publication identifier (e.g. dnacom). Prompted interactively if omitted.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        metavar="DIR",
        help="Where to write config files (default: configs/)",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip 'uv sync' dependency install step.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    configs_dir = Path(args.config_dir)

    print("=" * 60)
    print("Content-Intelligence — first-run setup")
    print("=" * 60)

    print("\nChecking prerequisites...")
    _check_python()
    has_uv = _check_uv()

    if not has_uv:
        print("\nInstall uv first, then re-run this script.")
        sys.exit(1)

    if not args.skip_sync:
        _sync_deps(repo_root)

    print(f"\nScaffolding {configs_dir}/...")
    _ensure_configs_dir(configs_dir)

    user_yaml = configs_dir / "user.yaml"
    user_example = _EXAMPLE_DIR / "user.example.yaml"
    copied_user = _copy_if_missing(user_example, user_yaml, "configs/user.yaml")

    env_file = repo_root / ".env"
    env_example = repo_root / ".env.example"
    if env_example.exists():
        _copy_if_missing(env_example, env_file, ".env")

    publication_name = _prompt_publication_name(args.publication)
    pub_yaml = configs_dir / f"{publication_name}.yaml"
    pub_example = _EXAMPLE_DIR / "publication.example.yaml"
    copied_pub = _copy_if_missing(pub_example, pub_yaml, f"configs/{publication_name}.yaml")

    print("\n" + "=" * 60)
    print("Setup complete. Next steps:")
    print("=" * 60)

    step = 1

    if copied_user:
        print(f"\n{step}. Fill in API keys in configs/user.yaml")
        print("   Required: openai, gemini, mistral")
        print("   Optional: perplexity, grok, claude, languagetool")
        print("   See docs/PROVIDERS.md for account setup instructions.")
        step += 1
    else:
        print(f"\n{step}. configs/user.yaml already exists — verify your API keys are set.")
        step += 1

    if copied_pub:
        print(f"\n{step}. Fill in your publication profile in configs/{publication_name}.yaml")
        print("   Required: publication_description, audience, voice_profile, wordpress.*")
        print("   See docs/CONFIGURATION.md for field reference.")
        step += 1

    print(f"\n{step}. Verify all credentials work:")
    print(f"   uv run python -m ci_article_review.check --publication {publication_name}")
    step += 1

    print(f"\n{step}. Run the pipeline:")
    print(f"   uv run python -m ci_article_review.pipeline \\")
    print(f"       --draft path/to/handoff.md --publication {publication_name}")
    print()


if __name__ == "__main__":
    main()
