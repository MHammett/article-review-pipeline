"""
First-run setup: create configs/ directory and copy example templates.

Usage:
    uv run ci-setup
    uv run ci-setup --publication dnacom
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


_EXAMPLE_DIR = Path(__file__).parent / "configs"


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


_TEMPLATE_DIR = Path(__file__).parent / "handoff_templates"

#: Copied into the working tree so they can be edited. The packaged copies live
#: inside site-packages, which is neither guessable nor a sane thing to edit —
#: the README used to name a path the user could not reasonably find or change.
_WORKING_TEMPLATES = (
    "draft_submission.template.md",
    "metadata_only.md",
    "publication.md",
)


def _copy_handoff_templates(dest_dir: Path) -> None:
    """Put the fill-in templates somewhere the user can actually edit them."""
    if not _TEMPLATE_DIR.is_dir():  # pragma: no cover - broken install
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCopying handoff templates into {dest_dir}/...")
    for name in _WORKING_TEMPLATES:
        src = _TEMPLATE_DIR / name
        if src.exists():
            _copy_if_missing(src, dest_dir / name, f"{dest_dir}/{name}")
    example = _TEMPLATE_DIR / "examples" / "draft_submission.filled-example.md"
    if example.exists():
        _copy_if_missing(
            example,
            dest_dir / "draft_submission.filled-example.md",
            f"{dest_dir}/draft_submission.filled-example.md",
        )


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

    _copy_handoff_templates(Path("handoff_templates"))

    publication_name = _prompt_publication_name(args.publication)
    pub_yaml = configs_dir / f"{publication_name}.yaml"
    pub_example = _EXAMPLE_DIR / "publication.example.yaml"
    copied_pub = _copy_if_missing(
        pub_example, pub_yaml, f"configs/{publication_name}.yaml"
    )

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
        print(
            f"\n{step}. configs/user.yaml already exists — verify your API keys are set."
        )
        step += 1

    if copied_pub:
        print(
            f"\n{step}. Fill in your publication profile in configs/{publication_name}.yaml"
        )
        print(
            "   Required: publication_description, audience, style_profile, wordpress.*"
        )
        print("   See docs/CONFIGURATION.md for field reference.")
        step += 1

    print(f"\n{step}. Verify all credentials work:")
    print(f"   uv run ci-check --publication {publication_name}")
    step += 1

    print(f"\n{step}. Fill in handoff_templates/draft_submission.template.md")
    print("   A worked example sits beside it as draft_submission.filled-example.md.")
    step += 1

    # One line, console-script form. A trailing backslash is a bash continuation
    # that splits the command on Windows cmd.exe — the documented primary
    # platform — and PR #51 standardised the docs on console scripts. Both are
    # now enforced against printed strings by test_docs_current.py.
    #
    # The --draft path names the template ci-setup just copied, rather than a
    # placeholder: a printed command should point at a file that exists.
    print(f"\n{step}. Run the pipeline:")
    print(
        "   uv run ci-review --draft handoff_templates/draft_submission.template.md"
        f" --publication {publication_name}"
    )
    print()


if __name__ == "__main__":
    main()
