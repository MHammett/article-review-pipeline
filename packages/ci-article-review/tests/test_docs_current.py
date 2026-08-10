"""Tests that fail when the documentation drifts out of sync with the code.

PR #44 corrected documentation that had gone stale across PRs #21-43: CLI flags
that shipped but never reached the README's options table, modules missing from
the project-structure tree, a citation adapter count frozen at four while the
code grew to ten, and doc links pointing at paths that moved during the monorepo
migration. Every one of those was mechanically detectable. These tests detect
them, in the PR that introduces them.

Each failure message names the specific flag/module/adapter/link and the file
that needs updating — the point is that whoever trips one can fix it in a minute.
"""

import re
from pathlib import Path

import pytest


def _find_repo_root():
    """Locate the workspace root by walking up from this file.

    These tests read repo files, so they must not depend on pytest's working
    directory (`uv run pytest packages/` and `cd packages/ci-article-review &&
    pytest` both have to work).
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "packages").is_dir() and (candidate / "README.md").is_file():
            return candidate
    raise RuntimeError(
        "Could not locate the content-intelligence repo root above "
        f"{Path(__file__).resolve()} — expected an ancestor containing both "
        "packages/ and README.md."
    )


REPO_ROOT = _find_repo_root()
README = REPO_ROOT / "README.md"
CITATIONS_DOC = REPO_ROOT / "docs" / "CITATIONS.md"
PUBLICATION_EXAMPLE = (
    REPO_ROOT
    / "packages"
    / "ci-article-review"
    / "src"
    / "ci_article_review"
    / "configs"
    / "publication.example.yaml"
)


def _mentions(text, token):
    """True if `token` appears in `text` as a standalone token.

    Plain substring matching gives false passes: "config.py" is a substring of
    "logging_config.py", and "--publish" of "--publish-live". Requiring a
    non-identifier character on each side avoids both. A leading "/" is allowed
    because the tree writes some entries path-prefixed ("grammar/languagetool.py").
    """
    return re.search(rf"(?<![\w.-]){re.escape(token)}(?![\w-])", text) is not None


# ---------------------------------------------------------------------------
# 1. CLI flag coverage
# ---------------------------------------------------------------------------

# Every CLI whose flags the README documents. Adding a CLI here is a one-line
# change — voice_pattern_report was added this way when PR #46 restored it.
CLI_MODULES = [
    "ci_article_review.pipeline",
    "ci_article_review.history_analytics",
    "ci_article_review.voice_pattern_report",
]


def _long_flags(module_name):
    """Long-form flags of a module's parser, introspected rather than scraped.

    Walking parser._actions picks up flags inside mutually exclusive groups too,
    which is where --draft/--raw-draft/--url/--publish live.
    """
    import importlib

    module = importlib.import_module(module_name)
    parser = module.build_parser()
    flags = set()
    for action in parser._actions:
        for option in action.option_strings:
            if option.startswith("--"):
                flags.add(option)
    flags.discard("--help")
    return sorted(flags)


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_every_cli_flag_is_documented_in_readme(module_name):
    readme = README.read_text(encoding="utf-8")
    flags = _long_flags(module_name)
    assert flags, f"{module_name}.build_parser() exposed no long-form flags"

    missing = [flag for flag in flags if not _mentions(readme, flag)]
    assert not missing, (
        f"{module_name} defines flags that {README.name} never mentions: "
        f"{', '.join(missing)}. Add a row for each to the relevant options "
        f"table in {README}."
    )


# ---------------------------------------------------------------------------
# 2. Module tree coverage
# ---------------------------------------------------------------------------

# Directories the README's project-structure tree deliberately summarizes as a
# single line instead of listing file by file. Each entry is a real editorial
# decision, not a way to quiet the test — the tree describes what a directory
# holds ("10 adapters: census, crossref, ...") and listing every member would
# bury the modules a reader actually needs to find.
SUMMARIZED_DIRS = {
    # "sources/  10 adapters: census, crossref, eia, epa, ferc, fhwa, fred, ..."
    "packages/ci-article-review/src/ci_article_review/adapters/citation/sources",
    # "adapters/  the six streaming provider adapters + call_provider/call_text dispatch"
    "packages/ci-core/src/ci_core/llm/adapters",
    # "collectors/  wordpress, gmail, outlook365, twitter, textfiles, custom/"
    "packages/ci-style-profile/src/ci_style_profile/collectors",
}


def _project_structure_tree():
    """The fenced code block under the README's '## Project structure' heading.

    Scoped deliberately: a module name mentioned in prose elsewhere in the
    README does not mean it appears in the tree.
    """
    readme = README.read_text(encoding="utf-8")
    heading = re.search(r"^## Project structure$", readme, re.MULTILINE)
    assert heading, f"{README} no longer has a '## Project structure' heading"
    fence = re.search(r"^```\n(.*?)^```", readme[heading.end() :], re.MULTILINE | re.S)
    assert fence, (
        f"No fenced code block found under '## Project structure' in {README} — "
        "the tree this test checks against is gone."
    )
    return fence.group(1)


def _source_modules():
    for path in sorted(REPO_ROOT.glob("packages/*/src/**/*.py")):
        parts = path.relative_to(REPO_ROOT).parts
        if path.name == "__init__.py":
            continue
        if "__pycache__" in parts or "tests" in parts:
            continue
        yield path


def test_every_source_module_appears_in_readme_tree():
    tree = _project_structure_tree()

    missing = []
    for path in _source_modules():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.rsplit("/", 1)[0] in SUMMARIZED_DIRS:
            continue
        if not _mentions(tree, path.name):
            missing.append(rel)

    assert not missing, (
        "These modules are missing from the project-structure tree in "
        f"{README}:\n  " + "\n  ".join(missing) + "\n"
        "Add a line for each under '## Project structure' (or, if the module's "
        "directory is meant to be summarized rather than itemized, add that "
        "directory to SUMMARIZED_DIRS in this file with a comment saying why)."
    )


def test_summarized_dirs_all_exist():
    """A stale allowlist entry would silently exempt nothing — or hide a rename."""
    missing = [d for d in sorted(SUMMARIZED_DIRS) if not (REPO_ROOT / d).is_dir()]
    assert not missing, (
        "SUMMARIZED_DIRS in this file lists directories that no longer exist: "
        f"{', '.join(missing)}. Update the allowlist to match the tree."
    )


# ---------------------------------------------------------------------------
# 3. Citation adapter coverage
# ---------------------------------------------------------------------------


def _documented_adapters_in_citations_doc():
    """Adapter names from the '| Adapter | Kind | Reaches |' table."""
    text = CITATIONS_DOC.read_text(encoding="utf-8")
    table = re.search(
        r"^\| Adapter \| Kind \| Reaches \|\n\|[-| ]+\|\n((?:\|.*\n)+)",
        text,
        re.MULTILINE,
    )
    assert table, (
        f"Could not find the '| Adapter | Kind | Reaches |' table in "
        f"{CITATIONS_DOC} — this test cross-checks it against "
        "resolver.ADAPTER_MAP."
    )
    return {
        row.split("|")[1].strip().strip("`")
        for row in table.group(1).strip().splitlines()
    }


def _documented_adapters_in_publication_example():
    """Adapter names from the '# Available adapters: ...' comment."""
    text = PUBLICATION_EXAMPLE.read_text(encoding="utf-8")
    comment = re.search(r"^\s*#\s*Available adapters:\s*(.+)$", text, re.MULTILINE)
    assert comment, (
        f"Could not find the '# Available adapters:' comment in "
        f"{PUBLICATION_EXAMPLE} — this test cross-checks it against "
        "resolver.ADAPTER_MAP."
    )
    return {name.strip() for name in comment.group(1).split(",") if name.strip()}


def _adapter_map():
    from ci_article_review.adapters.citation import resolver

    return set(resolver.ADAPTER_MAP)


@pytest.mark.parametrize(
    "doc_path, reader",
    [
        (CITATIONS_DOC, _documented_adapters_in_citations_doc),
        (PUBLICATION_EXAMPLE, _documented_adapters_in_publication_example),
    ],
    ids=["docs/CITATIONS.md", "configs/publication.example.yaml"],
)
def test_citation_adapters_are_documented(doc_path, reader):
    registered = _adapter_map()
    documented = reader()

    undocumented = sorted(registered - documented)
    phantom = sorted(documented - registered)

    assert not undocumented, (
        f"Citation adapters in resolver.ADAPTER_MAP that {doc_path} does not "
        f"list: {', '.join(undocumented)}. Add them to {doc_path}."
    )
    assert not phantom, (
        f"{doc_path} lists citation adapters that resolver.ADAPTER_MAP does not "
        f"define: {', '.join(phantom)}. Remove them from {doc_path}."
    )


def test_readme_states_the_real_adapter_count():
    """The README's sources/ line carries a count and a name list; both drift."""
    tree = _project_structure_tree()
    registered = _adapter_map()

    line = re.search(
        r"sources/\s+(\d+) adapters: ([^\n│]*(?:\n[^\n│]*│?[^\n]*)?)", tree
    )
    assert line, (
        f"Could not find the 'sources/  N adapters: ...' line in the "
        f"project-structure tree in {README}."
    )
    count = int(line.group(1))
    assert count == len(registered), (
        f"{README}'s project-structure tree says '{count} adapters' but "
        f"resolver.ADAPTER_MAP defines {len(registered)}. Update the tree."
    )

    listed = {name.strip(" ,\n") for name in re.split(r"[,\s]+", line.group(2))}
    missing = sorted(registered - listed)
    assert not missing, (
        f"{README}'s project-structure tree omits these citation adapters from "
        f"its sources/ line: {', '.join(missing)}."
    )


# ---------------------------------------------------------------------------
# 4. Relative link validity
# ---------------------------------------------------------------------------

_FENCED_CODE = re.compile(r"^```.*?^```", re.MULTILINE | re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_MD_LINK = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\s]+)\)")


def _markdown_files():
    for path in sorted(REPO_ROOT.glob("**/*.md")):
        parts = path.relative_to(REPO_ROOT).parts
        if any(p in {".git", ".venv", "node_modules", "__pycache__"} for p in parts):
            continue
        yield path


def _relative_link_targets(text):
    """Link targets in `text`, excluding anything inside code.

    Code spans and fences are stripped first rather than skipped file by file:
    ci-style-profile/PLAN.md documents its own markdown handling with a literal
    `` `[text](url)` `` example, which is prose about a link, not a link.
    Stripping code handles that precisely and generalizes to any future
    occurrence.
    """
    text = _FENCED_CODE.sub("", text)
    text = _INLINE_CODE.sub("", text)
    for target in _MD_LINK.findall(text):
        if re.match(r"^(?:[a-z][a-z0-9+.-]*:|//|#)", target, re.IGNORECASE):
            continue
        yield target


def test_all_relative_markdown_links_resolve():
    broken = []
    for md_path in _markdown_files():
        text = md_path.read_text(encoding="utf-8")
        for target in _relative_link_targets(text):
            path_part = target.split("#", 1)[0]
            if not path_part:
                continue  # pure in-page anchor
            resolved = (md_path.parent / path_part).resolve()
            if not resolved.exists():
                broken.append(
                    f"{md_path.relative_to(REPO_ROOT).as_posix()} → {target} "
                    f"(no such path: {resolved})"
                )

    assert not broken, "Broken relative markdown links:\n  " + "\n  ".join(broken)


# ---------------------------------------------------------------------------
# 5. CLI invocation style
# ---------------------------------------------------------------------------

# The docs invoke the CLIs as console scripts (`uv run ci-check`), not as
# modules (`uv run python -m ci_article_review.check`). Both run, but only the
# console script reports its own name in argparse's usage line — `python -m`
# prints "usage: check.py", a path that has not been runnable since the src/
# layout landed. Setting prog= to compensate would bake "uv run" into --help,
# which is wrong for anyone inside an activated venv, so the docs move instead.

_CONSOLE_SCRIPT_REF = re.compile(r"\buv run (ci-[a-z0-9-]+)")
_MODULE_FORM_REF = re.compile(r"\bpython -m (ci_[a-z_]+)\.([a-z_]+)")
_SCRIPTS_SECTION = re.compile(r"^\[project\.scripts\]\s*$(.*?)(?=^\[|\Z)", re.M | re.S)
_SCRIPT_ENTRY = re.compile(r"^([\w-]+)\s*=\s*[\"']([\w.]+):(\w+)[\"']", re.M)


def _declared_console_scripts():
    """Map console-script name -> "module:function", across every package.

    Parsed with a regex rather than tomllib because tomllib is 3.11+ and this
    project supports 3.10.
    """
    scripts = {}
    for pyproject in sorted(REPO_ROOT.glob("packages/*/pyproject.toml")):
        section = _SCRIPTS_SECTION.search(pyproject.read_text(encoding="utf-8"))
        if not section:
            continue
        for name, module, func in _SCRIPT_ENTRY.findall(section.group(1)):
            scripts[name] = f"{module}:{func}"
    return scripts


def test_documented_console_scripts_are_declared():
    """Every `uv run ci-foo` in the docs resolves to a real entry point.

    Without this, renaming a script in pyproject.toml leaves the docs quietly
    pointing at a command that no longer exists.
    """
    declared = _declared_console_scripts()
    assert declared, (
        "No [project.scripts] entries found under packages/*/pyproject.toml — "
        "the parser in this test is probably out of date."
    )

    unknown = []
    for md_path in _markdown_files():
        text = md_path.read_text(encoding="utf-8")
        for name in sorted(set(_CONSOLE_SCRIPT_REF.findall(text))):
            if name not in declared:
                unknown.append(f"{md_path.relative_to(REPO_ROOT).as_posix()} → {name}")

    assert not unknown, (
        "Docs invoke console scripts that no package declares:\n  "
        + "\n  ".join(unknown)
        + "\n\nDeclared: "
        + ", ".join(sorted(declared))
    )


def test_docs_do_not_reintroduce_the_module_invocation_form():
    """Docs use `uv run ci-foo`, not `uv run python -m ci_article_review.foo`.

    Only flags modules that actually have a console script — `probe` has a
    parser but no entry point, so documenting it as a module would be correct.
    """
    by_target = {target: name for name, target in _declared_console_scripts().items()}

    offenders = []
    for md_path in _markdown_files():
        text = md_path.read_text(encoding="utf-8")
        for package, module in set(_MODULE_FORM_REF.findall(text)):
            script = by_target.get(f"{package}.{module}:main")
            if script:
                offenders.append(
                    f"{md_path.relative_to(REPO_ROOT).as_posix()}: "
                    f"python -m {package}.{module} → use `uv run {script}`"
                )

    assert not offenders, (
        "Docs use the module invocation form where a console script exists:\n  "
        + "\n  ".join(sorted(offenders))
    )
