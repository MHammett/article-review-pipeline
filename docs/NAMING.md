# Naming Convention

Authoritative naming scheme for the **Content Intelligence** project. The goal is
that every name (a) sits at the correct level — platform vs. package — and (b) is
distinct: no two packages have names that could be confused for each other.

## The two levels

- **Platform (umbrella):** the whole project. Displayed as **"Content Intelligence"**;
  kebab form `content-intelligence` (repo, folder, GitHub).
- **Packages (components):** the individual tools in the uv workspace under `packages/`.
  Each carries the `ci-` prefix (= content-intelligence) and names *one distinct capability*.

**Core rule:** never label a component as the platform, or the platform as a component.
Repo-level artifacts say "Content Intelligence"; package-level artifacts say the
component name only.

## Naming by artifact type

| Artifact | Convention | Example |
|---|---|---|
| Platform display name | Title Case | Content Intelligence |
| Repo / folder / GitHub | kebab-case | `content-intelligence` |
| Package dist name (`pyproject.toml name`, dir under `packages/`) | kebab, `ci-` prefix | `ci-article-review` |
| Python import package (module dir, `import` statements) | snake_case mirror | `ci_article_review` |
| Component display name (that package's README title, `--help`, docstrings) | the component only — NOT "Content Intelligence" | "Article Review" |
| Runtime / outbound HTTP identifier (User-Agent, external log tags) | platform brand, single shared constant | `content-intelligence/<version>` |

## Package registry

Each package must name a **distinct capability**. If two package names could be
confused (e.g. one doing "web intelligence" while another already analyzes web URLs),
rename so the boundary is unambiguous.

| Dist name | Import package | Capability (must be distinct) | Status |
|---|---|---|---|
| `ci-core` | `ci_core` | Shared library used by the other packages | ✅ distinct & accurate |
| `ci-article-review` | `ci_article_review` | Review drafted/published articles (incl. `--url` web fetch) | ✅ distinct & accurate |
| `ci-style-profile` | `ci_style_profile` | Bootstrap a writing-style profile from a multi-source corpus (→ `publication.yaml`) | ✅ distinct & accurate |

### Resolved: `ci-web-intel` → `ci-style-profile`

The package was formerly named `ci-web-intel` (dist description "Web intelligence
gathering tools"; README titled "Voice Profile Bootstrap"). The name neither matched
its purpose — it only synthesizes a writing-style profile — nor stayed clear of
`ci-article-review`'s own web/URL features. **Resolved** by renaming the dist to
**`ci-style-profile`** (import `ci_style_profile`, console `style-profile-bootstrap`)
and aligning the implementation's terminology from "voice" to "style" so the name
matches the code. The publication.yaml output key is `style_profile` (the legacy
`voice_profile` key is still read for backward compatibility). "Voice" is retained
only where it denotes the distinct-authorial-persona detection task (see the package's
prompt files) and the `voice_style` review domain in `ci-article-review`.

## Consistency checklist (current audit)

Already correct — no change:
- Folder, repo, GitHub, `ci-` package dirs, `ci_*` import names
- `ci-core`, `ci-article-review` names
- Component-level `--help` descriptions and module docstrings in `ci-article-review`
  (`pipeline.py`, `check.py`, `discover.py`) — they correctly say "Article Review"

Fixed in this PR (`naming/platform-alignment`):
1. **Fix 1 — Root `README.md` framing** ✅ — retitled from `# Article Review Pipeline`
   to **Content Intelligence** and reframed the intro as the platform overview (three
   packages: `ci-core`, `ci-article-review`, `ci-style-profile`), with article-review
   presented as one component under its own heading rather than as the whole project.
2. **Fix 2 — User-Agent string** ✅ — `ArticleReviewPipeline/1.0` was hardcoded in 4
   places (`analysis/links.py` ×2, `adapters/citation/wayback.py`, `analysis/webpage.py`)
   plus a test. Replaced with a single shared constant `content-intelligence/<version>`
   in `ci-core` (`ci_core.http.USER_AGENT`), imported by all callers and asserted by
   the test — fixing both the off-scheme name and the duplication.

Done in a follow-up PR (`rename/ci-style-profile`):
3. **`ci-web-intel` → `ci-style-profile` rename** ✅ — resolved the open decision above and
   aligned the dist name, import package (`ci_web_intel` → `ci_style_profile`), README,
   pyproject description, console script (`style-profile-bootstrap`), Makefile, and the CI
   matrix in one coordinated pass. The accompanying "voice" → "style" terminology sweep
   renamed identifiers, the `publication.yaml` output key (`voice_profile` → `style_profile`,
   with backward-compatible fallback in `ci-article-review`), and synthesis prompts —
   preserving "voice" only for the persona-detection task and the `voice_style` review domain.
