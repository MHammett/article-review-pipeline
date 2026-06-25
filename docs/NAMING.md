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
| `ci-web-intel` | `ci_web_intel` | **Voice-profile bootstrapping** (analyze a writing corpus → voice/style profile) | ⚠️ **OPEN** — see below |

### Open decision: `ci-web-intel`

The dist description says "Web intelligence gathering tools," but the package's
README is titled "Voice Profile Bootstrap" and it only does voice-profile work.
That name (a) doesn't match its purpose and (b) overlaps with `ci-article-review`'s
own web/URL features. **Resolve before the package grows.** Two options:

- **Rename → `ci-voice-profile`** (import `ci_voice_profile`) if this package *is* the
  voice-profile tool. Matches the README; removes the overlap. (Likely choice.)
- **Keep `ci-web-intel`** only if the roadmap genuinely has more web-intelligence
  tools coming that aren't article-review — and then reframe its README so "Web Intel"
  is the package and Voice Profile Bootstrap is its first tool.

## Consistency checklist (current audit)

Already correct — no change:
- Folder, repo, GitHub, `ci-` package dirs, `ci_*` import names
- `ci-core`, `ci-article-review` names
- Component-level `--help` descriptions and module docstrings in `ci-article-review`
  (`pipeline.py`, `check.py`, `discover.py`) — they correctly say "Article Review"

Fixed in this PR (`naming/platform-alignment`):
1. **Fix 1 — Root `README.md` framing** ✅ — retitled from `# Article Review Pipeline`
   to **Content Intelligence** and reframed the intro as the platform overview (three
   packages: `ci-core`, `ci-article-review`, `ci-web-intel`), with article-review
   presented as one component under its own heading rather than as the whole project.
2. **Fix 2 — User-Agent string** ✅ — `ArticleReviewPipeline/1.0` was hardcoded in 4
   places (`analysis/links.py` ×2, `adapters/citation/wayback.py`, `analysis/webpage.py`)
   plus a test. Replaced with a single shared constant `content-intelligence/<version>`
   in `ci-core` (`ci_core.http.USER_AGENT`), imported by all callers and asserted by
   the test — fixing both the off-scheme name and the duplication.

Future coordinated pass (deferred — NOT in this PR):
3. **`ci-web-intel` rename** ⚠️ — resolve the open decision above, then align dist name,
   import package, README, and pyproject description in one coordinated pass. This
   renames a package and touches import paths and the CI matrix, so it is intentionally
   held back from this low-risk naming PR.

## Applying changes

The `ci-web-intel` rename touches files the in-flight branches also edit (streaming
work in `adapters/`/`analysis/`; package layout). Apply that rename as **one
coordinated pass in the primary dev environment** — ideally rebased *after* the
in-flight branches land — not as ad-hoc commits, to avoid merge conflicts. The Fix 1
and Fix 2 changes above are low-overlap and ship in this PR.
