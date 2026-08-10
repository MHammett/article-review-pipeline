# Content Intelligence

Content Intelligence is a [uv](https://docs.astral.sh/uv/) workspace of tools for
producing and reviewing written content. Code lives under `packages/`, one package
per capability (see [docs/NAMING.md](docs/NAMING.md) for the naming convention):

- **ci-core** (`ci_core`) — the shared foundation both application packages build on.
  It owns the LLM layer (`ci_core.llm`: the six streaming provider adapters, SSE
  handling, robust JSON extraction, cost estimation, the sliding-scale timeout model,
  and the model registry), outbound HTTP identity (`ci_core.http`), secret redaction
  (`ci_core.redact`), and the shared config helpers (`ci_core.config_helpers`). The
  provider/model reference data those read — `pricing.yaml`, `timeouts.yaml`,
  `model_registry.yaml` — lives in `ci_core/configs/` alongside its loaders.
  Dependencies flow one way, and the applications do not depend on each other; see
  [docs/NAMING.md](docs/NAMING.md#dependency-direction).

  It also carries settings (`ci_core.config`), SQLAlchemy persistence (`ci_core.db`,
  `ci_core.models`) and structured logging (`ci_core.logging`), driven by `alembic/`.
  Those are implemented and tested but have no production consumers yet — nothing
  outside ci-core's own tests and `alembic/env.py` imports them.
- **ci-article-review** (`ci_article_review`) — the article-review pipeline: runs a
  drafted or already-published article through grammar correction and ensemble
  multi-model AI review, then publishes to WordPress on approval. The mature package.
- **ci-style-profile** (`ci_style_profile`) — style-profile bootstrapping: analyzes a writing
  corpus across multiple sources and synthesizes a structured style profile for `publication.yaml`.
  Early-stage.

The rest of this README covers **ci-article-review**, the primary tool today.

---

## Article Review

A multi-pass automated review pipeline for published web content. Takes a drafted article through deterministic grammar correction and ensemble multi-model AI review, consolidates weighted feedback into a single prioritized report, and publishes to WordPress when you approve.

At `standard` thoroughness (one model per domain), typical cost is under $1.00 per article. At `thorough` or `maximum` thoroughness with all optional providers configured, expect $2–5 per run. The only fixed cost is LanguageTool Premium at $4.99/month (optional — skip if you already do a manual Grammarly pass).

---

## Requirements

- Python 3.10 or higher — https://www.python.org/downloads/
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- Git — https://git-scm.com/downloads

---

## Installation

All commands run in **PowerShell** on Windows. macOS/Linux: use forward slashes.

```powershell
git clone https://github.com/MHammett/content-intelligence.git
cd content-intelligence
uv sync
```

`uv sync` installs all dependencies (including dev tools) into a local `.venv/` — no `pip install` or manual venv activation needed.

---

## Quick start

**1. Run setup:**

```powershell
uv run python -m ci_article_review.setup
```

This creates `configs/`, copies the example templates, prompts for your publication name, and prints exactly what to fill in next. Run it with `--publication NAME` to skip the interactive prompt:

```powershell
uv run python -m ci_article_review.setup --publication dnacom
```

**2. Fill in `configs/user.yaml`** with your API keys and model selection. See [docs/PROVIDERS.md](docs/PROVIDERS.md) for account setup instructions per provider. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full config reference.

**3. Fill in `configs/your_publication_name.yaml`** with your style profile, audience description, and WordPress credentials.

**4. Verify your setup:**

```powershell
uv run python -m ci_article_review.check --publication your_publication_name
```

Makes one minimal call to each configured service and reports pass/fail with specific error messages before you run a real article through.

**Check for newer models** (optional, run any time):

```powershell
uv run python -m ci_article_review.discover
```

Queries each provider's live models API and reports what's available — so you always know when a newer model exists without reading every provider's changelog. Compares against your configured models and flags anything newer.

**5. Run the pipeline:**

```powershell
uv run python -m ci_article_review.pipeline --draft path/to/handoff.md --publication your_publication_name
```

Fill out `handoff_templates/draft_submission.md` and pass it as the `--draft` argument.

**6. Publish an approved draft:**

```powershell
uv run python -m ci_article_review.pipeline --publish path/to/publication_handoff.md --publication your_publication_name
```

Always saves as a WordPress draft unless you add `--publish-live`.

**Analyze an existing web page:**

```
python pipeline.py --url https://example.com/some-published-post --publication your_publication_name
```

Instead of a local handoff file, this fetches the page, extracts the main
article text (stripping nav/header/footer boilerplate), and runs the same
multi-model review on it — useful for an in-depth analysis of your own published
articles or a third-party piece.

For best extraction quality, install the optional `trafilatura` extra:

```
pip install trafilatura
```

Without it, a built-in heuristic extractor is used. URL mode only fetches
public hosts (an SSRF guard blocks private/loopback/cloud-metadata addresses)
and infers only the title and body — it cannot supply an author's
`primary_claim`, target audience, or pre-draft analysis (see
[docs/CONFIGURATION.md](docs/CONFIGURATION.md#url-input-mode)).

---

## The revision loop

The pipeline is built for a round trip between this tool and whatever chat model you drafted with. Each run writes two files side by side in `pipeline_history/<article-slug>/`:

- `run_N_<timestamp>_report.json` — the full machine-readable report
- `run_N_<timestamp>_review.md` — the same findings rendered as readable prose, SECTION 1 through SECTION 9

The markdown one is the artifact you actually work from. The end of every run prints its path:

```
Readable review (paste into chat): pipeline_history/my-article/run_1_20260809_143022_review.md
```

**The loop:**

1. **Run the pipeline** on your draft handoff (`--draft`) or a plain draft file (`--raw-draft`).
2. **Open the `review.md`.** Read it directly, or hand it to a model.
3. **Paste `handoff_templates/revise_after_review_prompt.md` into the same chat thread** that has the article's context, followed by the review's SECTION 1 through SECTION 8 content. That prompt has the model revise the draft *and* regenerate the metadata in one pass — so PRIMARY CLAIM, UNCERTAIN SECTIONS, KNOWN GAPS and the rest don't silently go stale against the revised text. Those sections are fed straight to the review models, so a stale KNOWN GAPS entry gets re-flagged on every subsequent run.
4. **Save what comes back** as two files: the revised draft, and the metadata block (the format matches `handoff_templates/metadata_only.md`).
5. **Re-run:**

   ```powershell
   uv run python -m ci_article_review.pipeline --raw-draft revised_draft.md --metadata metadata.md --publication your_publication_name
   ```

   The run number increments and the report gains a **Delta From Prior Run** block — word change, how many prior consensus flags you resolved, how many new ones appeared, and whether the primary claim or heading structure moved.

6. Repeat until the findings are ones you're content to ship, then publish with `--publish`.

`--raw-draft` on its own works too — it just uses the whole file as the article body and skips the metadata context, which means the review models lose the author-supplied framing. Pair it with `--metadata` whenever you have that context to give.

---

## Cross-run analytics

A single run tells you about one article. `ci-history-report` reads every `run_*_report.json` under `pipeline_history/` and reports what only shows up across many runs:

```powershell
uv run ci-history-report
```

| Flag | Purpose |
|---|---|
| `--history-root DIR` | Directory containing per-article run history (default `pipeline_history`) |
| `--article SLUG` | Scope to one article — the `pipeline_history/` subdirectory name, not the article title |
| `--recent-window N` | How many of the most recent calls/runs count as "recent" vs. baseline (default 5) |
| `--json` | Print the raw result as JSON instead of the console summary |
| `--verbose`, `-v` | DEBUG logging |

What it reports:

- **Provider reliability** — per-provider success rate over the most recent calls versus the historical baseline, with a `DEGRADED` flag when the recent failure rate is at least 40 points worse. This is the check that catches an expired API key failing across every domain, instead of noticing after several bad runs. Providers need at least 3 baseline calls before a comparison is made.
- **Cost** — total spend, average per run, and recent-versus-baseline direction (`increasing` / `decreasing` / `flat`, at a 15% relative threshold). Direction only — cost has no "better" way to editorialize.
- **Quality trends** — Flesch-Kincaid grade, SEO issue count, and broken link count, both globally (recent vs. baseline average) and per article across its own revision history (first run vs. latest, as `improved` / `worsened` / `unchanged`).

It reads `pipeline_history/` fresh every time — no database, no index. Reports missing a field are treated as "not enough history" rather than an error, since the report schema has grown over time.

---

## Command-line options

```powershell
uv run python -m ci_article_review.pipeline --draft HANDOFF --publication NAME [options]
```

Exactly one of `--draft`, `--raw-draft`, `--url`, or `--publish` is required — they are mutually exclusive.

| Flag | Purpose |
|---|---|
| `--draft PATH` | Run the review pipeline on a draft handoff document |
| `--raw-draft PATH` | Run on a plain article file with no handoff headers (e.g. pasted straight out of a chat session) — the whole file becomes the article body |
| `--url URL` | Fetch a published web page and run the review on its extracted content |
| `--publish PATH` | Publish an approved publication handoff (saves as draft unless `--publish-live`) |
| `--metadata PATH` | Metadata-only file (PRIMARY CLAIM, TARGET AUDIENCE, etc., no DRAFT section) to pair with `--raw-draft`. Requires `--raw-draft`. |
| `--publication NAME` | Publication config to use (`configs/NAME.yaml`) — **required** |
| `--publish-live` | Publish live instead of as a WordPress draft |
| `--config-dir DIR` | Config directory (default `configs`) |
| `--cost-preset PRESET` | Override `cost_preset` for this run: `economy` / `standard` / `balanced` / `thorough` / `maximum`. Doesn't modify `user.yaml`. |
| `--verbose`, `-v` | DEBUG logging |

**Calibration flags** (for measuring/tuning timeouts — see [docs/CONFIGURATION.md](docs/CONFIGURATION.md#timeouts-are-automatic-sliding-scale)):

| Flag | Purpose |
|---|---|
| `--no-timeout` | Disable timeout truncation so true completion times are measured, never cut off |
| `--only-model PROVIDER` | Run only one provider (e.g. `openai`) instead of the full ensemble |
| `--only-domain DOMAIN` | Run only one domain (`fact_check`, `voice_style`, `completeness`, `argument_integrity`, `red_team`) |

Example — measure one model/domain's true latency cheaply:

```powershell
uv run python -m ci_article_review.pipeline --draft handoff.md --publication mypub --cost-preset maximum --only-model openai --only-domain fact_check --no-timeout
```

> On Windows `cmd.exe`, keep the whole command on one line — backslash line-continuation is a bash feature and will split the command.

---

## Running the tests

```powershell
uv run pytest packages/
```

Around 600 tests, all external API calls mocked — no keys required.

---

## Project structure

The repository is a [uv](https://docs.astral.sh/uv/) workspace. Code lives under
`packages/`, one package per tool, each with its own `pyproject.toml`, `src/`
layout, and `tests/`. The article-review pipeline is the mature package; the
others are early-stage.

```
content-intelligence/
├── pyproject.toml                workspace root — [tool.uv.workspace] members, dev deps, mypy config
├── uv.lock                       resolved lockfile for the whole workspace
├── Makefile                      common dev tasks
├── requirements.txt              runtime dependencies
├── requirements-dev.txt          adds pytest and other dev tooling
├── .env.example                  documents all supported environment variables
├── .pre-commit-config.yaml       ruff + mypy pre-commit hooks
├── alembic.ini                   Alembic config for the ci-core database schema
├── alembic/                      migrations for ci_core.models (not used at runtime yet)
├── .github/                      CI workflows
│
├── packages/
│   ├── ci-article-review/        the article-review pipeline (the bulk of the code)
│   │   ├── pyproject.toml
│   │   ├── src/ci_article_review/
│   │   │   ├── pipeline.py            orchestration engine — start here
│   │   │   ├── setup.py               first-run scaffolding for configs/
│   │   │   ├── config_loader.py       config parsing and validation
│   │   │   ├── consolidation.py       weighted ensemble consolidation → one report
│   │   │   ├── handoff_parser.py      parses Template A and Template C documents
│   │   │   ├── history.py             saves run artifacts to pipeline_history/
│   │   │   ├── history_analytics.py   cross-run analytics over pipeline_history/ (ci-history-report)
│   │   │   ├── report_markdown.py     renders the readable run_N_*_review.md from the report
│   │   │   ├── check.py               connectivity/credential check for all services
│   │   │   ├── discover.py            live model discovery — queries provider APIs
│   │   │   ├── probe.py               lightweight provider reachability probe
│   │   │   │                          (the provider adapters, timeout model, model
│   │   │   │                          registry, cost, and redaction live in ci-core)
│   │   │   │
│   │   │   ├── adapters/
│   │   │   │   ├── grammar/languagetool.py   grammar correction (Pass 1)
│   │   │   │   ├── cms/wordpress.py          WordPress REST API publisher
│   │   │   │   └── citation/
│   │   │   │       ├── resolver.py           primary source resolution, checksums, confidence tiers
│   │   │   │       ├── wayback.py            Wayback archive check + Save Page Now submission
│   │   │   │       ├── topic_match.py        keyword gating for pointer-only adapters
│   │   │   │       └── sources/              10 adapters: census, crossref, eia, epa, ferc,
│   │   │   │                                 fhwa, fred, icc, ilga, pjm
│   │   │   │
│   │   │   ├── analysis/
│   │   │   │   ├── readability.py     Flesch-Kincaid grade, word count, sentence stats
│   │   │   │   ├── links.py           URL extraction, HTTP status check, Wayback archive check
│   │   │   │   ├── seo.py             title length, heading structure, meta description
│   │   │   │   └── webpage.py         webpage fetch/extraction helpers
│   │   │   │
│   │   │   ├── prompts/               system prompts for each review domain
│   │   │   ├── configs/               committed defaults: presets.yaml + *.example.yaml
│   │   │   │                          (real user.yaml + publication.yaml are gitignored;
│   │   │   │                          pricing/timeouts/model_registry live in ci-core)
│   │   │   └── handoff_templates/     fill these out to submit drafts and publish
│   │   └── tests/                     pipeline test suite, all external calls mocked
│   │
│   ├── ci-core/                  the shared foundation both applications import
│   │   ├── pyproject.toml
│   │   ├── src/ci_core/
│   │   │   ├── llm/
│   │   │   │   ├── adapters/     the six streaming provider adapters + call_provider/
│   │   │   │   │                 call_text dispatch (claude, gemini, grok, mistral,
│   │   │   │   │                 openai, perplexity)
│   │   │   │   ├── streaming.py       SSE accumulation and read-gap timeouts
│   │   │   │   ├── json_utils.py      robust JSON extraction (fences, think-preambles,
│   │   │   │   │                      truncation salvage)
│   │   │   │   ├── cost.py            token-based cost estimation
│   │   │   │   ├── timeout_model.py   sliding-scale timeout from size × model × effort
│   │   │   │   └── model_registry.py  current/superseded model detection
│   │   │   ├── http.py           USER_AGENT + DEFAULT_HEADERS for all outbound calls
│   │   │   ├── redact.py         scrubs API keys from error output before logging
│   │   │   ├── config_helpers.py load_yaml, resolve_env_recursive, normalize_model_configs
│   │   │   ├── configs/          pricing.yaml, timeouts.yaml, model_registry.yaml
│   │   │   ├── config.py         pydantic settings (no production consumer yet)
│   │   │   ├── db.py             async SQLAlchemy engine/session (no production consumer yet)
│   │   │   ├── models.py         ORM models (no production consumer yet)
│   │   │   └── logging.py        structured logging (no production consumer yet)
│   │   └── tests/
│   │
│   └── ci-style-profile/         style-profile bootstrapping (see PLAN.md)
│       ├── pyproject.toml
│       ├── sources.example.yaml  copy to src/ci_style_profile/sources.yaml
│       ├── configs/presets.yaml
│       ├── src/ci_style_profile/
│       │   ├── bootstrap.py      entry point (style-profile-bootstrap)
│       │   ├── collectors/       wordpress, gmail, outlook365, twitter, textfiles, custom/
│       │   ├── detect.py         style detection pass
│       │   ├── synthesize.py     profile synthesis
│       │   ├── style_consolidation.py
│       │   ├── normalize.py, callers.py, output.py, logging_config.py
│       │   ├── staging/          cached source text (gitignored)
│       │   └── profiles/         timestamped profile snapshots (gitignored)
│       └── tests/
│
├── pipeline_history/             run reports, readable reviews, and daily pipeline logs
│                                 (gitignored, local only)
└── docs/                         extended documentation
    ├── PROVIDERS.md              account setup for every service
    ├── CONFIGURATION.md          full config reference, thoroughness, ensemble weights
    ├── CITATIONS.md              Section 9 confidence tiers and archiving behavior
    ├── NAMING.md                 package/module naming convention
    ├── TERMINOLOGY.md            "voice" vs. "style" and other term definitions
    └── TROUBLESHOOTING.md        error messages and fixes
```

---

## Documentation

- **[docs/PROVIDERS.md](docs/PROVIDERS.md)** — Account setup and API keys for every service: OpenAI, Gemini (AI Studio + Vertex AI), Mistral, Perplexity, Grok, Claude, LanguageTool, WordPress.

- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** — Full `user.yaml` reference: model config (simple and extended forms), web search, Vertex AI, Azure, thoroughness levels, ensemble weighting.

- **[docs/CITATIONS.md](docs/CITATIONS.md)** — How Section 9 resolves claims to primary sources: the three confidence tiers (verified / pointer-only / unresolved), what each one does and doesn't prove, and the Wayback Machine archiving behavior.

- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — Error messages and fixes for every service, plus pipeline behavior edge cases.

- **[docs/NAMING.md](docs/NAMING.md)** / **[docs/TERMINOLOGY.md](docs/TERMINOLOGY.md)** — Package naming convention, and the deliberate "voice" vs. "style" distinction.
