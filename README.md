# Article Review Pipeline

A multi-pass automated review pipeline for published web content. Takes a drafted article through deterministic grammar correction and ensemble multi-model AI review, consolidates weighted feedback into a single prioritized report, and publishes to WordPress when you approve.

At `standard` thoroughness (one model per domain), typical cost is under $1.00 per article. At `thorough` or `maximum` thoroughness with all optional providers configured, expect $2–5 per run. The only fixed cost is LanguageTool Premium at $4.99/month (optional — skip if you already do a manual Grammarly pass).

---

## Requirements

- Python 3.10 or higher — https://www.python.org/downloads/
- Git — https://git-scm.com/downloads

---

## Installation

All commands run in **Command Prompt** or **PowerShell** on Windows. macOS/Linux: substitute `cp` for `copy` and use forward slashes.

```
git clone https://github.com/MHammett/content-intelligence.git
cd content-intelligence
pip install -r requirements.txt
```

Test dependencies (optional):

```
pip install -r requirements-dev.txt
```

---

## Quick start

**1. Copy the config templates:**

```
copy configs\user.example.yaml configs\user.yaml
copy configs\publication.example.yaml configs\your_publication_name.yaml
```

**2. Fill in `configs/user.yaml`** with your API keys and model selection. See [docs/PROVIDERS.md](docs/PROVIDERS.md) for account setup instructions. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full config reference.

**3. Fill in `configs/your_publication_name.yaml`** with your voice profile, audience description, and WordPress credentials.

**4. Verify your setup:**

```
python check.py --publication your_publication_name
```

Makes one minimal call to each configured service and reports pass/fail with specific error messages before you run a real article through.

**Check for newer models** (optional, run any time):

```
python discover.py
```

Queries each provider's live models API and reports what's available — so you always know when a newer model exists without reading every provider's changelog. Compares against your configured models and flags anything newer.

**5. Run the pipeline:**

```
python pipeline.py --draft path/to/handoff.md --publication your_publication_name
```

Fill out `handoff_templates/draft_submission.md` and pass it as the `--draft` argument.

**6. Publish an approved draft:**

```
python pipeline.py --publish path/to/publication_handoff.md --publication your_publication_name
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

## Command-line options

```
python pipeline.py --draft HANDOFF --publication NAME [options]
```

| Flag | Purpose |
|---|---|
| `--draft PATH` | Run the review pipeline on a draft handoff document |
| `--url URL` | Fetch a published web page and run the review on its extracted content (mutually exclusive with `--draft`/`--publish`) |
| `--publish PATH` | Publish an approved publication handoff (saves as draft unless `--publish-live`) |
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

```
python pipeline.py --draft handoff.md --publication mypub --cost-preset maximum --only-model openai --only-domain fact_check --no-timeout
```

> On Windows `cmd.exe`, keep the whole command on one line — backslash line-continuation is a bash feature and will split the command.

---

## Running the tests

```
pytest tests/
```

279 tests, all external API calls mocked — no keys required.

---

## Project structure

```
content-intelligence/
├── pipeline.py                   orchestration engine — start here
├── config_loader.py              config parsing and validation
├── consolidation.py              weighted ensemble consolidation → one report
├── handoff_parser.py             parses Template A and Template C documents
├── history.py                    saves run artifacts to pipeline_history/
├── check.py                      connectivity/credential check for all services
├── discover.py                   live model discovery — queries provider APIs for available models
├── model_registry.py             loads configs/model_registry.yaml — current/superseded model detection
├── timeout_model.py              sliding-scale per-call timeout from size × model × effort
├── redact.py                     scrubs API keys from error output before logging/printing
├── requirements.txt              runtime dependencies
├── requirements-dev.txt          adds pytest for development
├── .env.example                  documents all supported environment variables
│
├── adapters/
│   ├── grammar/languagetool.py   grammar correction (Pass 1)
│   ├── review/gemini.py          fact verification with live search
│   ├── review/openai.py          voice/style, completeness, optional web search
│   ├── review/mistral.py         argument integrity and red team
│   ├── review/perplexity.py      search-grounded fact-check (optional)
│   ├── review/grok.py            red team — contrarian corpus (optional)
│   ├── review/claude.py          argument integrity — independent lineage (optional)
│   ├── review/json_utils.py      shared robust JSON extraction (fences, think-preambles)
│   ├── cms/wordpress.py          WordPress REST API publisher
│   └── citation/
│       ├── resolver.py           primary source resolution and checksums
│       ├── wayback.py            Wayback Machine archive availability check
│       └── sources/              FRED, EIA, Census, FHWA data adapters
│
├── analysis/
│   ├── readability.py            Flesch-Kincaid grade, word count, sentence stats
│   ├── links.py                  URL extraction, HTTP status check, Wayback archive check
│   ├── seo.py                    title length, heading structure, meta description
│   └── cost.py                   token-based cost estimation with per-model pricing table
│
├── prompts/                      system prompts for each review domain
├── configs/                      API keys + publication settings (gitignored), plus
│                                 committed defaults: presets.yaml, pricing.yaml,
│                                 timeouts.yaml, model_registry.yaml
├── handoff_templates/            fill these out to submit drafts and publish
├── pipeline_history/             run reports saved here (gitignored, local only)
├── docs/                         extended documentation
│   ├── PROVIDERS.md              account setup for every service
│   ├── CONFIGURATION.md          full config reference, thoroughness, ensemble weights
│   └── TROUBLESHOOTING.md        error messages and fixes
└── tests/                        279 tests, all external calls mocked
```

---

## Documentation

- **[docs/PROVIDERS.md](docs/PROVIDERS.md)** — Account setup and API keys for every service: OpenAI, Gemini (AI Studio + Vertex AI), Mistral, Perplexity, Grok, Claude, LanguageTool, WordPress.

- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** — Full `user.yaml` reference: model config (simple and extended forms), web search, Vertex AI, Azure, thoroughness levels, ensemble weighting.

- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — Error messages and fixes for every service, plus pipeline behavior edge cases.
