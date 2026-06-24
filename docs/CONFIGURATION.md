# Configuration Reference

Configuration lives in several files:

- `configs/user.yaml` — your API keys, model selection, and pipeline behavior (gitignored, never committed)
- `configs/your_publication_name.yaml` — per-publication settings: voice profile, audience, WordPress credentials
- `configs/presets.yaml` — cost preset model assignments; edit to update model names without touching code
- `configs/pricing.yaml` — per-million-token pricing for cost estimation; update when providers change prices
- `configs/model_registry.yaml` — model deprecation tracking; edit to add superseded entries and bump the date
- `configs/timeouts.yaml` — sliding-scale timeout model (size × model × effort multipliers)

The first two are gitignored and have example templates you copy (`user.example.yaml`, `publication.example.yaml`, plus worked examples in `configs/examples/`). The other four are committed defaults — they ship with the repo and *are* their own reference; edit them in place, no copy step.

---

## Step 1: Run setup

The fastest way to scaffold configs is the built-in setup command, which also verifies dependencies:

```powershell
uv run python -m ci_article_review.setup --publication your_publication_name
```

This creates `configs/`, copies both example templates, and prints exactly what to fill in.

**Manual copy** (if you prefer to do it yourself):

```powershell
# PowerShell
New-Item -ItemType Directory -Force configs
copy packages\ci-article-review\src\ci_article_review\configs\user.example.yaml configs\user.yaml
copy packages\ci-article-review\src\ci_article_review\configs\publication.example.yaml configs\your_publication_name.yaml
```

```bash
# macOS / Linux / Git Bash
mkdir -p configs
cp packages/ci-article-review/src/ci_article_review/configs/user.example.yaml configs/user.yaml
cp packages/ci-article-review/src/ci_article_review/configs/publication.example.yaml configs/your_publication_name.yaml
```

The `.gitignore` excludes `configs/user.yaml` and all `configs/*.yaml` files that aren't examples or committed defaults. Your keys will not be committed.

---

## user.yaml

### API keys

```yaml
api_keys:
  openai:
    api_key: sk-...

  gemini:
    api_key: AI...        # omit when using provider: vertex_ai

  mistral:
    api_key: your_mistral_key

  # Optional
  perplexity:
    api_key: your_perplexity_key

  grok:
    api_key: your_grok_key

  claude:
    api_key: your_claude_key

  languagetool:
    username: your_email@example.com
    api_key: your_languagetool_key
```

Instead of putting keys directly in the YAML, you can use environment variables:

```yaml
api_keys:
  openai:
    api_key: ${OPENAI_API_KEY}
```

Copy `.env.example` to `.env`, fill in the values. The pipeline loads `.env` automatically.

---

### Model configuration

Two forms are accepted. Mix and match — each model can use either form independently.

**Simple form:** model name as a string, uses the provider's public API.

```yaml
models:
  openai: gpt-5.4             # gpt-5.5 for max quality, gpt-5.4-mini for economy
  gemini: gemini-2.5-flash    # best price-performance; gemini-3.5-flash for upgrade
  mistral: mistral-large-latest
  perplexity: sonar-reasoning-pro
  grok: grok-4.3              # grok-4.20-0309-reasoning for CoT (same price)
  claude: claude-opus-4-8     # claude-sonnet-4-6 for lower cost with extended thinking
```

**Extended form:** dict with `model`, optional `provider`, and provider-specific fields.

```yaml
models:
  gemini:
    provider: vertex_ai
    model: gemini-2.5-flash
    project: your-gcp-project-id
    location: us-central1
    # credentials_file: C:\Users\you\keys\my-project.json   # omit = ADC
```

#### Disabling a model

Use `enabled: false` to skip a model without removing its key — useful when comparing runs:

```yaml
models:
  grok:
    enabled: false
    model: grok-4.3
```

#### Restricting which prompts a model runs

Use `prompts:` to override the thoroughness preset for a specific model. This model will only run the listed domains regardless of the thoroughness setting:

```yaml
models:
  perplexity:
    model: sonar-pro
    prompts: [fact_check]   # only run fact-check; skip voice, argument, etc.
```

Valid domain names: `fact_check`, `voice_style`, `completeness`, `argument_integrity`, `red_team`

The `prompts:` list is an **infrastructure key** — it survives `cost_preset` overrides. If you set `prompts:` in `models:` for a provider, the preset will not clear it even when it overrides the model ID and reasoning flags.

Common use: excluding `fact_check` from Claude (which has no live web search and will always fail it):

```yaml
models:
  claude:
    model: claude-opus-4-8
    prompts: [voice_style, completeness, argument_integrity, red_team]
    # fact_check excluded: Claude has no live web search capability
```

#### The three timeout layers (streaming)

The review adapters stream responses (Server-Sent Events). That splits "the timeout" into three layers with distinct jobs — knowing which is which is the key to tuning:

| Layer | What it bounds | Typical size | Set by |
|---|---|---|---|
| **Inter-token read gap** | Max stall *between* streamed tokens (the socket read timeout) | small constant (~120s; 160s for grounded Gemini/Perplexity) | `stream_read_timeout` per model, else adapter default |
| **Per-task wall-clock backstop** | Total time one model+domain call may run before the pipeline thread kills it | the sliding-scale computed value (below) | `timeout_seconds` per model, else computed |
| **Global batch ceiling** | Outer bound on the whole parallel batch | slowest backstop + retry + slack | derived (`_global_ceiling()`) |

**Why streaming matters:** before streaming, a model that buffered its entire 16–30k-token reasoning+output and sent nothing until done forced the *socket* read timeout to cover the full compute time (gpt-5.5 xhigh needed an ~819s per-call timeout). With streaming, tokens arrive incrementally, so the socket timeout becomes the **gap between tokens** — small and constant regardless of total length, and a hang/stall is caught in ~120s instead of after the whole giant budget elapses.

Streaming does **not** make a long generation finish faster — that gpt-5.5 xhigh call still emits tokens for ~800s. So the **wall-clock backstop still must cover the genuine total generation time**; it just no longer has to absorb "model sent nothing for 800s, is it hung?" — the read-gap layer answers that.

#### Wall-clock backstop is automatic (sliding scale)

You normally don't set timeouts at all. After pre-analysis, the pipeline sizes each model's **wall-clock backstop** from the draft's **character count**, the **model**, and the **reasoning effort**, using the multiplier tables in [`configs/timeouts.yaml`](../configs/timeouts.yaml):

```
effective = clamp( base × size_mult × model_mult × effort_mult × variance_margin,  floor,  task_timeout_seconds − 15 )
```

So a short economy run gets a small backstop and a 10k-word `maximum` run gives gpt-5.5 xhigh its full budget (~1020s) — no hand-tuning. `pipeline.task_timeout_seconds` is the absolute ceiling the formula clamps to. Edit `timeouts.yaml` to retune (the model/effort values are calibrated; the size buckets are anchored to one ~74k-char document and are reasoned estimates elsewhere).

**`variance_margin`** is a single global safety buffer (default `1.25`) multiplied onto every computed backstop. The multipliers target *typical* completion time; this margin covers run-to-run variance in **total** time (reasoning output volume swings widely). Under streaming it no longer has to cover *stall* uncertainty — the read-gap layer catches stalls directly — so it can be tuned lower than under buffered POSTs. Raise it for more headroom (longer worst-case waits), lower it to fail faster. The trade is purely worst-case wait, not typical run time — calls return the instant they finish.

To **override the read gap** for a specific model, set `stream_read_timeout` (seconds) on it — useful for a grounded model whose live search delays the first token. To **override the wall-clock backstop**, set `timeout_seconds` explicitly — that value wins and skips the formula:

#### Per-model timeout overrides

Set `timeout_seconds` per model to override the sliding-scale **wall-clock backstop**; set `stream_read_timeout` to override the **inter-token read gap**:

```yaml
models:
  gemini:
    provider: vertex_ai
    model: gemini-2.5-flash
    timeout_seconds: 540        # wall-clock backstop for live-search fact-check
    stream_read_timeout: 200    # allow a longer first-token gap while it searches

  mistral:
    model: mistral-medium-3-5
    reasoning_effort: high
    timeout_seconds: 240        # total-time headroom for reasoning on long articles

  claude:
    model: claude-opus-4-8
    timeout_seconds: 240
```

Under streaming the adapter passes `timeout=(connect, read_gap)` to the HTTP request, where `read_gap` is the **inter-token** allowance (constant, from `stream_read_timeout` or the adapter default) — **not** `timeout_seconds`. The big `timeout_seconds` value is enforced separately as the pipeline's per-task thread wall-clock backstop. Set `pipeline.task_timeout_seconds` high enough to accommodate your slowest model's genuine total generation time (streaming detects stalls quickly but does not shorten a legitimately long generation).

`timeout_seconds` is an **infrastructure key** — it survives `cost_preset` overrides. If you set it in `models:` for a provider, the preset will not clear it.

---

### Reasoning controls

Each provider exposes reasoning differently. All can be set in the extended model config.

#### OpenAI — `reasoning_effort`

Valid for `gpt-5.x` models. Controls the depth of the model's chain-of-thought before output.

```yaml
models:
  openai:
    model: gpt-5.4
    reasoning_effort: medium    # none | low | medium | high | xhigh
    timeout_seconds: 300
```

`gpt-5.5` defaults to `medium` reasoning if you omit the parameter. `gpt-5.4` defaults to lower. `xhigh` adds 30–60s per call but noticeably improves argument critique.

#### Claude — adaptive vs extended thinking

**Critical distinction:** the mode depends on the model. Using the wrong mode causes a 400 error.

| Model | Thinking mode | How to configure |
|---|---|---|
| claude-opus-4-8 | Adaptive (always on) | `effort: low/medium/high` — controls depth |
| claude-fable-5 | Adaptive (always on, not configurable) | No config needed |
| claude-sonnet-4-6 | Adaptive (always on) | `effort: low/medium/high` — controls depth |
| claude-haiku-4-5-20251001 | Extended (opt-in) | `thinking_budget: N` — token ceiling |

**Do not** set `thinking_budget` on Opus 4.8, Fable 5, or Sonnet 4.6 — they use adaptive thinking and the parameter is ignored (or causes an error). Use `effort:` instead. Only Haiku 4.5 uses extended thinking with `thinking_budget`.

```yaml
# Opus 4.8 or Sonnet 4.6 — adaptive thinking, control effort level
models:
  claude:
    model: claude-opus-4-8
    effort: high            # "low" | "medium" | "high"
    timeout_seconds: 240

# Haiku 4.5 — extended thinking with token budget
models:
  claude:
    model: claude-haiku-4-5-20251001
    thinking_budget: 5000   # allocates up to 5K reasoning tokens
    timeout_seconds: 240
```

#### Gemini — `thinking_budget`

```yaml
models:
  gemini:
    provider: vertex_ai
    model: gemini-2.5-flash
    thinking_budget: 8192   # 0 = disable; omit = dynamic default
    project: your-gcp-project-id
    location: us-central1
```

All Gemini 2.5 and 3.x models support this. Gemini 2.5 Flash already uses dynamic thinking by default; setting a budget caps the maximum token allocation.

#### Grok — model selection

Grok reasoning is model-based, not parameter-based:

```yaml
models:
  grok:
    model: grok-4.20-0309-reasoning   # reasoning variant (same $1.25/$2.50 price)
    timeout_seconds: 180
```

#### Mistral — `reasoning_effort`

Mistral reasoning runs on `mistral-medium-3-5` (not on standard models like `mistral-large-latest`). Two critical constraints:

1. **Model ID:** The reasoning model is `mistral-medium-3-5` — the `-latest` suffix variant does not exist and returns a 400 error.
2. **Accepted values:** Only `"high"` and `"none"` are accepted. `"low"` and `"medium"` return a 400 error.

Standard Mistral models (`mistral-large-latest`, `mistral-small-latest`) do not support `reasoning_effort` at all and will return a 400 error if you add it.

```yaml
# Reasoning model (replaces deprecated magistral-medium-latest)
models:
  mistral:
    model: mistral-medium-3-5
    reasoning_effort: high    # only "high" or "none" accepted on this model
    timeout_seconds: 240

# Standard model (no reasoning)
models:
  mistral:
    model: mistral-large-latest
    timeout_seconds: 240
```

The `balanced` cost preset uses `mistral-medium-3-5` without a `reasoning_effort` flag (no `"low"` tier available). `thorough` and `maximum` use `reasoning_effort: "high"`.

---

### OpenAI web search

GPT-5.x can run through the OpenAI Responses API with live web search enabled. Set `web_search: true` in the openai model config:

```yaml
models:
  openai:
    model: gpt-5.4
    web_search: true
```

When enabled, the pipeline uses the Responses API with `web_search_preview` for all OpenAI calls. If the Responses API is unavailable it falls back to standard chat completions silently. When web search is active, OpenAI's weight for `fact_check` effectively becomes similar to Gemini's — useful at `thorough` or `maximum` thoroughness.

---

### Vertex AI (Gemini)

Switch Gemini from AI Studio to a reserved capacity pool:

```yaml
models:
  gemini:
    provider: vertex_ai
    model: gemini-2.5-flash
    project: your-gcp-project-id       # GCP project ID (not display name)
    location: us-central1
    # credentials_file: path\to\key.json  # omit to use Application Default Credentials
```

See [PROVIDERS.md](PROVIDERS.md#option-b--vertex-ai-reserved-capacity-no-503s) for the full setup walkthrough.

---

### Azure OpenAI

```yaml
api_keys:
  openai:
    api_key: your-azure-resource-key   # NOT an OpenAI key

models:
  openai:
    provider: azure
    model: gpt-5.4                     # informational label only
    endpoint: https://my-resource.openai.azure.com
    deployment: my-gpt5-deployment
    api_version: "2024-02-01"          # optional; defaults to 2024-02-01
```

Azure provisioned throughput deployments do not 503; the fallback chain is bypassed.

---

### Azure AI (Mistral serverless)

```yaml
api_keys:
  mistral:
    api_key: your-azure-endpoint-key

models:
  mistral:
    provider: azure
    model: mistral-large-latest        # informational label only
    endpoint: https://Mistral-Large-abc.eastus2.inference.ai.azure.com
```

---

### Pipeline behavior

```yaml
pipeline:
  grammar_pass: true            # false = skip LanguageTool entirely
  parallel_review_calls: true
  retry_on_failure: true
  retry_delay_seconds: 10
  abort_if_all_provider_calls_fail: false
  task_timeout_seconds: 1100    # absolute ceiling for the sliding-scale timeout model; formula clamps to this − 15
  cost_preset: balanced         # economy | standard | balanced | thorough | maximum

  link_validation: true         # check HTTP status of every URL in the draft
  wayback_link_check: true      # also query the Wayback Machine for each URL
  wayback_snapshot_stale_days: 180  # snapshots older than this are flagged [STALE]
```

**`wayback_snapshot_stale_days`** controls when a Wayback Machine snapshot is considered stale. At 180 days (default), a snapshot from more than six months ago triggers a `[STALE]` flag and a manual re-archive recommendation. Lower this for publications with high source-freshness standards (e.g., 90 days for breaking-news adjacent pieces).

---

### Cost presets

The `cost_preset` setting is the easiest way to control quality vs cost. It sets model variants, reasoning flags, and thoroughness level as a bundle. You set one value instead of configuring six providers separately.

Preset model assignments live in [`configs/presets.yaml`](../configs/presets.yaml). When providers release new models, edit that file to update the model names — no code change needed.

**When `cost_preset` is set:**
- It overrides model names and reasoning flags for all configured providers
- Provider infrastructure settings (vertex_ai, azure, credentials_file) are preserved
- Providers you haven't configured (no API key) are skipped
- Setting `enabled: false` on a provider still takes precedence
- Set `thoroughness:` separately to override just that part of the preset

**Estimated per-article costs** (assumes a 1500-word article ≈ 4000 in / 2000 out tokens per call):

| Preset | Thoroughness | Models used | Reasoning | Est. cost |
|---|---|---|---|---|
| `economy` | standard | gpt-5.4-mini, gemini-2.5-flash, mistral-small | none | $0.04–$0.10 |
| `standard` | standard | gpt-5.4, gemini-2.5-flash, mistral-large, grok-4.3, claude-haiku | none | $0.10–$0.40 |
| `balanced` | thorough | o4-mini, gemini-2.5-flash, mistral-medium-3-5, grok-4.3+reasoning, claude-sonnet-4-6+adaptive | light | $0.50–$1.50 |
| `thorough` | thorough | o4-mini, gemini-2.5-flash, mistral-medium-3-5+reasoning, grok-4.3+reasoning, claude-opus-4-8 | deep | $1.00–$2.50 |
| `maximum` | maximum | o3, gemini-2.5-pro, mistral-medium-3-5+reasoning, grok-4.3+reasoning, claude-opus-4-8 | max | $2.50–$5.00 |

**Guidance:**
- Single-digit cents per article is genuinely cheap for a publishing workflow. `balanced` is the recommended default — it gives thorough search-grounded fact-check plus light reasoning on all argument passes for about $1/article.
- `economy` is for volume workflows where speed and cost matter more than depth.
- `maximum` is for high-stakes pieces where you want every model running every domain at max reasoning. The marginal improvement between `thorough` and `maximum` is real but not dramatic.

---

### Preset detail — exact model and reasoning flags per platform

The tables below show exactly what settings each preset applies to each provider. "User model" means the preset does not change that provider's model — it uses whatever you have configured in `models:`.

#### economy preset — standard thoroughness, ~$0.04–$0.10/article

| Provider | Model | Reasoning | Notes |
|---|---|---|---|
| openai | `gpt-5.4-mini` | none | Mini variant; lower cost, reduced depth |
| gemini | _(user model)_ | none | Budget applied dynamically (model default) |
| mistral | `mistral-small-latest` | none | Small variant; sufficient for basic review |
| perplexity | `sonar` | — | Lightweight search-grounded; no CoT |
| grok | **disabled** | — | Excluded at this cost tier |
| claude | **disabled** | — | Excluded at this cost tier |

#### standard preset — standard thoroughness, ~$0.10–$0.40/article

| Provider | Model | Reasoning | Notes |
|---|---|---|---|
| openai | `gpt-5.4` | none | Flagship value model |
| gemini | _(user model)_ | none | Default dynamic thinking |
| mistral | `mistral-large-latest` | none | Full reasoning model |
| perplexity | `sonar-pro` | — | Search-grounded, no CoT trace |
| grok | `grok-4.3` | — | Standard model |
| claude | `claude-haiku-4-5-20251001` | none | Cheapest Claude variant |

#### balanced preset — thorough thoroughness, ~$0.50–$1.50/article *(recommended default)*

| Provider | Model | Reasoning | Param | Notes |
|---|---|---|---|---|
| openai | `o4-mini` | `reasoning_effort` | `"low"` | Light CoT, modest latency increase |
| gemini | _(user model)_ | — | not set (dynamic) | Dynamic thinking (model default) |
| mistral | `mistral-medium-3-5` | — | — | Reasoning model; `low`/`medium` not accepted — preset omits effort flag |
| perplexity | `sonar-reasoning-pro` | — | — | CoT+search grounding |
| grok | `grok-4.3` | `reasoning_effort` | `"low"` | Light CoT |
| claude | `claude-sonnet-4-6` | `effort` | `"medium"` | Adaptive thinking (always on on Sonnet 4.6); effort controls depth |

#### thorough preset — thorough thoroughness, ~$1.00–$2.50/article

| Provider | Model | Reasoning | Param | Notes |
|---|---|---|---|---|
| openai | `o4-mini` | `reasoning_effort` | `"medium"` | Standard CoT depth; ~10–20s overhead per call |
| gemini | _(user model)_ | — | not set (dynamic) | Dynamic thinking (model default) |
| mistral | `mistral-medium-3-5` | `reasoning_effort` | `"high"` | Deep CoT; only `"high"` or `"none"` accepted on this model |
| perplexity | `sonar-reasoning-pro` | — | — | CoT+search grounding |
| grok | `grok-4.3` | `reasoning_effort` | `"medium"` | Standard CoT depth |
| claude | `claude-opus-4-8` | `effort` | `"high"` | Adaptive thinking (always on); effort=high pushes harder |

#### maximum preset — maximum thoroughness, ~$2.50–$5.00/article

| Provider | Model | Reasoning | Param | Notes |
|---|---|---|---|---|
| openai | `o3` | `reasoning_effort` | `"high"` | Highest-capability model at full reasoning depth |
| gemini | `gemini-2.5-pro` | `thinking_budget` | `16000` | Upgraded to pro; 16K thinking budget; flash doesn't support `thinking_budget` in Vertex AI |
| mistral | `mistral-medium-3-5` | `reasoning_effort` | `"high"` | Deep CoT; only `"high"` or `"none"` accepted |
| perplexity | `sonar-reasoning-pro` | — | — | CoT+search grounding |
| grok | `grok-4.3` | `reasoning_effort` | `"high"` | Full CoT depth |
| claude | `claude-opus-4-8` | `effort` | `"high"` | Adaptive thinking, max effort |

**Notes on the preset tables:**
- "_(user model)_" means the preset does not override that provider's model name — your `models:` entry is used as-is. The preset may still add reasoning flags.
- Gemini's `thinking_budget` is only set at `maximum` (requires `gemini-2.5-pro`) — for other presets the model uses its dynamic default. Set `thinking_budget: 0` in your Gemini model config to disable thinking for any preset.
- `mistral-medium-3-5` is the reasoning-capable Mistral model (replaces the deprecated `magistral-medium-latest`). It only accepts `reasoning_effort: "high"` or `"none"` — not `"low"` or `"medium"`. The `-latest` suffix variant (`mistral-medium-3-5-latest`) does not exist and returns a 400 error.
- Claude's adaptive thinking is always on for Sonnet 4.6, Opus 4.8, and Fable 5. The `effort:` parameter controls reasoning depth; `thinking_budget:` is not used on these models.
- Provider infrastructure settings (Vertex AI, Azure, credentials) are always preserved regardless of preset.
- If you haven't configured a provider (no API key), the preset silently skips it.

**Per-model cost reference** (per call at ~6000 tokens total):

| Provider / model | $/call (approx.) | With reasoning |
|---|---|---|
| gemini-2.5-flash | $0.006 | +$0.01–$0.03 (dynamic thinking) |
| gemini-3.5-flash | $0.024 | +$0.05+ |
| gpt-5.4-mini | $0.012 | — |
| gpt-5.4 | $0.040 | +$0.01–$0.06 (effort low→xhigh) |
| gpt-5.5 | $0.080 | +$0.03–$0.10 |
| grok-4.3 / grok-4.20-reasoning | $0.010 | same price |
| mistral-large | $0.030 | +$0.01–$0.04 |
| claude-haiku-4-5 | $0.014 | +$0.02+ (extended thinking) |
| claude-sonnet-4-6 | $0.042 | +$0.05+ (extended thinking) |
| claude-opus-4-8 | $0.070 | adaptive, always included |
| perplexity sonar-pro | $0.042 | — |
| perplexity sonar-reasoning-pro | $0.10–$0.20 | CoT trace, high variance |

---

### Thoroughness

Controls how many models run each review domain per pipeline run.

| Level | Fact-check | Voice/style | Completeness | Argument | Red team | Approx. calls |
|---|---|---|---|---|---|---|
| `standard` | Gemini | OpenAI | OpenAI | Mistral + Claude* | Mistral + Grok* | 5–7 |
| `thorough` | Gemini + Perplexity* | OpenAI + Claude* | OpenAI + Mistral | Mistral + Claude* + OpenAI | Mistral + Grok* + Claude* | 10–15 |
| `maximum` | All configured | All configured | All configured | All configured | All configured | up to 30 |

\* = only included when that model's key is configured

**`standard`** (default) — current baseline behavior. One primary model per domain. Lowest cost, fastest.

**`thorough`** — two to three well-suited models per domain. Search-grounded models (Gemini + Perplexity) cover fact-check; argument integrity gets three independent perspectives. Recommended when you have Perplexity and Claude configured.

**`maximum`** — every configured model runs every domain. Domain weights sort signal from noise — a general-purpose model running red team at weight 1.0 contributes less than Grok at 1.2, so its findings are ranked lower but still present. Best coverage; ~3–5× the cost of standard.

Per-model `prompts:` overrides take precedence over the thoroughness preset for that model.

---

### Ensemble weighting

The consolidation pass uses a weighted scoring system to identify consensus findings (Section 1) and sort findings within each section (Sections 2–6).

**How it works:**

For each passage flagged by one or more models, the pipeline sums the weights of all models that flagged it. When the sum meets the `consensus_threshold`, the passage is promoted to Section 1 (Consensus). Findings within each section are sorted by source weight descending — higher-weight model findings appear first.

LanguageTool adds a partial vote (`lt_weight`) when it independently flagged the same passage.

**Built-in default weights:**

| Model | Default | fact_check | voice_style | completeness | argument_integrity | red_team |
|---|---|---|---|---|---|---|
| gemini | 1.0 | **1.5** | 1.0 | 1.0 | 1.0 | 1.0 |
| perplexity | 1.0 | **1.5** | 1.0 | 1.0 | 1.0 | 1.0 |
| openai | 1.0 | 1.0 | **1.2** | **1.2** | 1.0 | 1.0 |
| mistral | 1.0 | 1.0 | 1.0 | 1.0 | **1.2** | **1.1** |
| grok | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **1.2** |
| claude | 1.0 | 1.0 | **1.1** | **1.1** | **1.3** | 1.0 |

Gemini and Perplexity receive a 1.5× bonus for fact_check because their responses are grounded in live web sources. Claude receives a 1.3× bonus for argument_integrity based on observed reasoning depth.

**Consensus threshold logic (default 2.0):**

- Two general models (1.0 + 1.0 = 2.0) → consensus
- One grounded model alone (1.5 < 2.0) → not consensus; needs corroboration
- One grounded + one general (1.5 + 1.0 = 2.5) → consensus
- Two grounded models (1.5 + 1.5 = 3.0) → strong consensus

**Configuring weights:**

```yaml
ensemble:
  consensus_threshold: 2.0   # lower = more aggressive consensus detection
  lt_weight: 0.5             # LanguageTool partial vote

  weights:
    # Override only what you want to change.
    # Omitted keys use built-in defaults.
    gemini:
      fact_check: 2.0        # trust Gemini fact-check findings more strongly
    openai:
      default: 0.8           # lower all OpenAI domain weights
      voice_style: 1.5       # but keep voice_style at 1.5
```

---

### Customizing a preset (preset_overrides)

`preset_overrides` lets you adjust individual fields of a cost preset without specifying the entire config from scratch. Think of it as: "use `balanced`, but change these specific things."

```yaml
pipeline:
  cost_preset: balanced
  preset_overrides:
    openai:
      reasoning_effort: high     # balanced uses "low"; bump to "high"
    claude:
      model: claude-opus-4-8     # balanced uses sonnet; use opus instead
      effort: high
      thinking_budget: null      # null out sonnet's extended thinking budget
    grok:
      enabled: false             # skip Grok for this run
```

**How it works:**
1. The preset is applied first (sets model + reasoning + thoroughness for all providers)
2. `preset_overrides` is then applied on top — only the keys you list are changed
3. Keys you don't mention keep the preset's value
4. Setting a key to `null` neutralizes it (adapters guard with `if value:`, so `null` disables it)
5. Only providers already configured in `models:` are affected — overrides for unconfigured providers are silently ignored

**What you can override per provider:**

| Key | Applies to | Values |
|---|---|---|
| `model` | all | any model ID string |
| `reasoning_effort` | openai, grok | `none \| low \| medium \| high \| xhigh` |
| `reasoning_effort` | mistral (`mistral-medium-3-5` only) | `"high"` or `"none"` only — `"low"`/`"medium"` return 400 |
| `thinking_budget` | gemini, claude-haiku | integer (tokens) or `null` to disable |
| `effort` | claude (opus, sonnet, fable) | `low \| medium \| high` |
| `enabled` | all | `true \| false` |
| `timeout_seconds` | all | integer (seconds) |
| `prompts` | all | list of domain names |

**Common recipes:**

```yaml
# Use thorough preset but stay on gpt-5.4 not gpt-5.5:
pipeline:
  cost_preset: thorough
  # (thorough already uses gpt-5.4, so no override needed here)

# Use maximum preset but save money by keeping gemini on 2.5-flash:
pipeline:
  cost_preset: maximum
  preset_overrides:
    gemini:
      model: gemini-2.5-flash   # maximum would use gemini-3.5-flash
      thinking_budget: 8192     # but keep the large thinking budget

# Use balanced but push Mistral to medium reasoning:
pipeline:
  cost_preset: balanced
  preset_overrides:
    mistral:
      reasoning_effort: medium  # balanced uses "low"

# Use standard but add Perplexity reasoning (normally standard uses sonar-pro):
pipeline:
  cost_preset: standard
  preset_overrides:
    perplexity:
      model: sonar-reasoning-pro
```

---

### Live model discovery

Run `discover.py` any time you want to check whether newer models are available from any provider — without reading every provider's changelog yourself. The script calls each provider's live models API using your existing API keys.

```
python discover.py
python discover.py --provider openai
python discover.py --provider gemini --provider claude
```

**Example output:**
```
Model Discovery Report — 2026-06-18
Built-in registry last updated: 2026-06-18 (0 days ago)
======================================================================

OpenAI  (configured: gpt-5.4)
   NEW  gpt-5.5        2026-01-15  (5mo ago)  ← newer than configured
    ✓   gpt-5.4        2025-11-20  (7mo ago)  ← configured
        gpt-5.4-mini   2025-11-20  (7mo ago)
    ⚠   gpt-4o         2024-05-13  (1.1yr ago)  ⚠ superseded → gpt-5.4

Gemini  (configured: gemini-2.5-flash via Vertex AI)
  SKIP  Gemini is configured via Vertex AI — model listing not supported here.
        Check https://ai.google.dev/models for available Gemini models.

Anthropic / Claude  (configured: claude-opus-4-8)
   NEW  claude-fable-5      2026-03-01  (4mo ago)  ← newer than configured
    ✓   claude-opus-4-8     2025-10-10  (8mo ago)  ← configured
        claude-sonnet-4-6   2025-08-15  (10mo ago)
        claude-haiku-4-5-20251001  2025-08-01  (10mo ago)
```

**What each marker means:**
- `NEW` — model exists at the provider and has a creation date newer than your configured model
- `✓` — your currently configured model
- `⚠` — model ID appears in the built-in superseded registry
- (none) — available, not configured, not flagged

**Notes on Vertex AI:** Gemini via Vertex AI cannot be queried for model lists without the gcloud SDK. The script notes this and skips. Check [Google AI for Developers](https://ai.google.dev/models) manually.

**Notes on Perplexity:** Perplexity does not publish a models list API. The script shows the documented set as a static fallback.

**After discovery, to update your configured model:**
1. Edit `models:` in `configs/user.yaml` (or update the `cost_preset` which sets models automatically)
2. Run `python check.py --publication your_pub` to verify the new model responds
3. Optionally update `model_registry.py` to add the old model to `_SUPERSEDED`

---

### Model currency detection

Every pipeline run checks your configured model IDs against a built-in registry of known-current and superseded models. Results appear in the terminal summary and the saved report JSON.

**Three signal levels:**

| Signal | Condition | What it means |
|---|---|---|
| Superseded warning | Configured model ID is in the deprecated list | Model has a newer replacement; update user.yaml |
| Upgrade notice | Configured model has a newer variant available | Current model is fine; newer one exists if you want it |
| Registry staleness | Registry data is 60+ days old | Re-check provider docs for new releases |

**Example terminal output:**
```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODEL CURRENCY: Outdated model(s) detected — update user.yaml
  openai: 'gpt-4o' → replace with 'gpt-5.4' (GPT-5 family available (2026))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Note: model registry last updated 2026-06-18 (73 days ago). Consider re-checking for newer models.
```

**Keeping the registry current:**

The registry lives in [`configs/model_registry.yaml`](../configs/model_registry.yaml). After any provider model audit:
1. Add entries to `superseded:` for models that have been replaced
2. Update `newer_available:` for current models that have newer variants
3. Bump `registry_date:` to today's date

The registry date drives the staleness notice — bumping it resets the 60-day clock even if you haven't changed any model entries. No code change is needed; the pipeline reloads the YAML at each run.

---

## Publication config

Open `configs/your_publication_name.yaml`. Key fields:

| Field | What to put there |
|---|---|
| `publication_description` | One paragraph: what you cover, who reads it, what makes a piece unpublishable |
| `audience.primary` | Who reads it, what they know, what makes them stop reading |
| `voice_profile` | Your characteristic style — see PLAYBOOK.md for how to develop this |
| `style_rules.banned_words` | Words you never want in your published writing |
| `style_rules.banned_phrases` | Phrases you never want |
| `seo_rules.title_max_chars` | SEO title length ceiling (default 60) |
| `seo_rules.title_min_chars` | SEO title length floor (default 20) |
| `seo_rules.min_article_words` | Minimum word count before thin-content warning (default 300) |
| `wordpress.site_url` | `https://yoursite.com` |
| `wordpress.username` | Your WordPress login username |
| `wordpress.application_password` | The application password from your WordPress profile |

**`seo_rules`** is optional — omit it to use the defaults. Add it when your publication has different SEO standards from the defaults:

```yaml
seo_rules:
  title_max_chars: 55       # tighter ceiling for a publication with long-title history
  title_min_chars: 20
  min_article_words: 500    # longer minimum for long-form-only publication
```

You can use an environment variable for the application password:

```yaml
wordpress:
  application_password: ${WP_APPLICATION_PASSWORD}
```

See `configs/examples/` for complete worked examples.

---

## Delta assessment

When a prior run's report exists, the pipeline compares the new draft against it and recommends whether to re-run. Controls:

```yaml
delta:
  word_change_threshold_pct: 15        # re-run if >15% of words changed vs prior run
  claim_change_triggers_rerun: true    # re-run if the handoff PRIMARY CLAIM changed
  structure_change_triggers_rerun: true # re-run if the heading outline changed
```

A re-run is recommended when **any** of these is true: word change exceeds the threshold; a new consensus flag appeared; the `PRIMARY CLAIM` differs from the prior run (when `claim_change_triggers_rerun`); or the markdown heading outline was added to, removed from, renamed, or reordered (when `structure_change_triggers_rerun`).

- **Claim comparison** is whitespace- and case-insensitive, and only fires when both runs supplied a claim — reports from before claim tracking won't trigger a spurious re-run.
- **Structure comparison** looks only at headings (`#`–`######`), so body-only edits don't count as a structural change.

---

## URL input mode

Instead of a local handoff document, you can point the pipeline at an already
published web page:

```
python pipeline.py --url https://example.com/some-post --publication your_publication_name
```

`--url` is mutually exclusive with `--draft` and `--publish`, and still requires
`--publication` (the publication config supplies the voice profile, audience,
and style rules the review prompts need).

### How a handoff is synthesized

A normal draft run reads a handoff document and parses many fields
(`PRIMARY CLAIM`, `PRE-DRAFT ANALYSIS SUMMARY`, `TARGET AUDIENCE`, etc.). The
pipeline body only *strictly* needs the article **title** and **draft body** —
everything else is optional. URL mode therefore builds an in-memory handoff with
just those two fields plus `run_number: 1`, then feeds it into the exact same
review path:

```python
{"title": "<page title>", "draft": "<extracted article text>", "run_number": 1}
```

### What it can and can't infer

| Field | URL mode |
|---|---|
| `title` | Page `<title>`, or the first `<h1>` if there's no title tag |
| `draft` | Extracted main-article markdown (headings preserved) |
| `primary_claim` | **Not inferred** — empty; the delta "claim changed" check won't fire |
| `pre_draft_analysis` | **Not inferred** — argument/completeness models get less context |
| `target_audience` / `additional_context` | **Not inferred** — comes only from the publication config |
| `seo` | **Not inferred** — there's no author-supplied SEO block |

If you want the richer author-intent context, use a `--draft` handoff instead.

### Fetching and extraction

- **SSRF guard.** Only public hosts are fetched. The same check used for link
  validation (`analysis/links.py`) rejects loopback, private, link-local, and
  cloud-metadata (`169.254.169.254`) addresses *before* any request is made.
- **Extraction.** If [`trafilatura`](https://trafilatura.readthedocs.io/) is
  installed it's used for main-content extraction (`pip install trafilatura`).
  Otherwise a built-in heuristic strips `<script>`/`<style>`/`<nav>`/`<header>`/
  `<footer>`/`<aside>`, prefers the `<article>` or `<main>` element, and keeps
  `<h1>`–`<h3>` as markdown headings (the SEO and structure checks key off
  heading markup).
- **Thin-extraction warning.** If fewer than ~200 words are extracted, the run
  warns loudly — usually a paywall, a JavaScript-rendered page, or a bot-block —
  and proceeds on whatever content was recovered.
