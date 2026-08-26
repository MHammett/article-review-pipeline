# Provider Setup

API keys and account setup for every service the pipeline can use.

**Required:** OpenAI, Gemini, Mistral  
**Optional:** Perplexity, Grok, Claude, LanguageTool, WordPress

Skip optional sections for services you don't plan to use. The pipeline detects missing keys and skips those passes automatically — nothing will break if a section is absent.

---

## OpenAI (required)

**Where:** https://platform.openai.com

**Steps:**
1. Create an account at https://platform.openai.com/signup
2. Go to API keys: https://platform.openai.com/api-keys
3. Click **Create new secret key**, give it a name, and copy it immediately — shown once only
4. Add a payment method: https://platform.openai.com/account/billing
5. Load a minimum of $5 in credits to start

**What you need:**
- API key: the key you just created

**Current models (June 2026):**

| Model | Input | Output | Context | Notes |
|---|---|---|---|---|
| `gpt-5.5` | $5.00/MTok | $30/MTok | 1M tokens | Highest capability, reasoning defaults to medium |
| `gpt-5.4` | $2.50/MTok | $15/MTok | 1M tokens | Best value flagship; **recommended default** |
| `gpt-5.4-mini` | $0.75/MTok | $4.50/MTok | 400K tokens | Economy option |

Reasoning is controlled via `reasoning_effort: none | low | medium | high | xhigh` in the model config. Default is `medium` on gpt-5.5; omit for model defaults.

**Expected cost:** ~$0.04–$0.10 per call at default settings.

**What it does:** Voice/style review (detects AI-generated phrasing, hedging language, banned words) and completeness analysis (finds gaps a technically literate critic would notice). Optional web-search upgrade — see [CONFIGURATION.md](CONFIGURATION.md#openai-web-search).

---

## Google Gemini (required)

Gemini runs the fact-check pass with live Google Search grounding. There are two access paths. Start with AI Studio; move to Vertex AI if you hit consistent 503 capacity errors.

### Option A — AI Studio (quick start, free tier available)

**Where:** https://aistudio.google.com

**Steps:**
1. Sign in with a Google account
2. Click **Get API key** in the left sidebar
3. Click **Create API key**
4. Copy the key shown

**What you need:**
- API key: the key you just generated

**Billing:** No credit card required. At one article per month you will stay within the free tier for Gemini Flash.

**Limitation:** AI Studio draws from a shared capacity pool. At peak hours you may get 503 errors. If that happens consistently, use Vertex AI instead.

**Current models (June 2026):**

| Model | Input | Output | Notes |
|---|---|---|---|
| `gemini-3.5-flash` | $1.50/MTok | $9.00/MTok | Latest generation, stable |
| `gemini-2.5-flash` | $0.30/MTok | $2.50/MTok | Best price-performance; **recommended default** |
| `gemini-2.5-pro` | $1.25/MTok | $10/MTok | More thorough, higher cost |

All Gemini models support `thinking_budget` for controlling reasoning token allocation. Set `0` to disable thinking (faster/cheaper). Gemini 2.5 Flash already thinks dynamically by default; a budget applies a ceiling.

**Config:**
```yaml
models:
  gemini: gemini-2.5-flash
```

---

### Option B — Vertex AI (reserved capacity, no 503s)

Vertex AI uses a separate capacity pool not shared with free-tier traffic.

**Steps:**
1. Create or select a GCP project at https://console.cloud.google.com
2. Enable billing — Vertex AI calls fail without it even if the API is enabled
3. Enable the Agent Platform API (formerly Vertex AI API). Either run:
   ```
   gcloud services enable aiplatform.googleapis.com
   ```
   Or open https://console.cloud.google.com/apis/library/aiplatform.googleapis.com and click Enable

**Credentials — easiest for local use (Application Default Credentials):**

Install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install

On Windows, use the Google Cloud CLI Installer (`.exe`) linked on that page — no WSL required. It adds `gcloud` to your PATH and works in Command Prompt and PowerShell. Then run:

```
gcloud auth application-default login
```

This opens a browser to complete sign-in and stores credentials in your user profile. No file to manage.

**Credentials — service account (for servers or CI):**
1. Go to **IAM & Admin → Service Accounts** in your GCP project
2. Click **Create Service Account**, give it a name, click **Create and Continue**
3. In the "Grant this service account access to project" step, search for **Agent Platform User**, select it, click **Continue**, then **Done**
4. Click the service account in the list to open it
5. Go to the **Keys** tab
6. Click **Add Key → Create new key**
7. Select **JSON** and click **Create** — the key file downloads to your Downloads folder
8. Move it somewhere permanent, e.g.:
   ```
   C:\Users\your-username\gcp-keys\my-project-vertex.json
   ```
9. Reference it in `configs/user.yaml`:
   ```yaml
   models:
     gemini:
       provider: vertex_ai
       model: gemini-2.5-flash
       project: your-gcp-project-id
       location: us-central1
       credentials_file: C:\Users\your-username\gcp-keys\my-project-vertex.json
   ```

**Install the auth library:**
```
pip install "google-auth>=2.22.0,<3.0"
```

**What you need:**
- GCP **project ID** — the unique identifier like `my-project-123`, not the display name. Find it in the project selector dropdown in the GCP console (smaller text below the project name, also visible in the browser URL).
- Region (e.g. `us-central1`)
- ADC set up, or a service account JSON file

**Config (ADC, no file needed):**
```yaml
models:
  gemini:
    provider: vertex_ai
    model: gemini-2.5-flash
    project: your-gcp-project-id
    location: us-central1
```

**Billing:** Same per-token price as AI Studio paid tier. Under $0.10/month at one article per month.

---

## Mistral AI (required)

**Where:** https://console.mistral.ai

**Steps:**
1. Create an account
2. Go to **API keys** in the left sidebar
3. Click **Create new key**, name it, copy the key
4. Add a payment method under **Billing**

**What you need:**
- API key: the key you created

**Current models (June 2026):**

| Model | Notes |
|---|---|
| `mistral-large-latest` | Flagship non-reasoning model; **recommended default** |
| `mistral-medium-3-5` | Reasoning model; replaces deprecated `magistral-medium-latest` |
| `mistral-small-latest` | Economy option; reduced depth |

**Reasoning constraints (important):** `mistral-medium-3-5` is the only Mistral model that supports `reasoning_effort`. It only accepts `"high"` or `"none"` — `"low"` and `"medium"` return a 400 error. Standard models (`mistral-large-latest`, `mistral-small-latest`) reject `reasoning_effort` entirely. The `-latest` suffix variant (`mistral-medium-3-5-latest`) does not exist and returns a 400 error.

```yaml
# Standard (no reasoning)
models:
  mistral: mistral-large-latest

# Reasoning
models:
  mistral:
    model: mistral-medium-3-5
    reasoning_effort: high      # "high" or "none" only
    timeout_seconds: 240
```

**Expected cost:** ~$0.03–$0.10 per call for standard; higher for reasoning on long articles.

**What it does:** Argument integrity review (logical gaps, unstated assumptions, conclusions that outrun their evidence) and red team analysis (most attackable claim, audience alienation risk, credibility risk). European company, architecture independent from Google and OpenAI — independent analytical perspective matters.

---

## Perplexity AI (optional — recommended)

Perplexity's sonar models run every response through live web search by default. Adding a Perplexity key gives you a second independent search-grounded fact-checker alongside Gemini. Both carry a 1.5× weight in consensus scoring.

**Where:** https://www.perplexity.ai/settings/api

**Steps:**
1. Create an account at https://www.perplexity.ai
2. Go to **Settings → API**
3. Click **Generate** under API Keys
4. Copy the key

**What you need:**
- API key: the key you generated

**Current models (June 2026):**

| Model | Notes |
|---|---|
| `sonar-reasoning-pro` | CoT reasoning + web search; **recommended default** |
| `sonar-pro` | Web search, no CoT trace; good for standard tier |
| `sonar` | Lightweight; economy option |
| `sonar-deep-research` | Extended research; highest cost and latency |

Perplexity's reasoning is model-selection based — use `sonar-reasoning-pro` for CoT, `sonar-pro` for standard search grounding. There is no separate reasoning parameter.

**Expected cost:** ~$0.04–$0.20 per call depending on model (sonar-reasoning-pro has higher variance due to reasoning trace overhead).

**What it adds:** At `standard` thoroughness, Perplexity only runs if you explicitly add it to a model's `prompts:` list. At `thorough` thoroughness, it runs fact_check automatically alongside Gemini. Two independently grounded models flagging the same claim is very strong signal.

**Config:**
```yaml
api_keys:
  perplexity:
    api_key: your_key_here

models:
  perplexity: sonar-reasoning-pro
```

---

## Grok / xAI (optional)

**Where:** https://console.x.ai

**Steps:**
1. Create an account (sign in with X/Twitter or email)
2. Go to **API keys** and generate a key
3. Copy the key

**What you need:**
- API key: the key you generated

**Current models (June 2026):**

| Model | Input | Output | Notes |
|---|---|---|---|
| `grok-4.3` | $1.25/MTok | $2.50/MTok | General purpose; **recommended default** |
| `grok-4.20-0309-reasoning` | $1.25/MTok | $2.50/MTok | Reasoning variant; same price as standard |
| `grok-4.20-0309-non-reasoning` | $1.25/MTok | $2.50/MTok | Explicit non-reasoning variant |
| `grok-build-0.1` | $1.00/MTok | $2.00/MTok | Economy fallback |

Grok reasoning is **model-selection based** — use `grok-4.20-0309-reasoning` for CoT. Unlike OpenAI/Mistral/Claude, there is no reasoning parameter; you switch models. Since the reasoning model costs the same as the standard model, there is no reason not to use it at `balanced` and above.

**Billing:** xAI currently offers free tier credits. At low article volume you may stay within free limits. Check https://console.x.ai for current pricing.

**What it adds:** A second red team pass. Grok is trained on a different corpus (heavy X/Twitter data) and tends toward more direct, contrarian responses — useful for attack angles the other models miss. At `standard` thoroughness, both Mistral and Grok red team results appear in Section 6 of the report.

---

## Anthropic (Claude) (optional)

**Where:** https://console.anthropic.com

**Steps:**
1. Create an account at https://console.anthropic.com
2. Go to **API Keys** in the left sidebar
3. Click **Create Key**, give it a name, copy it immediately — shown once only
4. Add a payment method under **Billing** and load a minimum of $5

**What you need:**
- API key: the key you created

**Current models (June 2026):**

| Model | Input | Output | Context | Thinking |
|---|---|---|---|---|
| `claude-opus-4-8` | $5/MTok | $25/MTok | 1M | Adaptive (always on; control via `effort`) |
| `claude-fable-5` | $10/MTok | $50/MTok | 1M | Adaptive always-on, not configurable |
| `claude-sonnet-4-6` | $3/MTok | $15/MTok | 1M | Adaptive (always on; control via `effort`) |
| `claude-haiku-4-5-20251001` | $1/MTok | $5/MTok | 200K | Extended (`thinking_budget`) opt-in only |

**Thinking modes — important distinction:**

- **Adaptive thinking** (Opus 4.8, Fable 5, Sonnet 4.6): reasoning is always on and model-controlled. Add `effort: low/medium/high` to control depth. Do NOT set `thinking_budget` on these models — extended thinking is deprecated for them.
- **Extended thinking** (Haiku 4.5): opt-in via `thinking_budget: N`. Only Haiku 4.5 uses this mode.

**Important for `fact_check`:** Claude has no live web search capability. The `fact_check` prompt requires it and Claude will always fail or produce non-JSON output. Add `prompts: [voice_style, completeness, argument_integrity, red_team]` to your Claude config to exclude `fact_check`. This setting survives `cost_preset` overrides. See [CONFIGURATION.md](CONFIGURATION.md#restricting-which-prompts-a-model-runs).

For argument integrity depth, `claude-opus-4-8` with `effort: high` is the recommended choice. For cost efficiency, `claude-sonnet-4-6` with `effort: medium` gives strong reasoning at lower per-call cost.

**Expected cost:** ~$0.01–$0.10 per call depending on model and thinking mode.

**What it adds:** A second argument integrity pass. Claude's training lineage is independent from the rest of the stack and it tends to catch logical gaps the other models miss. At `standard` thoroughness, both Mistral and Claude argument results are merged into Section 4.

---

## LanguageTool (optional)

The grammar correction pass applies deterministic rule-based corrections before the AI review passes, so the models aren't distracted by surface errors. It's the only component that modifies your draft without asking.

Skip it if you already do a manual, thorough pass yourself (e.g. Grammarly Premium) — you're covering the same ground.

**To skip:** Set `grammar_pass: false` in `configs/user.yaml`, or simply omit the `languagetool` credentials block. The pipeline skips automatically and reminds you to run a manual check.

Two ways to use it, at different cost:

### Self-hosted (free)

Run the open-source LanguageTool server yourself and point the pipeline at it — no LanguageTool account, no `username`/`api_key`.

```bash
docker run -d -p 8010:8010 erikvl87/languagetool
```

```yaml
api_keys:
  languagetool:
    server_url: http://localhost:8010/v2/check
```

**Trade-off:** this runs the open-source Community rule set, not the fuller Premium one. Verified 2026-08-17 against languagetool.org's own account settings page (`/editor/settings/access-tokens`): Premium's "Access Tokens" there are scoped to *native app integrations* (Obsidian, LibreOffice, their browser add-on), not general programmatic access — so self-hosting isn't giving up something the $4.99–19.99/mo personal tier would otherwise unlock. See the note below.

### Hosted API (paid)

**This is not the same product as the $4.99–19.99/mo personal "Premium" subscription.** That tier's account settings page frames its own API tokens as being for LanguageTool's supported native integrations, not general HTTP access — despite the site being genuinely unclear about this. Programmatic access to the full rule set (what `username`/`api_key` credentials against `api.languagetool.org` require) is a separate commercial product, the [Proofreading API](https://languagetool.org/proofreading-api), starting around $40/month.

**To use:** Sign up for the Proofreading API, get a `username`/`api_key` pair, and configure:

```yaml
api_keys:
  languagetool:
    username: your_email@example.com
    api_key: your_languagetool_key
```

---

## WordPress Application Password

**Where:** Your WordPress admin dashboard

**Steps:**
1. Log in to your WordPress admin
2. Go to **Users → Profile** (or **Users → Your Profile**)
3. Scroll to the **Application Passwords** section
4. Type a name like `article-pipeline` and click **Add New Application Password**
5. Copy the password shown immediately — it will not be displayed again
6. Note your site URL and WordPress username

**Verify it works:** Visit `https://yoursite.com/wp-json/wp/v2` in a browser. A JSON response means the REST API is active. A 404 means the REST API is disabled — go to Settings → Permalinks in your WordPress admin and click Save Changes to rebuild the rewrite rules.

**What you need:**
- Site URL: `https://yoursite.com`
- Username: your WordPress login username
- Application password: the password generated above (spaces included, as shown)

Use an application password rather than your login password — it is scoped, can be revoked individually, and never exposes your main account credentials.
