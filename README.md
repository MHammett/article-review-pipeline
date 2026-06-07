# Article Review Pipeline

A multi-pass automated review pipeline for published web content. Takes a drafted article through deterministic grammar correction and structured multi-model AI review, consolidates feedback into a single prioritized report, and publishes to WordPress when you approve.

At one article per month, total API cost is under $1.00. The only meaningful fixed cost is LanguageTool Premium at $4.99/month.

---

## Requirements

- Python 3.10 or higher — https://www.python.org/downloads/
- Git — https://git-scm.com/downloads

---

## Installation

```bash
git clone https://github.com/MHammett/article-review-pipeline.git
cd article-review-pipeline
pip install -r requirements.txt
```

To also install test dependencies:

```bash
pip install -r requirements-dev.txt
```

Verify the install:

```bash
python pipeline.py --help
```

---

## Account Setup and API Keys

You need accounts with three services before the pipeline can run (OpenAI, Gemini, Mistral). Grok and LanguageTool are optional — see below.

---

### Grok / xAI (optional)

**Where:** https://console.x.ai

**Steps:**
1. Create an account (sign in with X/Twitter or email)
2. Go to API keys and generate a key
3. Copy the key

**What you need:**
- API key: the key you generated

**Billing:** xAI currently offers free tier credits. At low article volume you may stay within free limits indefinitely. Check https://console.x.ai for current pricing.

**What it adds:** When a Grok API key is present, the pipeline runs a second red team pass using Grok in addition to Mistral's red team pass. Grok is trained on a different corpus (heavy X/Twitter data) and tends toward more direct, contrarian responses — useful for finding attack angles the other models miss. Both red team results appear in Section 6 of the report. If no key is present, the pipeline runs exactly as before with Mistral's red team only.

---

### LanguageTool (optional)

The grammar correction pass is useful but not required. If you already do a manual Grammarly pass before publishing, you are covering the same ground. Skip LanguageTool if the cost doesn't justify the automation at your volume.

**To skip it:** set `grammar_pass: false` in `configs/user.yaml`, or simply omit the `languagetool` credentials block. The pipeline detects missing credentials and skips the pass automatically. The summary will remind you to run a manual grammar check before publishing.

**To use it:** LanguageTool offers a Premium API. Go to https://languagetool.org, create an account, subscribe, go to account settings, and generate an API key. Copy your API key and the email address you signed up with.

**What you need (if using):**
- Username: your email address
- API key: from account settings

**Why it's here when used:** LanguageTool applies deterministic rule-based corrections before the AI review passes, so the models aren't distracted by surface errors. It's the only component that writes to your draft without asking first.

---

### Google AI Studio (Gemini)

**Where:** https://aistudio.google.com

**Steps:**
1. Sign in with a Google account
2. Click "Get API key" in the left sidebar
3. Click "Create API key"
4. Copy the key shown — you won't be able to see it again from this screen, but you can generate a new one

**What you need:**
- API key: the key you just generated

**Billing:** No credit card required to start. At one article per month you will stay within free tier limits for Gemini Flash. Billing only activates if you exceed free tier.

**Why it's here:** Gemini has native Google Search grounding, meaning it checks your factual claims against live web sources during the review pass. No other major model API includes this by default.

---

### OpenAI

**Where:** https://platform.openai.com

**Steps:**
1. Create an account at https://platform.openai.com/signup
2. Go to API keys: https://platform.openai.com/api-keys
3. Click "Create new secret key", give it a name, and copy it immediately — it is only shown once
4. Add a payment method: https://platform.openai.com/account/billing
5. Load a minimum of $5 in credits to start

**What you need:**
- API key: the key you created above

**Expected cost:** At one article per month, approximately $0.10–$0.30 per run depending on article length. $5 will last several months.

**Why it's here:** The `gpt-4o` model handles voice/style review and completeness analysis. It is well-calibrated for detecting AI-generated phrasing and hedging language.

---

### Mistral AI

**Where:** https://console.mistral.ai

**Steps:**
1. Create an account
2. Go to API keys in the left sidebar
3. Click "Create new key", name it, and copy the key shown
4. Add a payment method under Billing

**What you need:**
- API key: the key you created above

**Expected cost:** At one article per month, approximately $0.05–$0.15 per run. Very low volume.

**Why it's here:** Mistral is a European company with an architecture independent from both Google and OpenAI. It tends toward harder logical scrutiny, making it well-suited to argument integrity review. Three independent analytical perspectives matter.

---

### WordPress Application Password

**Where:** Your WordPress admin dashboard

**Steps:**
1. Log in to your WordPress admin dashboard
2. Go to **Users → Profile** (or **Users → Your Profile**)
3. Scroll down to the **Application Passwords** section
4. In the "New Application Password Name" field, type something like `article-pipeline`
5. Click **Add New Application Password**
6. Copy the password shown immediately — it will not be displayed again
7. Note your WordPress site URL (e.g., `https://yoursite.com`)
8. Note your WordPress username (the one you used to log in)

**Verify it works:** Visit `https://yoursite.com/wp-json/wp/v2` in a browser. You should see a JSON response. If you get a 404, the REST API may be disabled — check with your host or under Settings → Permalinks (re-saving permalinks often fixes this).

**What you need:**
- Site URL: `https://yoursite.com`
- Username: your WordPress login username
- Application password: the password generated above (spaces included, as shown)

**Why application passwords instead of your login password:** Application passwords are scoped and can be revoked individually without changing your main password. Never use your main WordPress password in a script.

---

## Configuration

### Step 1: Copy the example files

```bash
cp configs/user.example.yaml configs/user.yaml
cp configs/publication.example.yaml configs/your_publication_name.yaml
```

The `.gitignore` is already set to exclude `configs/user.yaml` and any `configs/*.yaml` files that aren't examples, so your keys will not be committed.

### Step 2: Fill in user.yaml

Open `configs/user.yaml` and add your API keys:

```yaml
api_keys:
  languagetool:
    username: your_email@example.com
    api_key: your_languagetool_key

  openai:
    api_key: sk-...

  gemini:
    api_key: AI...

  mistral:
    api_key: your_mistral_key
```

**Alternative — environment variables:** Instead of putting keys directly in the YAML, you can reference environment variables:

```yaml
api_keys:
  openai:
    api_key: ${OPENAI_API_KEY}
```

Copy `.env.example` to `.env` and fill in the values. The pipeline loads `.env` automatically on startup.

### Step 3: Fill in your publication config

Open `configs/your_publication_name.yaml`. The fields that require your input:

| Field | What to put there |
|---|---|
| `publication_description` | One paragraph: what you cover, who the reader is, what would make a piece unpublishable |
| `audience.primary` | Who reads it, what they know, what they'd stop reading over |
| `voice_profile` | Your characteristic style — see the PLAYBOOK.md Prerequisites section for how to develop this |
| `style_rules.banned_words` | Words you never want in your published writing |
| `style_rules.banned_phrases` | Phrases you never want |
| `wordpress.site_url` | `https://yoursite.com` |
| `wordpress.username` | Your WordPress login username |
| `wordpress.application_password` | The application password from the step above |

For `wordpress.application_password`, you can use an environment variable instead of putting it in the file:

```yaml
wordpress:
  application_password: ${WP_APPLICATION_PASSWORD}
```

See `configs/examples/` for complete worked examples across three publication types.

---

## Usage

### Review a draft

```bash
python pipeline.py --draft path/to/handoff.md --publication your_publication_name
```

Fill out `handoff_templates/draft_submission.md` and pass it as the `--draft` argument. The publication name is the filename of your config without `.yaml`.

### Publish an approved draft

```bash
python pipeline.py --publish path/to/publication_handoff.md --publication your_publication_name
```

Fill out `handoff_templates/publication.md`. This always saves as a WordPress draft unless you add `--publish-live`.

### Options

```
--verbose, -v       Enable debug logging (shows per-call timing, raw errors)
--config-dir        Path to config directory (default: configs/)
--publish-live      Publish directly to live instead of saving as draft
```

---

## Running the tests

```bash
pytest tests/
```

All tests mock external API calls — no API keys required to run the test suite.

---

## Project structure

```
article-review-pipeline/
├── pipeline.py                   orchestration engine — start here
├── config_loader.py              config parsing and validation
├── consolidation.py              merges five model responses into one report
├── handoff_parser.py             parses Template A and Template C documents
├── history.py                    saves run artifacts to pipeline_history/
├── requirements.txt              runtime dependencies
├── requirements-dev.txt          adds pytest for development
├── .env.example                  documents all supported environment variables
│
├── adapters/
│   ├── grammar/languagetool.py   grammar correction (Pass 1)
│   ├── review/gemini.py          fact verification with live search
│   ├── review/openai.py          voice/style and completeness review
│   ├── review/mistral.py         argument integrity and red team
│   ├── cms/wordpress.py          WordPress REST API publisher
│   └── citation/                 primary source resolution and checksums
│
├── prompts/                      system prompts for each review pass
├── configs/                      your API keys and publication settings (gitignored)
├── handoff_templates/            fill these out to submit drafts and publish
├── pipeline_history/             run reports saved here (gitignored, local only)
└── tests/                        39 tests, all external calls mocked
```

---

## Troubleshooting

**`User config not found`** — You haven't created `configs/user.yaml` yet. Copy `configs/user.example.yaml` to `configs/user.yaml` and fill in your keys.

**`Environment variable X is not set`** — You used `${VAR_NAME}` syntax in a config but the variable isn't in your `.env` file or shell environment.

**`Invalid publication name`** — Publication names can only contain letters, numbers, hyphens, and underscores. No slashes or spaces.

**`No DRAFT section found`** — Your handoff document doesn't have a line that reads exactly `DRAFT` followed by the article text. Use the template in `handoff_templates/draft_submission.md`.

**WordPress returns 404** — Visit `https://yoursite.com/wp-json/wp/v2` in a browser. If that 404s, go to Settings → Permalinks in your WordPress admin and click Save Changes. This rebuilds the rewrite rules that enable the REST API.

**WordPress returns 401** — Your application password is wrong or the username doesn't match. Re-generate the application password in Users → Profile.

**LanguageTool returns 401** — Your API key or username is wrong. Log in to languagetool.org and check your account settings page. Or set `grammar_pass: false` to skip the grammar pass entirely.

**A model pass timed out** — The default per-task timeout is 180 seconds. If a model is consistently slow, increase `pipeline.task_timeout_seconds` in `configs/user.yaml`. The pipeline continues with whatever results it has and notes the timeout in the report.
