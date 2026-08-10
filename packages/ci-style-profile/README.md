# Style Profile Bootstrap

Analyzes your writing corpus across multiple platforms and synthesizes a structured style profile for use in `publication.yaml`.

> **"voice" vs. "style":** these are distinct terms here — "voice" = a distinct authorial persona (what the detection step finds), "style" = the writing fingerprint and the published profile. The detection prompts keep "voice" on purpose. See [docs/TERMINOLOGY.md](../../docs/TERMINOLOGY.md).

## Prerequisites

- Python 3.10+ and [uv](https://docs.astral.sh/uv/) — the package is a member of the
  workspace at the repository root, installed by `uv sync` there
- API keys in `configs/user.yaml` (same as the review pipeline — at least one model required)
- For Twitter: API v2 Basic tier ($100/month) for general corpus access; Free tier is limited to your own recent tweets (last 7 days)

## Setup

All paths below are relative to the workspace root. Dependencies come from the
workspace, not this package — there's no separate install step.

```bash
# From the repository root — installs the whole workspace into .venv/
uv sync

# Create sources.yaml from the template.  Note the two different directories:
# the template ships at the package root, but the loader reads it from the
# source tree next to bootstrap.py.
cp packages/ci-style-profile/sources.example.yaml \
   packages/ci-style-profile/src/ci_style_profile/sources.yaml
# Edit sources.yaml — fill in your site URL and credentials

# Set credentials as environment variables (or use .env file)
export WP_USER=your_wordpress_username
export WP_APPLICATION_PASSWORD=your_app_password
```

## Usage

### 1. Dry run first (free for canonical/per-source; one API call for detect)

```bash
# Zero cost — collect, normalize, print stats
uv run style-profile-bootstrap \
  --publication mikehammett \
  --sources wordpress \
  --style canonical \
  --dry-run

# Detect mode dry-run runs one detection API call (not free)
uv run style-profile-bootstrap \
  --publication mikehammett \
  --sources wordpress \
  --style detect \
  --dry-run
```

### 2. Full run

```bash
uv run style-profile-bootstrap \
  --publication mikehammett \
  --sources wordpress,gmail \
  --style detect \
  --preset balanced
```

### 3. Write to a specific file

```bash
uv run style-profile-bootstrap \
  --output-yaml my_profile.yaml \
  --sources textfiles \
  --style canonical
```

## CLI Reference

```
--publication NAME       Resolve to configs/<name>.yaml (mutually exclusive with --output-yaml)
--output-yaml PATH       Explicit output path
--sources SRC[,SRC...]   Comma-separated source names (default: all in sources.yaml)
--since DATE             ISO date; applied at API level where supported
--style MODE             canonical | detect | per-source (overrides preset)
--max-styles N           Max detected style count (detect mode only; 0 = no limit)
--preset PRESET          economy | standard | balanced | thorough | maximum (default: balanced)
--refresh                Ignore staging cache; re-fetch all sources
--dry-run                Collect + normalize + stats only (detect: runs detection pass)
--no-stage               Don't write staging files; process in memory only
--continue-on-error      Skip failed collectors; continue with remaining sources
--format FORMAT          yaml | markdown | json (default: yaml)
--overwrite              Skip confirmation prompt when merging into existing file
--log-level LEVEL        DEBUG | INFO | WARNING | ERROR (overrides sources.yaml)
--check-draft PATH       Not implemented — accepted by the parser, does nothing yet
```

## Style Modes

| Mode | Description | API calls | Cost |
|------|-------------|-----------|------|
| `canonical` | Single unified style profile | M + 1 | Low |
| `detect` | Discover N distinct styles automatically | D + 1 + (N×M) + 1 | Medium |
| `per-source` | Separate profile per source type | (G×M) + 1 | Medium |

M = configured models, D = detection models, N = detected styles, G = source groups

## Output Format

The synthesized profile is written into your `configs/<publication>.yaml` as YAML, preserving all existing non-style sections. Style sections added:

```yaml
style_profile: |
  Prose description of the author's style...

audience:
  primary: Who this author writes for
  secondary: Secondary audience (if any)

style_rules:
  banned_words: [utilize, leverage, synergy]
  banned_phrases: [at the end of the day]
  positive_rules:
    - Lead with the main claim, then support it
    - Use concrete examples over abstractions

# In detect/per-source modes:
style_profiles:
  technical analysis:
    style_profile: |
      What's distinctive about this style...
    additional_banned_words: [basically]
    additional_positive_rules: [Use numbered lists]
    source_distribution: {wordpress: 0.85, textfiles: 0.15}
```

## Gmail OAuth Setup

The first time you use the Gmail source, you'll be prompted to authenticate:

1. Ensure `credentials_file` in `sources.yaml` points to your `credentials.json` (downloaded from Google Cloud Console)
2. Set OAuth credentials mode to `Desktop application`
3. Run bootstrap; a browser window opens for Google sign-in
4. After authorizing, a token file is saved alongside the credentials file

**Important:** The credentials file should have mode `600` (owner-read only):
```bash
chmod 600 ~/.config/style-bootstrap/gmail_credentials.json
```

## Email Source Privacy

Gmail and Outlook365 sources contain private email text. The tool takes these precautions:
- Staging files are gitignored (`packages/ci-style-profile/src/ci_style_profile/staging/`)
- Use `--no-stage` to never persist email text to disk (synthesizes in memory only)
- Be cautious with cloud sync (Dropbox, iCloud, etc.) on your home directory
- Staging files contain cleaned plain text, not raw emails

## Adding a Custom Collector

Drop a Python file in `packages/ci-style-profile/src/ci_style_profile/collectors/custom/`:

```python
from ci_style_profile.collectors.base import Collector, Document

class MyCustomCollector(Collector):
    SOURCE_NAME = "mycustom"

    @classmethod
    def validate_config(cls, config):
        # raise ConfigError on missing keys
        pass

    def fetch(self, since=None):
        # yield Document objects
        yield Document.from_text(...)
```

The collector is auto-discovered — no registration needed.

## Interpreting Results

**`confidence` field:** `high` = strong, well-separated styles with adequate corpus; `medium` = adequate but borderline; `low` = synthesis completed but quality uncertain.

**`synthesis_notes` log line:** Logged at INFO level, not written to YAML. Contains model agreement notes and detection observations.

**Profile versioning:** Every run saves a timestamped snapshot in `packages/ci-style-profile/src/ci_style_profile/profiles/<publication>/` (gitignored). With `--output-yaml` instead of `--publication`, snapshots go under `profiles/_output/<stem>/`. Diff consecutive snapshots to track how your profile evolves with more data.

## Presets

| Preset | Mode | Budget | Styles | Cost range |
|--------|------|--------|--------|------------|
| economy | canonical | 40k chars | — | ~$0.01–$0.05 |
| standard | detect | 80k chars | 3 | ~$0.10–$0.30 |
| balanced | detect | 120k chars | 5 | ~$0.50–$1.50 |
| thorough | detect | 160k chars | 7 | ~$1.50–$3.00 |
| maximum | detect | 200k chars | 10 | ~$3.00–$8.00 |
