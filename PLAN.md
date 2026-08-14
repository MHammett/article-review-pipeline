# Work plan — content-intelligence, 2026-08-12

Everything proposed in this session, in the order it should happen and with the
reason for the order. Nothing here is committed yet.

The organising principle: **litellm supersedes about a third of what was written
today.** Anything that touches the provider adapter layer is held until the
migration decision resolves. Anything above or beside that layer lands now.

---

## 0. Fix first — broken in production, independent of everything else

### 0a. Wayback rate limiting

The archive thread is dead and has been for at least two runs.

| run | resolved citations | archived | HTTP 429 |
|---|---|---|---|
| 2026-08-12 | 32 | 7 | 25 |
| 2026-08-11 | 49 | **0** | ~49 |

`archive.org/wayback/available` throttles at IP level with a long window: 12
consecutive requests all 429, including the first, and still 429 after a 45s
cooldown at 1 request / 6s. The pipeline has **no backoff, no pacing, and no
SPN2 credentials** — `wayback.submit()` accepts `access_key`/`secret_key` and
nothing supplies them.

Until this is fixed, two things already written are inert: the stale-snapshot
resubmission and the live+archive citation pairing will render "Archive: none"
for nearly everything.

**Do:** serialise availability checks with backoff, honour `Retry-After`, and
wire authenticated SPN2 credentials. Decide first whether to adopt
`waybackpy`/`savepagenow` (see UPSTREAM.md #3 — verify whether they already
handle this before writing our own).

### 0b. Claude is configured everywhere and never runs

The `maximum` preset lists `claude` for all five domains — 6 models x 5 domains =
**30 calls**. The 2026-08-12 run made **25**, and `claude` was never assigned.

It has a model in `configs/user.yaml`, an `api_key: ${CLAUDE_API_KEY}` reference,
and `.env` carries the key — which is valid: a direct Claude API call succeeded
during the prompt-caching test the same day. So the credential works and the
preset asks for it, but `_build_assignments` drops it.

Not yet pinned to a line. The two candidates are the `${...}` substitution not
resolving in `config_loader`, or the assignment filter's key-presence check.
**Start by logging what `_build_assignments` sees for each model.**

Impact: every `maximum` run has been paying maximum-preset prices for
five-sixths of the configured ensemble, silently.

Note when fixing: Claude drafting the article and then reviewing it correlates
blind spots, most acutely for `voice_style` — a model is a poor judge of its own
characteristic patterns, and `ai_speak.txt` exists precisely to catch those. The
reasoning domains (`fact_check`, `completeness`, `argument_integrity`) test
claims against the world rather than prose against a style, so the correlation
does not bite there. Consider per-domain weights rather than a flat enable.

### 0c. Cost model is blind to cached tokens

No mention of "cached" in `cost.py`, `tokens.py`, or `pricing.yaml`. Grok caches
automatically today and xAI prices cached input at **$2 vs $12.50 / MTok** — the
report bills it all at full rate. Consequence: **none of the caching work can be
measured**, including the prompt-cache layout that is currently switched off
awaiting exactly that measurement.

**Do:** add cached-token pricing. If the litellm migration goes ahead, do this
as part of it rather than twice — litellm carries its own cost tables and models
cached tokens already.

---

## 1. Land now — survives the litellm migration

These sit above or beside the adapter layer. Split from
`fix/grounding-coverage-and-run-quality`, which currently carries seven concerns
and ~1,850 lines. Land as separate PRs off master, sequentially, not stacked.

### PR 1 — Grok timeout budget
`configs/timeouts.yaml`, `tests/test_timeout_model.py`

Grok used 100% of its 126s budget on a 130k-char draft (completeness timed out;
two more domains finished within 14s of the ceiling). Every other provider
finished inside 39%. Multiplier 1.2 → 2.0, giving 210s. The calibration note now
records the measured budget utilisation for all five providers at the 150k size
bucket, which had never been exercised before.

### PR 2 — Same-provider call stagger
`pipeline.py` (`_stagger_offsets`, `_delay_start`), `tests/test_pipeline_timeout.py`

All 25 calls fired simultaneously, so Perplexity's five competed for one account
quota — two returned 429 within one second of each other and one failed
outright. Same-provider calls now start 3s apart; different providers still
start together. The offset is added to the budget, not taken from it.

### PR 3 — Citation claim deduplication
`pipeline.py` (`_claim_key`, `_is_duplicate_claim`), `tests/test_citation_claim_collection.py`

Dedup was exact-text, so five models paraphrasing one fact produced five claims:
29 near-duplicate pairs among 144, one differing from its twin only by a
trailing full stop. Now normalised, collapsing at ≥0.9 token overlap (144 → 137
on the real run). Threshold set high deliberately — 0.8 collapsed more but began
merging claims that differ in a material number.

**Contains a regression fix worth reviewing carefully:** the grounded-URL map is
now keyed by the same normalised key, because dedup collapsing a paraphrase
while the URL map used raw text made a registered source unreachable.

### PR 4 — Link checking
`analysis/links.py`, `ci_core/http.py`, `packages/ci-core/pyproject.toml`, `tests/test_links.py`

Three tiers: honest HEAD → honest GET on 403/405 → browser TLS fingerprint via
`curl_cffi` (new optional `unblock` extra, degrades to previous behaviour if
absent). Measured against six real blocked citations: browser *headers* alone
recovered 0/6 (the blocks are Cloudflare TLS fingerprinting), TLS impersonation
recovered 2/5 with real content, and the GET retry exposed one link reported as
"403, likely still valid" that is a genuine **404**.

The three academic publishers refuse everything and return a ~6 KB challenge
page — almost certainly subscription gates, not bot gates.

### PR 5 — Citation durability: live + archive pairing
`report_markdown.py`, `citation/resolver.py`, `tests/test_report_markdown.py`, `tests/test_resolver.py`

Snapshot URLs were collected on every run and rendered nowhere. Section 9 now
pairs every citation's live URL with its archive copy, distinguishing four
states (archived / stale / submitted this run / never archived) because each
needs different follow-up. Stale snapshots are now resubmitted for capture
alongside missing ones.

**Blocked on 0a for real effect.**

### PR 6 — Run bookkeeping and honest messages
`pipeline.py`, `history.py`, `consolidation.py`, `analysis/seo_suggest.py`,
`tests/test_history.py`, `tests/test_seo_suggest.py`, golden report

- Run numbers come from the handoff, so re-running wrote a second `run_16`.
  Collisions are detected and bumped with a warning to fix the handoff.
- Titles under 8 characters of content no longer claim a history directory
  (`pipeline_history/t/` and `title/` are how this showed up).
- The grammar-skip message named the wrong cause — it told an operator with
  working credentials in `.env` to go configure credentials, when the real
  reason was `grammar_pass: false`.
- Failed passes that degrade another section now say so next to the failure. A
  Perplexity rate-limit silently cost Section 9 all of its grounded URLs and
  dropped citation resolution from 48% to 22%, with nothing connecting the two.
- The delta warns when its baseline was itself an incomplete run.
- SEO meta-description constraint strengthened (the model was told the limit and
  overran it on consecutive runs: 157/155, 177/155). The value is still never
  truncated — that is a deliberate decision, machine truncation reads worse than
  the author trimming.

---

## 2. Hold — litellm supersedes these

Do not merge until the migration decision resolves. If litellm is adopted, most
of this is deleted rather than landed.

| work | why it is held |
|---|---|
| `fix/credit-exhaustion-detection` **entire branch** | litellm raises a proper exception on the in-band streaming error this branch exists to catch |
| `quota.py` | superseded except the terminal-vs-transient classifier, which goes upstream |
| `schema_format.py` + adapter wiring | `instructor`, or litellm's own `response_format` passthrough |
| `streaming.py` stream-error capture (4 accumulators) | litellm raises instead |
| `tokens.py` cached-token handling | litellm models cached tokens already |
| OpenAI `web_search` repair | keep the finding; re-apply only if we keep our own adapter |
| Gemini `grounding_chunks` capture | litellm exposes it as `vertex_ai_grounding_metadata` |

**`resolve_grounding_urls` survives regardless** — litellm hands back Gemini's
metadata, but the URIs are still `vertexaisearch...` redirect wrappers that
expire in ~30 days, and resolving them is our problem either way.

---

## 3. The litellm migration

Spiked against 1.96.2 on 2026-08-12. **Verdict: migrate.** Everything the
pipeline depends on survives: Perplexity `citations`/`search_results`, Gemini
grounding, truncation via `finish_reason`, cached tokens, structured output,
Perplexity search params, and — critically — in-band streaming errors surface as
exceptions rather than as empty content.

**One hard constraint, measured:** OpenAI must go through `litellm.responses()`,
not `litellm.completion()`.

| surface | time to first byte | total | max gap |
|---|---|---|---|
| `completion(stream=True)` | 79.1s | 79.1s | 79.1s (100% silence) |
| `responses(stream=True)` | 0.8s | 76.5s | 17.2s, 1318 summary deltas |

`completion()` routes reasoning models through Chat Completions and sends zero
bytes while thinking — the exact regression the Responses migration fixed. So
the "one call shape" benefit is partial: five providers through `completion()`,
OpenAI through `responses()`.

**Phases:**

1. Replace the six adapters + `streaming.py` + `tokens.py` + `cost.py` +
   `model_registry.py` (~3,900 lines) with litellm, OpenAI on `responses()`.
2. Port the timeouts calibration — it is measured knowledge and survives as
   config, but litellm has its own timeout/retry semantics to fit it to.
3. Re-verify the five preserved fields against the real pipeline, not a spike.
4. Then the small swaps: `instructor`, `rapidfuzz`, `language-tool-python`.

**Not recommended:** replacing `analysis/links.py` with lychee. It is a Rust
binary (subprocess, not import) and the tier semantics would need rebuilding
around its output. 239 lines is not where the pain is.

---

## 4. Upstream — see UPSTREAM.md

| # | item | status |
|---|---|---|
| 1 | litellm classifies credit exhaustion as `RateLimitError`, so retry logic retries a dead account. Offer `quota.py`'s classifier + tests as the PR. | ready |
| 4 | litellm `completion()` silently drops reasoning-summary streaming for OpenAI reasoning models. | ready |
| 2 | Link checkers have no TLS-impersonation tier. | verify — file against whichever checker we adopt |
| 3 | Wayback clients: 429 backoff and authenticated SPN2. | verify — check current releases first |

---

## 5. Decisions needed

1. **Adopt litellm?** Everything in section 2 depends on the answer.
2. **Prompt-cache layout** — built, defaulted off. **Still open, blocked.** It
   moves the domain instruction from before the article to after it (76% of
   input cached on calls 2+, ~$0.56–1.01/run).

   Two things settled on 2026-08-14 that whoever cuts this PR needs:

   - **The golden report cannot verify it.** `test_pipeline_end_to_end.py` stubs
     `_run_domain`, and that is exactly where `prompt_cache_layout` is applied,
     so flipping the flag produces an *empty* golden diff by construction. An
     empty diff there is evidence the code never ran, not evidence the findings
     held. Verifying it means a live run diffed against a prior live run of the
     same article — and reading that diff means checking `model_failures` first,
     because a run that lost a provider looks exactly like a behavioural change.
   - **Documentation is written and waiting.** A `### Prompt cache layout`
     section for `docs/CONFIGURATION.md` and the `user.example.yaml` comment
     block were drafted for PR #84 and pulled back out, because master has no
     such setting yet. Recover them from that PR's branch history and land them
     alongside the feature so it does not ship undocumented.

   Blocked on OpenAI credits regardless: the entire measured effect is on
   OpenAI's prefix, and the account has returned `credit_balance_exhausted`
   since 2026-08-11.
3. ~~**Delete the junk history directories?**~~ **Done, 2026-08-14.** `t/` and
   `title/` are gone. The Jun 8 report was refiled into the article's own
   directory as `run_1_20260608_075204_report.json`. It turned out to be the
   smallest part of a larger problem: the same article occupied three more
   directories because its title kept being revised, which let one article clear
   `voice_pattern_report`'s `MIN_ARTICLES = 3` on its own. All 47 reports are now
   under `pipeline_history/dc-environment/`, and a `History key:` handoff field
   (PR #84) stops title revisions forking a history again.
4. ~~**Enable OpenAI `web_search`?**~~ **Done, 2026-08-14 (PR #84).** Scoped to
   `fact_check` rather than enabled flat — it is a per-model flag, so at
   `maximum` it was billing a search on all five domains when only `fact_check`
   can use one. Note it also never survived `cost_preset`, which rebuilds the
   model dict and dropped it; fixed in the same PR. What it buys is a
   live-fetched `source` field instead of training recall — not annotations,
   which are structurally empty under JSON-only prompts.
5. ~~**Enable Claude?**~~ **Done, 2026-08-14 (PR #84).** Superseded by a general
   drafting-model exclusion: `pipeline.drafting_model` (or `Drafted with:` in a
   handoff) drops the declared drafter from `voice_style` only, so Claude can be
   enabled without judging its own phrasing habits.

---

## Known open items, not addressed here

- Only **9 of 144 claims** in the last run were verified against a document the
  pipeline actually read. The rest rest on a model asserting a source exists.
  The tier names say so; the framing invites more confidence than earned.
- Rerun nondeterminism: consensus flags vary between identical runs.
- Grok's output volume is far below the other providers on identical domains
  (602 tokens vs 9,377 and 22,414 on `voice_style`). An explicit `max_tokens`
  now removes the provider default as a suspect; the cause is still unknown.
