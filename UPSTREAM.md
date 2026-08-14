# Upstream contribution queue

Things this project needs that a maintained library nearly does. The rule
(see also the reuse-first note in CONTRIBUTING): when a library is 90% right,
**file the issue or PR upstream** rather than carrying a local fork of the
behaviour. A local workaround is a maintenance liability; an upstream fix is
one everybody gets.

Each entry carries the evidence needed to open it — a reproduction, a
measurement, and the proposed change. Do not add an entry without one.

Status values: `ready` (evidence complete, needs filing) · `verify`
(candidate, claim not yet checked against the current release) · `filed`
(link the issue/PR) · `landed` · `dropped` (with the reason).

---

## 1. litellm — credit exhaustion is classified as a rate limit

**Status:** ready
**Repo:** BerriAI/litellm (tested against 1.96.2)

An account with no credits raises `litellm.exceptions.RateLimitError` with
`status_code=429`. The provider message survives intact, but the *type* says
"transient, back off and retry" when the true condition is "terminal, a human
must add credits". `RateLimitError` is the class retry logic keys on, so a
credit-dead account gets retried on every call in a batch.

**Reproduction** (a real credit-exhausted OpenAI key, 2026-08-12):

```python
litellm.completion(model="gpt-5.5", messages=[...], api_key=DEAD_KEY)
# litellm.exceptions.RateLimitError, status_code=429
# "OpenAIException - You have no credits remaining. Add credits to continue
#  using the API at https://platform.openai.com/settings/organization/billing/"
```

The same key streaming (`stream=True`) also raises `RateLimitError` — good, in
that it raises at all, but the same misclassification.

**Proposed change:** a distinct exception (or a flag on the existing one) for
terminal billing states, so retry/fallback logic can skip them. The providers
signal it unambiguously and differently from each other, which is exactly the
kind of normalisation litellm exists to do:

| provider | signal |
|---|---|
| OpenAI | `insufficient_quota` / `credit_balance_exhausted`, HTTP 429 |
| Anthropic | `credit_balance_too_low`, HTTP 400 |
| xAI | "insufficient credits", HTTP 403 |
| several | HTTP 402 Payment Required |

**We have a working classifier** in `ci_core/llm/quota.py` (phrase-matching led,
provider codes second, deliberately returning `None` rather than guessing) plus
its test suite, including the verbatim SSE event from the production failure.
Offer it as the PR.

---

## 2. Link checkers — no TLS-impersonation tier for Cloudflare-blocked links

**Status:** verify
**Candidate repo:** lycheeverse/lychee (or whichever checker we adopt)

Citation checking hits sites that 403 every honest request. Measured against six
real blocked citations on 2026-08-12:

- browser **headers** alone: **0/6** — the hard blocks are Cloudflare, which
  returns 403 to a full browser header set *on the domain root*. It fingerprints
  the TLS handshake; no header string changes that.
- browser **TLS fingerprint** (`curl_cffi`, `impersonate="chrome"`): **2/5**
  recovered with real content — congress.gov (488 KB) and a CDC PDF (661 KB).

The remaining three (ASME, Wiley/AGU, Royal Society) return a ~6 KB challenge
page to everything and are very likely subscription gates, not bot gates.

**Proposed change:** an opt-in final tier that retries a blocked URL with a
browser TLS fingerprint before reporting it dead. Opt-in matters — it should be
a deliberate choice, not a default.

**Verify first:** we are not currently adopting lychee (it is a Rust binary, and
our tier semantics would need rebuilding around its output). File this against
whichever checker we actually use, or as a standalone note if we keep our own.

---

## 3. Wayback clients — rate-limit backoff and authenticated SPN2

**Status:** verify
**Candidate repos:** akamhy/waybackpy, pastpages/savepagenow

`https://archive.org/wayback/available` throttles hard at IP level with a long
window. Measured 2026-08-12: 12 consecutive requests all 429, the *first* one
included, and still 429 after a 45s cooldown at 1 request / 6s.

Impact on this project, from its own run reports:

| run | resolved citations | archived | 429 |
|---|---|---|---|
| 2026-08-12 | 32 | 7 | 25 |
| 2026-08-11 | 49 | 0 | ~49 |

**Verify first:** check whether the current releases already implement backoff
and authenticated SPN2 before filing — if they do, this is our bug, not theirs,
and the entry becomes `dropped`.

---

## 4. litellm — completion() silently drops reasoning-summary streaming

**Status:** ready
**Repo:** BerriAI/litellm (tested against 1.96.2)

`litellm.completion()` routes OpenAI reasoning models through Chat Completions,
which sends **nothing** while the model thinks. `litellm.responses()` uses the
Responses API and streams `response.reasoning_summary_text.delta` throughout.
Same model, same effort, same prompt, measured 2026-08-12:

| surface | time to first byte | total | max inter-chunk gap |
|---|---|---|---|
| `completion(stream=True)` | **79.1s** | 79.1s | **79.1s** |
| `responses(stream=True)` | **0.8s** | 76.5s | 17.2s |

The first row is 100% silence before any byte arrives. Anyone setting a socket
read timeout from observed inter-token gaps will size it correctly for every
provider except OpenAI reasoning models, then see them time out — with a hang
and a long think indistinguishable on the wire.

**Proposed change:** either route reasoning models to the Responses API from
`completion()` when a summary is available, or document the limitation
prominently. The silent part is the problem: nothing in the response indicates
that summary streaming was dropped.

---

## Checked and found to be non-issues

Recording these so nobody re-investigates them.

- **litellm passes Perplexity search parameters through.** `search_mode`,
  `search_recency_filter`, and `search_domain_filter` all reach the API.
  Verified 2026-08-12: `search_mode="academic"` took scholarly citations from
  4/18 to 9/22, and a domain filter constrained every citation to the requested
  domain.
- **litellm preserves the provider-specific fields we depend on.** Perplexity
  `citations` / `search_results` as top-level extras; Gemini grounding as
  `vertex_ai_grounding_metadata`; truncation as `finish_reason="length"`; cached
  tokens as `prompt_tokens_details.cached_tokens` and Gemini
  `cache_read_input_tokens`.
- **litellm raises on an in-band streaming error.** The HTTP-200-then-error case
  that this pipeline used to report as "Malformed JSON response" surfaces as a
  proper exception.
