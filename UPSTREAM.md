# Upstream contribution queue

Things this project needs that a maintained library nearly does. The rule
(see also the reuse-first note in CONTRIBUTING): when a library is 90% right,
**file the issue or PR upstream** rather than carrying a local fork of the
behaviour. A local workaround is a maintenance liability; an upstream fix is
one everybody gets.

Each entry carries the evidence needed to open it — a reproduction, a
measurement, and the proposed change. Do not add an entry without one.

**Whether we use the package is not the test.** If we find a real bug in
somebody's library, we file it, adopted or not — the finding is worth the same
to their users either way, and sitting on it because it no longer affects us is
just hoarding. Two corollaries, both learned the hard way on entries 2 and 3:

- *"We don't depend on it" is not a verification.* Deciding not to file is a
  claim about **their** code, so it has to be grounded in reading their code.
  Entry 3 was dropped on non-adoption and, when the source was actually read,
  turned out to contain a genuine bug in one library — and a claim that was
  simply wrong about the other.
- *A bug and a missing feature are filed differently.* A defect gets an issue.
  A capability the library never claimed gets our measurement added to whatever
  thread already exists, not a new issue implying they broke something.

Status values: `ready` (evidence complete, needs filing) · `verify`
(candidate, claim not yet checked **against their source**, not merely against
whether we still use it) · `filed` (link the issue/PR) · `contributed` (added to
an existing upstream thread — link the comment) · `landed` · `split` (parts
resolved differently — say which) · `dropped` (with the reason).

---

## 1. litellm — credit exhaustion is classified as a rate limit

**Status:** contributed — added to an existing open issue rather than filed anew.
[BerriAI/litellm#32785](https://github.com/BerriAI/litellm/issues/32785) already
reported this on 2026-07-10 (and traced it further than we had, to the OpenAI
branch of `exception_type()` testing `is_error_str_rate_limit()` before ever
consulting the body's `insufficient_quota` code). Our evidence went there as
[a comment](https://github.com/BerriAI/litellm/issues/32785#issuecomment-5299586013)
on 2026-08-14: the cross-provider table below (which #32785 offered to enumerate
and did not), confirmation that it still reproduces on v1.96.2 (they tested
1.91.1 and HEAD at 2026-07-10), and the in-band SSE case — which their proposed
fix would miss, since patching `exception_type()` alone leaves the HTTP-200
streaming path silent. `ci_core/llm/quota.py` is offered there as the PR,
pending a maintainer steer on which of their three shapes they want.
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

**Status:** contributed — measurement added to
[lycheeverse/lychee#1439](https://github.com/lycheeverse/lychee/issues/1439#issuecomment-5302215817)
on 2026-08-15. Not filed as a defect: lychee treats a 403 correctly, so there is
no bug here, only a capability it lacks — and #1439 has had that request open
for two years. Our numbers argue for a lighter answer than the FlareSolverr
sidecar being discussed. The local tier stays ours (see the note at the end).
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

**Verified 2026-08-14, contributed 2026-08-15.** We did not adopt lychee (a Rust
binary whose output our tier semantics would need rebuilding around) and built
the tier ourselves: `ci_core/http.py` `impersonating_get` (`curl_cffi`,
`impersonate="chrome"`, behind the optional `unblock` extra), wired into
`analysis/links.py` as the final tier after an honest request is refused —
opt-in, as the entry wanted.

Not adopting it is *not* a reason to withhold the finding, so the measurement
went upstream anyway. What it is not, though, is a bug: lychee reporting a 403
as a 403 is correct behaviour. So it went as data on lychee's existing open
request for Cloudflare bypass (#1439, open two years, currently discussing a
FlareSolverr sidecar) rather than as a new issue — the useful contribution being
that a TLS fingerprint gets a real slice of the same benefit with no sidecar.

---

## 3. Wayback clients — rate-limit backoff and authenticated SPN2

**Status:** split. Our own throttling failure was our bug and is fixed (see the
verification note at the end). Separately, reading the candidate libraries'
source turned up a real defect in waybackpy, filed as
[akamhy/waybackpy#200](https://github.com/akamhy/waybackpy/issues/200) on
2026-08-15. savepagenow: no defect, and half of this entry's original claim was
wrong about it.
**Candidate repos:** akamhy/waybackpy, palewire/savepagenow (not pastpages —
the project moved)

`https://archive.org/wayback/available` throttles hard at IP level with a long
window. Measured 2026-08-12: 12 consecutive requests all 429, the *first* one
included, and still 429 after a 45s cooldown at 1 request / 6s.

Impact on this project, from its own run reports:

| run | resolved citations | archived | 429 |
|---|---|---|---|
| 2026-08-12 | 32 | 7 | 25 |
| 2026-08-11 | 49 | 0 | ~49 |

**Verified 2026-08-14/15. The run-report damage was ours and is fixed — but
reading the libraries anyway found a real bug in one of them.**

*Our half, fixed — on the second attempt:* the measured impact (49 resolved, 0
archived, ~49 429s) came from our own pipeline having no pacing and no backoff
at all. The guard added in response paced calls, honoured `Retry-After`, and
tripped a breaker once archive.org had refused repeatedly — but it was written
as if lookups were sequential, and `check()` runs on `_MAX_PARALLEL` resolver
threads. Under that, the backoff slept in the failing thread while the others
kept the pace up, and a success zeroed a counter shared with every other thread,
so a run throttled four-in-five never tripped the breaker at all. Fixed properly
in PR #99, where the 429 moves a clock every thread waits on and the count is a
per-run budget. Recording the sequence because the first fix looked right and
tested green: concurrency was the part nobody checked.
`submit()` already supported authenticated SPN2 via an archive.org S3-style key
pair (`Authorization: LOW <access>:<secret>`).

*savepagenow — no defect, and this entry was wrong about it.* `capture()` takes
`authenticate=True` and reads `SAVEPAGENOW_ACCESS_KEY` / `SAVEPAGENOW_SECRET_KEY`
to send the same `LOW` header, so "no authenticated SPN2" was simply incorrect.
It has no 429 backoff, but it raises a typed `TooManyRequests` carrying the
response headers, so `Retry-After` reaches the caller intact — raising and
letting the caller decide is a design choice, not a defect.

*waybackpy — real bug, filed as
[#200](https://github.com/akamhy/waybackpy/issues/200) and fixed in
[PR #201](https://github.com/akamhy/waybackpy/pull/201).*
`availability_api.py`'s `setup_json()` calls `.json()` with no status check.
archive.org answers a throttled availability call with an HTML error page, so
`.json()` raises `JSONDecodeError`, which surfaces as
`InvalidJSONInAvailabilityAPIResponse` — a rate limit reported as malformed
data, with the server's `Retry-After` discarded. Their **Save** API already
handles this correctly (`Retry` adapter plus `TooManyRequestsError`, added for
their own #131 in 2022); the fix never crossed to the Availability API, and
`TooManyRequestsError` is already in their tree, so it is a two-line status
check. Reproduced against 3.0.6 with a stubbed 429 — the live endpoint was not
throttling on 2026-08-15, which is itself worth noting: archive.org's throttling
is episodic, so the 2026-08-12 measurement is a throttled window, not a constant.

Note the same shape as entry 1: a throttle misreported as a data-format error,
sending the reader hunting for a parser bug that isn't there.

Caveat on expectations: waybackpy looks abandoned — last release 3.0.6 on
2022-03-15, last commit 2022-11-17, last repo push 2024-02-26, with open issues
sitting unanswered for years. Filed anyway; a searchable report of the error
string has value to the next person even if no maintainer answers.

---

## 4. litellm — completion() silently drops reasoning-summary streaming

**Status:** filed — [BerriAI/litellm#36992](https://github.com/BerriAI/litellm/issues/36992),
2026-08-14.
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

**Prior art checked before filing** (both cited in the issue, so it cannot be
waved off as already-solved): #13780 asked for reasoning summaries through
`/chat/completions` and was closed *completed* on 2025-11-19, but the resolution
was `extra_body` param pass-through — that carries params *to* the Responses
API, it does not make the default `completion(stream=True)` path stream
summaries. PR #14117, which would have routed reasoning models to the Responses
API and is the actual fix, was **closed unmerged on 2026-02-24**. The routing
option has therefore been attempted and dropped once already, which is why the
issue leads with the cheaper ask: signal the drop, don't hide it.

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
