# Citations (Section 9)

After the fact-check pass, the pipeline takes the claims that came out of it and tries to trace each one to a primary source. The results land in **Section 9** of the report — both in `run_N_<timestamp>_report.json` and in the readable `run_N_<timestamp>_review.md`.

Section 9 is not a pass/fail list. A claim can be resolved at very different levels of confidence, and the difference matters: one tier means a model read the source and confirmed it backs the claim, another means "here is a portal that is probably about the right topic." Reading those as equivalent is exactly the mistake this section is structured to prevent.

---

## Confidence tiers

Every entry carries a `verification` field. Four outcomes reach the report.

### Verified — `verification: "checksum"`

The strongest tier. The source URL was fetched, its content SHA-256 checksummed and recorded, **and** a model read that content and confirmed it supports the specific claim.

This tier is only reachable two ways:

1. **A `known_url` citation** — the fact-check model supplied a source URL alongside the claim. The pipeline fetches it, checksums it, then makes a separate cheap model call (`mistral-small-latest`) asking whether the page content actually supports the claim. The verdict must come back `supports`. A verdict of `contradicts`, `not_addressed`, or `inconclusive` demotes the entry out of this tier entirely (see *Content mismatch* below).

   This check exists because a `known_url` frequently comes from a model recalling a URL from training data rather than from a live search. **A URL that loads is not evidence that the page says what the claim says.** The fetch proves the page exists; only the relevance check speaks to whether it's the right page.

2. **A data-fetching source adapter** — `census`, `crossref`, `eia`, or `fred`. These retrieve actual data from an API, so the checksummed content *is* the evidence.

**How much to trust it:** high. The source was retrieved and its relevance affirmatively checked. Still worth a glance — the relevance check is one cheap model call, not a human — but this tier is doing real verification work.

> **If the relevance check can't run,** the entry does not stay here — it moves to *Could not be verified* below. Reaching this tier always means a model read the extracted content and affirmed it. With no Mistral key configured, expect no verified `known_url` entries at all.

**What the model actually reads.** The fetched body is reduced to readable text before verification: main-article extraction for HTML (nav, header, footer, script and style blocks stripped) and `pypdf` text extraction for PDFs. The excerpt sent to the model is then centred on the passage containing the claim's distinctive terms and figures, rather than the first N characters of the document — in a long PDF the supporting sentence is rarely near the top.

### Pointer-only — `verification: "pointer"`

A topic-relevant source was identified. **Nothing was verified.**

Six adapters are pointer-only: `epa`, `ferc`, `fhwa`, `icc`, `ilga`, `pjm`. They match a claim against a keyword list for a regulatory or statistical topic and, on a hit, return the URL of the relevant portal or publication — a place a human could go look this up. They do not retrieve the figure, do not confirm it, and do not know whether the portal actually contains anything supporting the claim.

The report labels this tier "topic-relevant source identified, NOT independently verified — confirm manually before citing," and that label is literal.

Keyword matching is gated by `topic_match.py`, which discards a keyword hit when it appears in the same sentence as a credential phrase ("credentials in", "degree in", "expertise in", …). Without that gate, a sentence like *"He does not hold credentials in environmental engineering or air quality analysis"* genuinely contains "air quality" and would resolve to the EPA Air Quality System portal — a claim about a person's background pointed at an emissions database. The gate deliberately errs toward not resolving rather than resolving to the wrong topic, so expect some claims to land in *Unresolved* that a human would have matched.

**How much to trust it:** treat it as a research lead, not a citation. Open the URL and confirm before the claim ships.

### Unresolved — `resolved: false`

No configured adapter matched, or the source URL couldn't be fetched. The entry carries a `note` explaining which. Nothing was established.

**How much to trust it:** nothing to trust — this is a to-do.

### Content mismatch — `verification: "content_mismatch"`

A distinct failure mode that lands in the *Unresolved* section of the readable report. The source URL fetched and checksummed fine, but the relevance check came back saying the page does **not** support the claim. The entry records the verdict (`contradicts`, `not_addressed`, or `inconclusive`) and the model's one-sentence reason.

This is worth more attention than an ordinary unresolved entry. An ordinary one means "we couldn't find a source." This one means "a source was proposed and it doesn't check out" — and a `contradicts` verdict in particular is a signal about the claim, not just about the citation.

This tier asserts something about the source, so it is only ever reached when the document was genuinely read. If the content could not be extracted, or the check could not run, the entry becomes *Could not be verified* instead — never this.

### Could not be verified — `verification: "unverifiable"`

The source URL fetched and checksummed fine, but no judgement about it was possible. Either the content could not be read, or the relevance check could not run:

- the document is a PDF with no extractable text layer (a pure scan, or password-protected);
- the page is JavaScript-rendered, paywalled, or otherwise yielded no article text;
- the response was a bot-check, CAPTCHA, or paywall interstitial served as HTTP 200 rather than the document itself (`content_kind: "access_wall"`);
- no Mistral API key is configured, or the relevance call failed or returned an unparseable verdict.

`resolved` stays `true` — a real document was fetched, and it is still archived and shown to you for manual checking. But nothing was confirmed and, importantly, **nothing was refuted**.

**How much to trust it:** treat it exactly like pointer-only — a lead to check by hand. The one thing it never means is that the source failed to support the claim. That distinction is the point of the tier: an honest "we couldn't read this" is useful, while a wrong "this source doesn't back you up" is actively misleading.

---

## Content drift

The checksum recorded on a verified citation isn't only a fingerprint of what was fetched — it's comparable across runs. Before resolution starts, the pipeline builds a URL → prior-checksum index from every report under `pipeline_history/`: every run, every article, since the same primary sources (EPA eGRID, EIA state profiles) get cited across many articles for one publication. When a citation resolves and its URL is already in that index with a different checksum, the entry gains a `content_changed_since` field recording the prior checksum, and which run of which article last saw it.

It surfaces as its own block above the tiers in the readable report, and in the console summary:

```
############################################################
WARNING: 2 verified source(s) have changed content since a prior run checksummed them:
  https://www.eia.gov/electricity/state/illinois/
    Last matched in run 3 of 'grid-reliability-piece' on 2026-01-04T09:12:00
Claims previously verified against these sources may need re-checking.
############################################################
```

**What a mismatch proves:** that the bytes at that URL are not the bytes that were there the last time this pipeline fetched it. That's all. It does not say what changed, how much, or whether it matters.

The change may be a substantive revision — a figure restated, a methodology note added, a table replaced with updated annual data — which is exactly what you want to know about a source a previously-published claim rests on. Or it may be a rotating ad slot, a "last updated" timestamp, a session token in the markup, or a randomized nav element. A SHA-256 over full page content cannot tell those apart, and this feature does not try to. Treat a flag as *go look at the page*, not as *the source changed its story*.

**What it never does:** block. A mismatch does not fail resolution, does not demote the entry out of its tier, and does not change `resolved`. A citation flagged for drift is still a verified citation; the flag is a note for the human reviewer sitting alongside it.

**Scope: the verified tier only.** Drift is computed for `verification: "checksum"` entries and compared only against prior entries that were themselves at that tier. Pointer-only citations are excluded in both directions. Their checksum is taken over whatever the adapter returned as content, which for a portal pointer is usually nothing at all or a static blurb — so a mismatch there would be measuring the adapter, not the source. Excluding them from the index also matters: a URL first cited pointer-only and later fetched for real would otherwise flag as "changed" on the strength of the tiers differing. Entries from reports predating the `verification` field are skipped for the same reason — their tier can't be established, so no comparison from them is trustworthy.

Content-mismatch entries (`verification: "content_mismatch"`) are not drift-flagged either. They've already left the verified tier carrying a stronger signal, and stacking a second one on top adds noise, not information.

There's no drift check on a first run against an empty or missing `pipeline_history/`, and none for a URL never cited before — no prior checksum means nothing to compare.

**Scope: matching checksum bases only.** Two checksums are compared only when they were taken over the same thing, recorded as `checksum_basis`. A `known_url` checksum used to cover the raw HTTP response body and now covers the extracted article text (`checksum_basis: "extracted_text"`), so comparing across that change would report every previously-cited source as having changed when none of them moved. Reports written before the switch carry no `checksum_basis` and are therefore not compared against current ones. This suppression is self-healing: once a URL has been re-checksummed on the new basis, the run after that compares normally. Adapter-sourced citations carry no basis label at all — what their checksum covers never changed, so they stay comparable against every prior run.

---

## Wayback Machine behavior

Every resolved citation URL is checked against the Wayback Machine, and the pipeline actively works to get sources archived rather than only reporting on them.

**Availability check.** Each resolved URL is looked up via archive.org's availability API. The result records whether a snapshot exists, its timestamp, its age in days, and whether it's stale. "Stale" defaults to older than 180 days and is set by `pipeline.wayback_snapshot_stale_days` in `user.yaml`. If the cited URL is itself a `web.archive.org` link, its date is read straight from the URL and no archive-of-an-archive lookup happens.

**Save Page Now submission.** After resolution finishes, any resolved citation whose URL is *not* yet archived gets submitted to archive.org's Save Page Now API. This runs as a follow-up pass at a lower concurrency than resolution itself (2 vs. 8), because each submission triggers a real page capture on archive.org's side rather than a cheap read.

Submission is **fire-and-forget by design**. Captures run asynchronously and can take seconds to minutes, and the pipeline does not poll for completion — it won't block your run on someone else's crawler. The console reports it plainly:

```
  3 resolved URL(s) submitted for archiving (check back later — archive.org processes asynchronously)
```

A later run's availability check will pick up the snapshot once it exists. A submission failure degrades to "still shows as unarchived" and never fails the run.

**Unreadable-origin fallback.** When a direct fetch of a `known_url` fails in a way that means *we couldn't read the origin* rather than *the resource is gone*, the pipeline makes one attempt (never a retry loop) to read an archived snapshot of that same URL instead, and uses the snapshot's content for checksumming and relevance verification.

It fires on:

| Failure | Recorded `origin_failure` |
|---|---|
| 401 Unauthorized | `auth_required` |
| 403 Forbidden | `blocked` |
| 429 Too Many Requests | `rate_limited` |
| Connect/read timeout | `timeout` |
| DNS or connection failure | `unreachable` |

It deliberately does **not** fire on 404/410 — the resource is genuinely gone, and surfacing that is the point; an archive copy would mask a problem you need to fix by re-sourcing the claim. It also does not fire on 5xx, which is the origin's own failure rather than a refusal aimed at us: a transient 5xx will be fine by the time a reader clicks, and a persistent one means the source needs replacing. Neither is helped by quietly substituting an archived copy.

The entry records `verified_via: "wayback_fallback"`, the `origin_failure` reason above, and an `archive_provenance` note stating that the checksum and relevance verdict describe the archived copy, not the live page. Staleness is not suppressed: a 245-day-old snapshot that satisfied a timeout is still reported stale, and the provenance note says so.

The same rules govern draft link validation (`analysis/links.py`), so a link recovered from the archive after a timeout reads `OK (via archive: origin timed out)` rather than being flattened into a plain `OK` or a bare `BROKEN`.

### archive.org credentials (optional)

Submissions work without credentials via the unauthenticated capture endpoint, subject to tighter and less predictable rate limits. Supplying an S3-style key pair from <https://archive.org/account/s3.php> uses the authenticated SPN2 endpoint, which gets higher rate limits and returns a job id:

```yaml
api_keys:
  archive_org:
    access_key: your_access_key_here
    secret_key: your_secret_key_here
```

See [CONFIGURATION.md](CONFIGURATION.md#api-keys).

---

## Configuring sources

Adapters are selected per publication in `configs/<publication>.yaml`:

```yaml
citation_sources:
  - name: EIA
    adapter: eia
    description: Energy Information Administration data
  - name: FRED
    adapter: fred
    description: Federal Reserve Economic Data (BLS, Census series)
```

Adapters are tried in the order listed, and the first one that resolves wins. A claim carrying a `known_url` skips adapter matching entirely.

| Adapter | Kind | Reaches |
|---|---|---|
| `census` | data fetch | Verified |
| `crossref` | data fetch | Verified |
| `eia` | data fetch | Verified |
| `fred` | data fetch | Verified |
| `epa` | pointer | Pointer-only |
| `ferc` | pointer | Pointer-only |
| `fhwa` | pointer | Pointer-only |
| `icc` | pointer | Pointer-only |
| `ilga` | pointer | Pointer-only |
| `pjm` | pointer | Pointer-only |

Adding an adapter: see [the source adapter interface](../packages/ci-article-review/src/ci_article_review/adapters/citation/sources/README.md).

---

## Cost

The relevance check runs once per `known_url` citation, on a deliberately cheap model. When the pipeline passes its `api_call_log` through, each check is recorded as a `citation_verification:known_url` entry and flows into the run's cost total like any other model call.
