# Terminology: "voice" vs. "style"

This project uses **"voice"** and **"style"** as related but *distinct* terms. They
are not interchangeable, and a blanket find-and-replace between them will break
behavior. This document defines them and records where each is used on purpose, so
the distinction survives future edits.

See also [NAMING.md](NAMING.md) for package/dist naming.

## Definitions

- **Voice** — a distinct authorial *persona*: a recognizable mode of writing a person
  uses. One author can have several (an analytical voice in blog posts, a casual voice
  in tweets, a formal voice in email). Detecting voices means *separating the personas
  mixed together in a corpus*.
- **Style** — the mechanical / structural *fingerprint* of writing: sentence length,
  passive-voice ratio, hedging, vocabulary richness, banned words/phrases. Style is
  measured, not "discovered as a persona."
- **Style profile** — the package's published *output*: a structured description of how
  an author writes (prose + rules), written into `publication.yaml` for the review
  pipeline. It is the product of `ci-style-profile`; "style profile" is the public,
  product-level name even though internally it is assembled from one or more detected
  *voices*.

**Rule of thumb:** if the task is *finding separate personas*, it's **voice**. If it's
*describing or enforcing the writing fingerprint / the output artifact*, it's **style**.

## Three look-alike tokens (do not conflate)

All three contain the substring "voice", but they are different things in different
packages. Renaming one does not imply renaming the others.

| Token | What it is | Package | Status |
|---|---|---|---|
| `style_profile` (was `voice_profile`) | Data field in `publication.yaml` holding the author's style description | contract: written by `ci-style-profile`, read by `ci-article-review` | **renamed** → `style_profile`; legacy `voice_profile` still read as a fallback |
| `StyleCluster`, `detect_styles`, `max_styles`, … (was `Voice*`) | The bootstrap tool's own identifiers/keys | `ci-style-profile` | **renamed** to `style*` |
| `voice_style` | Name of one of the five review **domains** (`fact_check`, `voice_style`, `completeness`, `argument_integrity`, `red_team`) | `ci-article-review` | **left untouched** — not part of the style-profile package; renaming it is out of scope and would break the review engine |

## Where "voice" is kept on purpose

These are intentional. Do **not** "normalize" them to "style":

1. **Detection prompts** — `packages/ci-style-profile/src/ci_style_profile/prompts/detect_styles.txt`
   and `consolidate_detection.txt` keep "voice" in their instructions. Their job is to
   *discover and reconcile distinct authorial personas* in a mixed corpus. Telling the
   model to "identify distinct **styles**" would nudge it to cluster by mechanical
   features instead of by persona — a different, likely worse, behavior. Only the JSON
   schema key the code parses was renamed (`detected_voices` → `detected_styles`); the
   human-facing instructions still say "voices."
2. **The `voice_style` review domain** in `ci-article-review` — see the table above. It
   appears in `consolidation.py` (model weights), `pipeline.py`, configs, prompts
   (`ai_speak.txt`'s `VOICE PROFILE:` heading, `section_3_voice`), and tests. It is the
   review system's own vocabulary, unrelated to the style-profile package.

## Where "voice" was changed to "style"

- **Synthesis prompts** — `synthesize_canonical.txt`, `synthesize_per_source.txt`,
  `synthesize_per_style.txt`, `synthesize_reconcile.txt`. By the synthesis stage the
  personas are already detected; these prompts describe the **output profile**, which is
  the "style profile." "voice cluster" → "style cluster" there because the cluster is an
  input the synthesis step is *describing*, not *discovering*.
- **The emitted output key** `voice_profile` → `style_profile` (with the backward-compatible
  fallback noted above).
- **Identifiers, CLI flags, config keys** (`--voice` → `--style`, `voice_mode` →
  `style_mode`, etc.), and the **user-facing docs** (README, PLAN.md), where "voice" was
  the product/output concept rather than the persona-detection task.

## History

The package was renamed `ci-web-intel` → `ci-style-profile` and its terminology aligned
"voice" → "style" so the dist name matches the implementation. See [NAMING.md](NAMING.md)
for the resolved naming decision.
