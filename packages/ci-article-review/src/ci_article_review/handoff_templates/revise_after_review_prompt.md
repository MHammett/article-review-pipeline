REVISE DRAFT AND METADATA AFTER A REVIEW PASS
Use this prompt after a pipeline.py review run, to have the chat model revise
the draft AND regenerate the metadata sections consistently in the same pass —
so the two never drift apart across a Chat -> script -> Chat -> script loop.

Why this exists: PRIMARY CLAIM, PRE-DRAFT ANALYSIS SUMMARY, SOURCES ALREADY
CITED, UNCERTAIN SECTIONS, and KNOWN GAPS all reach the review models directly
(see pipeline.py's _build_user_prompt). If you hand-patch just the DRAFT
section after a revision, those sections silently go stale — a KNOWN GAPS
entry the revision already closed keeps getting re-flagged, or an UNCERTAIN
SECTIONS entry the reviewers already resolved keeps drawing extra scrutiny.
This prompt has the model that made the revision also reconcile the metadata,
in the same exact header format every round, so the file stays parseable and
accurate without you doing it by hand.

Copy everything between the dashed lines into the SAME chat thread that has
the article's context (or paste the CURRENT metadata file if starting fresh),
then open the pipeline run's `run_N_<timestamp>_review.md` file (saved next
to the JSON report in `pipeline_history/<article-slug>/`) and paste its
SECTION 1 through SECTION 8 content below it.

──────────────────────────────────────────────────────────────────────────────

You previously helped me with a draft article. I ran it through a multi-model
review pipeline and I'm pasting the consolidated findings below. Revise the
draft to address the findings you agree are valid, then output BOTH the
revised draft and an updated metadata block — in the exact format below, with
no additional commentary outside the two files.

Rules for the metadata update:
- PRIMARY CLAIM: leave unchanged UNLESS a finding directly challenges the core
  thesis itself (not a supporting detail). If you do change it, say so in one
  sentence before the output blocks — this is a rare, high-impact edit.
- PRE-DRAFT ANALYSIS SUMMARY: update only the specific steelman/strawman/
  counterargument bullets that the findings affected. Leave the rest as-is.
- SOURCES ALREADY CITED: append any new citations you added while addressing
  the findings. Don't touch existing entries you didn't change.
- UNCERTAIN SECTIONS: remove any entry the findings resolved (cite which
  section finding resolved it). Add any new ones you introduced by revising
  the draft, or that the reviewers surfaced but you're choosing not to fully
  resolve this round.
- KNOWN GAPS: same rule as UNCERTAIN SECTIONS — remove what you closed, add
  what you didn't.
- TARGET AUDIENCE and ADDITIONAL CONTEXT FOR REVIEW MODELS: leave unchanged
  unless a finding specifically implicates audience fit or context accuracy.
- Increment "Pipeline run" by 1.

Here are the consolidated review findings:
[paste SECTION 1 through SECTION 8 from the run's `run_N_<timestamp>_review.md`
file here]

Here is the current metadata (paste your metadata_only.md content, or the
full handoff document's non-DRAFT sections):
[paste current metadata here]

Here is the current draft:
[paste current draft here — or omit this and the DRAFT block below if you're
using the --raw-draft / --metadata two-file workflow and will keep the
revised draft in its own file separately]

──────────────────────────────────────────────────────────────────────────────
REQUIRED OUTPUT FORMAT — two blocks, in this order:

1) A one-paragraph summary of what changed and why, OUTSIDE any file content
   (for your own sanity-check — not parsed by the pipeline).

2) The metadata file, exactly matching metadata_only.md's header order:

DRAFT SUBMISSION HANDOFF
Generated: [today's date]
Pipeline run: [incremented number]
Article: [title]
Publication: [publication name]

PRIMARY CLAIM
[unchanged or updated]

TARGET AUDIENCE
[unchanged unless implicated]

PRE-DRAFT ANALYSIS SUMMARY
[updated bullets only where findings applied]

SOURCES ALREADY CITED
[existing + any newly added]

UNCERTAIN SECTIONS
[resolved entries removed, new ones added]

KNOWN GAPS
[resolved entries removed, new ones added]

ADDITIONAL CONTEXT FOR REVIEW MODELS
[unchanged unless implicated]

3) If you pasted the draft in step above (not using the two-file workflow),
   also include:

DRAFT
[the fully revised article text]

──────────────────────────────────────────────────────────────────────────────

AFTER GENERATING: save the metadata block as your-article-metadata.md
(overwriting the prior round's version) and, if using the two-file workflow,
save the revised draft as your-article-draft.md. Run:

    uv run ci-review --raw-draft your-article-draft.md --metadata your-article-metadata.md --publication your_publication_name

Or, if you used the single-file format (draft included in this same
response), save the whole thing as your-article-handoff.md and run:

    uv run ci-review --draft handoff_templates/your-article-handoff.md --publication your_publication_name
