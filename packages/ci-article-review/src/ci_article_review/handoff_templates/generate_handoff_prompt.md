GENERATE DRAFT SUBMISSION HANDOFF
Use this prompt in any LLM chat (Claude, ChatGPT, Gemini, etc.) to produce a
properly-structured draft_submission.md for the article review pipeline.

Copy everything between the dashed lines and paste it as a single message.
Replace the bracketed fields. The model will output a complete handoff document
you can save as your-article-handoff.md and pass to `python pipeline.py --draft`.

TWO-FILE VARIANT: for long articles, chat UIs can truncate or mangle a single
giant paste that mixes analysis and the full article text. If that's a
problem, ask the model to stop before the DRAFT section and output only
everything from "DRAFT SUBMISSION HANDOFF" through "ADDITIONAL CONTEXT FOR
REVIEW MODELS" (see metadata_only.md for the exact shape). Save that as
your-article-metadata.md, save the raw article text on its own (whatever you
already have — no wrapping needed) as your-article-draft.md, and run:

    python pipeline.py --raw-draft your-article-draft.md --metadata your-article-metadata.md --publication your_publication_name

This produces the same result as a single --draft handoff — it's just split
across two files so the draft never has to survive a round trip through the
chat model.

──────────────────────────────────────────────────────────────────────────────

You are helping me prepare a draft article for a multi-model AI review pipeline.
I need you to produce a structured handoff document in the exact format the
pipeline parser expects. I will provide the article draft and background context
below. Produce the document with no additional commentary — only the formatted
handoff text.

ARTICLE TITLE: [paste the article's title here]

PUBLICATION: [mikehammett, or your publication config name]

PRIMARY CLAIM (one or two sentences — the core argument the piece is built on):
[paste or describe the thesis]

ABOUT THIS PUBLICATION:
[One short paragraph: what you cover, who reads it, your editorial standards.
Example: "mikehammett.net covers data center infrastructure, ISP operations, and
Northern Illinois community impact from a technically literate but non-trade
perspective. The primary reader is a technically credible practitioner or local
official who will verify primary sources."]

ARTICLE DRAFT:
[paste the full article text here]

─────────────────────────────────
OPTIONAL BACKGROUND (include what you have; leave out what you don't):

SOURCES ALREADY CITED (list them if you have a citations section):
[e.g., "28 citations. Primary sources include EPA eGRID 2023, LBNL 2024 Data
Center Energy Usage Report, Virginia JLARC RD206..."]

UNCERTAIN SECTIONS (claims you want the pipeline to scrutinize closely):
[e.g., "The water-per-acre comparison in section 3 uses my own calculations from
public records, not metered utility data."]

KNOWN GAPS (what you know is missing or underdeveloped):
[e.g., "No peer-reviewed comparison of data center vs warehouse impacts on
matched parcels exists — I note this explicitly in the piece."]

STEELMANNED POSITION (the strongest honest version of your argument):
[e.g., "The tier system distinguishes compliant facilities from misconduct cases
and local geography from national aggregates. Both distinctions are defensible
and consistent."]

STRAWMANNED POSITION (the weakest link in your argument — where you'd lose):
[e.g., "The residential water comparison figures are my own calculations, not
metered data. If that methodology is challenged, the comparison weakens."]

STEELMANNED OPPOSITION (the best case against your thesis):
[e.g., "The opposition's strongest argument is cumulative regulatory failure, not
individual facility impacts. The frameworks were not designed for this scale."]

STRAWMANNED OPPOSITION (the weakest case against your thesis):
[e.g., "The PEC/Cork air quality analysis modeled permit ceilings as operational
norms. Virginia DEQ formally rebutted all three methodological failures."]

COUNTERARGUMENTS ADDRESSED IN THE PIECE:
[list 2–4 specific objections you engaged with directly]

COUNTERARGUMENTS DISMISSED (and why):
[list 1–3 objections you explicitly dismissed as unsupported]

KNOWN READER OBJECTIONS:
[list 2–4 reactions you expect in comments or public meetings when this is cited]

INTENDED USE OF THIS ARTICLE:
[e.g., "Cited in planning commission hearings in Illinois where claims from
Virginia and Arizona are presented as directly applicable to DeKalb County."]

RELATED PRIOR ARTICLES (helps the pipeline assess completeness and consistency):
[list any prior articles on related topics from your publication]

─────────────────────────────────
REQUIRED OUTPUT FORMAT:

Produce a document with exactly these section headers, in this order, with each
header on its own line followed by a blank line before the content:

DRAFT SUBMISSION HANDOFF
Generated: [today's date]
Pipeline run: 1
Article: [title]
Publication: [publication name]

PRIMARY CLAIM
[your primary claim here]

TARGET AUDIENCE
[derive from the publication description and article content: who is the primary
reader, what do they know, what makes them stop reading; then secondary audience
who will check sources]

PRE-DRAFT ANALYSIS SUMMARY
[Steelmanned position: ...

Strawmanned position: ...

Steelmanned opposition: ...

Strawmanned opposition: ...

Counterarguments addressed in this piece:
- ...

Counterarguments dismissed:
- ...

If no steelman/strawman context was provided, derive the most defensible and
most vulnerable points from the article content itself.]

SOURCES ALREADY CITED
[paste or summarize; if none provided, write "None provided."]

UNCERTAIN SECTIONS
[paste or summarize; if none provided, write "None identified by author."]

KNOWN GAPS
[paste or summarize; if none provided, write "None identified by author."]

ADDITIONAL CONTEXT FOR REVIEW MODELS
[Prior articles: ...

Known reader objections: ...

Intended use: ...]

DRAFT
[paste the complete article text here, preserving all markdown formatting,
headings, tables, and citation markers exactly as provided]

──────────────────────────────────────────────────────────────────────────────

AFTER GENERATING: Save the output as your-article-name-handoff.md in
handoff_templates/ and run:

    python pipeline.py --draft handoff_templates/your-article-name-handoff.md --publication your_publication_name

The pipeline will tell you if any required sections are missing before running.

If the article is long enough that you'd rather not have it re-typed or
re-pasted by the chat model at all, use the two-file variant described above
instead: ask for everything up to (not including) the DRAFT section, save
that alone as metadata.md (see metadata_only.md for the exact format), and
pass your own already-in-hand draft file separately with --raw-draft.
