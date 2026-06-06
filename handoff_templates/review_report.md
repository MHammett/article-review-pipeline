REVIEW REPORT HANDOFF
Generated: [timestamp]
Pipeline run: [number]
Article: [title]
Publication: [publication config name]

LANGUAGETOOL CORRECTIONS APPLIED
Total corrections: [count]
[For each correction: original text → corrected text, rule ID]

DELTA FROM PRIOR RUN
[If run 1: n/a]
[If re-run: word count change %, claims modified, sections restructured,
 prior consensus flags resolved, new flags introduced, recommendation: re-run / stable]

API CALL LOG
[Model, alias resolved to, response status, tokens in/out, for each call]
[Any failures or degraded responses]

CORRECTED DRAFT
[Full article text after LanguageTool pass]

---

CONSOLIDATED REVIEW REPORT

SECTION 1: CONSENSUS FLAGS (3+ models, or 2+ models + LanguageTool)
[Each item: passage quoted from draft, models that flagged it, nature of problem, suggested resolution]

SECTION 2: FACTUAL VERIFICATION (Gemini, search-grounded)
Confirmed: [claim | source | confidence]
Outdated: [claim | current value | source | confidence]
Contradicted: [claim | contradiction | source | confidence]
Unverifiable: [claim | what was checked | why unverifiable]
Primary source resolution required: [claim | best candidate source]

SECTION 3: VOICE AND AI-SPEAK (OpenAI)
[Each item: passage | problem | suggested rewrite]

SECTION 4: ARGUMENT INTEGRITY (Mistral)
[Each item: passage | logical problem | steelman considered | why it survived]

SECTION 5: COMPLETENESS AND FRAMING (OpenAI second pass)
[Each item: what is missing or misframed | specific passage reference | audience affected]

SECTION 6: RED TEAM FINDINGS (Mistral second pass)
Most vulnerable claim: [passage | attack vector | supporting evidence for the attack]
Highest audience risk: [passage | risk | which audience segment]
Highest credibility risk: [passage | risk | what a critic would say]

SECTION 7: LOW-CONFIDENCE FLAGS (did not survive model's own steelman)
[Listed for awareness only -- dismiss unless something catches your attention]

SECTION 8: ADDITIONAL FINDINGS
[Anything identified outside standard pass scope]
