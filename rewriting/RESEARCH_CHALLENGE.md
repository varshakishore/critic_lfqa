---
project: critique-guided-rewriting-pipeline
date: 2026-07-17
git_path: /Users/varshak/Documents/critic_lfqa
---

# Critique-Guided Rewriting Pipeline

## Summary

This project set out to improve and evaluate a pipeline that rewrites long-form
scientific QA answers — and their DR Tulu reasoning traces — using structured
critiques, in order to produce clean DPO preference pairs whose chosen/rejected
versions diverge at exactly one critique-flagged span. The core hypothesis of the
broader effort is that localized, critique-guided edits give a cleaner training
signal than global answer rewrites: a pair that differs at a single specific
location teaches the model exactly what to fix and where.

Using the `asta-flows` skill, the work was modeled as a beads epic
(scope → definitions → empirical review → fixes → validation). We ran an
evidence-driven audit of an initial pipeline run (`test_samples/`, 3 records,
later 2) against four key questions: whether trace rewriting inserts new searches
in valid DR Tulu format, whether placeholder searches are filled, whether the
rewrite stays within its `<can_edit>` bounds, and whether rewrites actually
address the critiques.

The audit surfaced five distinct defects: (1) malformed `<think>` search
insertions (call_tool/tool_output nested in an unclosed think); (2) malformed
`<can_edit>` tags produced when two critiques shared a span and the tagger
double-spliced; (3) silent `edit_span` anchor-match failures because answer
critiques were anchored against the `<answer>`-wrapped trace text but tagged
against the wrapper-less answer field; (4) genuine out-of-bounds drift (the model
editing identical text outside its tagged spans); and (5) reproduction corruption
of retrieved evidence (`tricuspid aortic stenosis` → `tricuspid a stenosis`
inside a tool_output the model was supposed to copy verbatim).

All five were fixed: a well-formedness prompt rewrite plus a structural validator;
merging overlapping/duplicate spans before tagging; tagging the `<answer>`-wrapped
block extracted from the trace; a diff-based content splice that keeps model text
only inside editable spans and reverts everything else to the original; and
snapping search-insertion spans to their enclosing complete `<think>` block so a
new search round lands at a block boundary. The final validated run (2 records)
produced zero structure problems, zero out-of-bounds drift on both trace and
answer, all searches preserved (rec0 +6, rec1 +7), and evidence corruption
reverted. Status: fixes validated on 2 records; the remaining open task is a
50-record prevalence scan to confirm the fixes hold at scale.

## Asta skills used

| Skill | Role on this project | Useful? |
|---|---|---|
| `asta-flows` | Modeled the session as a beads epic; ran init (installed `bd`, wired Dolt to the git remote), plan (bootstrapped scope/definitions/literature_review), recorded findings and fixes as tracked tasks, and regenerated `summary.md`. | yes |
| `research-challenge` | This report + submission. | yes (in use) |

No literature/search skills (`find-literature`, `semantic-scholar`, etc.) were
used — this was an engineering/debugging project. The pipeline itself calls the
Semantic Scholar snippet API, but that is application code, not a skill invocation.

## Self-critique

### What went well
- **Evidence over assertion.** Every claimed defect was backed by a reproducible
  check (tag-balance walk, drift measurement, anchor-match scan, corruption grep),
  and each fix was verified on real data before moving on.
- **Root-causing, not blaming the model.** The "model strips `<can_edit>` tags"
  symptom was traced to an actual tagger bug (double-splicing identical spans into
  nested, word-splitting tags) rather than written off as model flakiness.
- **Tight fix → re-run → validate loop**, with offline verification of each change
  against the existing outputs before spending a paid re-run.
- **Recorded the trail** in beads as it happened, so `summary.md` and the issue
  graph stayed an accurate ledger of findings and fixes.

### Where the agent fell short
- **Over-claimed "drift is tiny" early.** I conflated *total* edit size with
  *out-of-bounds* drift and had to walk it back once the answer-vs-trace and
  in-bounds-vs-out-of-bounds distinctions were made explicit. I should have
  separated those axes from the first measurement.
- **Proposed a redesign that didn't survive contact.** I recommended per-span
  structured edits; the user correctly pointed out that overlapping critiques
  produce conflicting independent rewrites. I should have caught that failure mode.
- **First splice was brittle.** The content-anchored splice aborted on boundary
  drift (rec0). It took the user's push ("why can't we splice the trace?") to move
  to a diff-based splice — which I could have reasoned to directly.
- **A wrong blanket rule in the prompt.** I first told the model to "close `<think>`
  first" unconditionally; the correct behavior is context-dependent (between-blocks
  vs mid-think). The user caught it.
- **Span-snapping needed a second re-run.** My first snap kept the span end at the
  anchor, leaving an unbalanced `<think>` that caused a double-`</think>`. Reasoning
  through the "editable span must be a balanced block" requirement up front would
  have saved a paid run.

### Friction
- Several offline drift measurements gave misleading numbers (a "98%/192% drift"
  artifact) because large search insertions confounded segment alignment. It took a
  few iterations to land on an insertion-tolerant measurement.
- I could not run the paid `gpt-5.4` pipeline myself, so every validation depended
  on a user re-run — inherent, but it made boundary bugs (which only surface in
  model output) cost a full round-trip each.

## Suggested skill improvements

### asta-flows
- **Observation:** The bootstrap frontier is science-shaped (scope → definitions →
  literature_review → hypothesis → experiment → analysis → synthesis). This project
  was an engineering audit, so I had to re-purpose `literature_review` as "empirical
  review of pipeline outputs" and model each bug/fix as a bare `task`, hand-wiring
  `parent-child`/`blocks`/`discovered-from` edges myself.
  - **Suggested change:** add an engineering/debugging track (or task types like
    `audit` → `defect` → `fix` → `validate`) so the graph and its schemas fit
    code-improvement projects without re-purposing science task types.
- **Observation:** `plan`'s replan table auto-resolves `literature_review` gaps into
  `hypothesis` tasks. My "gaps" were defects and remediation steps, which don't fit
  the hypothesis schema, so I created and wired tasks by hand.
  - **Suggested change:** a "gap → fix-task" replan branch for non-science gaps that
    creates a `fix`/`validate` pair with the right edges automatically.
- **Observation:** I did most of the work inline and recorded outputs into beads
  afterward, never formally entering the `execute` workflow.
  - **Suggested change:** document/support an "inline work, record-after" path in
    `execute` so recording results doesn't require having driven the task through the
    workflow machinery.

## Artifacts

- `mission.md`
- `pg_dr_tulu.py`
- `rewrite_answer_from_critiques.py`
- `rewrite_review_findings.md`
- `summary.md`
- `test_samples/` (pipeline inputs and rewritten outputs)
- `.beads/issues.jsonl` (exported beads issue graph, 10 issues)
