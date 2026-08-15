---
beads_snapshot: cf602f97a861fdecf4424192b409a1b70c7767f2f2a76919bf33d61ace01fa29
beads_epic: rewriting-5wj
generated_at: 2026-07-17T05:40:05Z
issue_count: 10
ready_count: 1
---

# Critic-guided rewriting: minimal targeted edits for DPO preference data

## Mission
Generate high-quality DPO preference data for long-form scientific QA by making
localized, critique-guided edits (cleaner training signal than global rewrites).
This session improves and evaluates the pipeline
(`pg_dr_tulu.py` → `rewrite_answer_from_critiques.py`) so rewrites make minimal,
targeted edits confined to `<can_edit>` spans — in both the trace and the answer,
since DPO runs on both.

## Research Question & Scope
Does the pipeline make minimal, targeted edits confined to `<can_edit>` spans that
correctly address flagged critiques, keep the trace well-formed, and fill searches
correctly — yielding clean single-locus DPO pairs on both trace and answer?
- **In scope:** `rewrite_answer_from_critiques.py` Steps 1–3 and its `test_samples/` outputs.
- **Out of scope:** critique-generation quality, the generation model, DPO training itself.
- **Success:** edits ONLY inside `<can_edit>` (trace + answer); well-formed DR Tulu
  trace; placeholders filled; searches in correct format; critiques actually addressed.

## Operational Definitions
- **well-formed trace** — `<think>`/`<call_tool>`/`<tool_output>` never nest, all balanced.
- **in-bounds edit** — every character outside a `<can_edit>` span is byte-identical to original.
- **anchor match** — `edit_span` start/end substrings are locatable in the exact text tagged.
- **critique addressed** — the flagged issue is resolved within the flagged span.

## Related Work
Empirical review of the `test_samples/` outputs → full write-up in
[`rewrite_review_findings.md`](rewrite_review_findings.md). Key findings:
- Q1: new searches inserted, counts match collected queries; but insertions could nest
  `call_tool`/`tool_output` in an unclosed `<think>`.
- Q2: placeholders fill correctly; no snippet-ID collisions; original snippets preserved.
- Q3: `<can_edit>` tags leak (acceptable) AND genuine out-of-bounds drift occurs even with clean tags.
- Q4: rewrites address critiques well semantically.
- Bugs: B1 malformed `<think>` insertions; B2 malformed `<can_edit>` tags (unmerged duplicate spans);
  B3 silent anchor-match failure on `<answer>`-wrapped anchors.

## Hypotheses
n/a — this session is an engineering review; gaps are tracked as fix/gap tasks below.

## Experimental Designs
n/a

## Results Summary
Three fixes applied and **re-validated on a fresh pipeline run** (G2, rewriting-4zw):

| Bug | Fix | Task | Re-run result |
|---|---|---|---|
| B1 malformed `<think>` inserts | prompt non-nesting invariant + `validate_trace_structure()` | rewriting-ry7 | ✅ both records 0 structure problems |
| B2 malformed `<can_edit>` tags | merge overlapping/duplicate spans before tagging | rewriting-tyl | ⚠️ input tags clean, but model still strips output tags in rec1 trace |
| B3 anchor-match failure | tag `<answer>`-wrapped block from the trace | rewriting-bas | ✅ rec0 answer: 0 unmatched anchors, 0 out-of-bounds drift; answers wrapped |

Placeholders all filled. rec1 fully in-bounds (trace + answer). **Remaining (rec0 trace only):**
device-name fix propagated into an untagged recap block, and `aortic`→`a` garbling of a
retrieved snippet inside a `<tool_output>` (reproduction corruption of non-editable evidence).

## G1 — out-of-bounds edits ELIMINATED (closed, rewriting-117)
Solved with three interlocking pieces, then validated on a fresh run:
1. **Snap** search-insertion spans (`search_required` + `start==end`) to their enclosing
   complete `<think>…</think>` block, so a new round lands at a block boundary (in `compute_edit_spans`).
2. **Prompt** teaches the model the well-formed insertion cases (between-blocks vs mid-`<think>`).
3. **Diff-based splice** (`splice_edits`) keeps the model's text ONLY inside editable spans and
   reverts everything else to the original — killing drift AND evidence corruption, and not
   relying on `<can_edit>` tags surviving.

Final re-run (both records): structure problems **0**, trace drift **0**, answer drift **0**,
all searches preserved (rec0 +6, rec1 +7), evidence corruption reverted, no leftover placeholders.

## Open Questions
- G3: whether the drift/corruption/malformation patterns (and the stray-`</think>` seen in
  sample 3's source) are widespread across all 50 records — only 2 validated so far.

## Status
- Closed: 8 (scope, definitions, literature_review, F1, F2, F3, G2, G1)
- In progress: 0
- Ready: 1 — rewriting-fsq (G3)
- Blocked: 0

### Next Steps
- rewriting-fsq [G3, ready]: run the prevalence scan across all 50 records to confirm the
  fixes hold at scale and quantify any residual patterns.
