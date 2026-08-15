# Critic-Guided Rewriting — Pipeline Review Findings

Empirical review of `rewrite_answer_from_critiques.py` against the `test_samples/`
outputs (3 records, later 2 after sample 3 was dropped). Organized by the four
key questions in `mission.md`, plus bugs found and fixes applied.

## The four key questions

1. **Trace rewriting & new-search insertion.** New searches ARE inserted, and
   the count matches the collected queries (rec0: 2→8 `<call_tool>` blocks = +6,
   matching its 6 unique queries). BUT insertions could break DR Tulu structure:
   the model nested `<call_tool>`/`<tool_output>` inside an unclosed `<think>`.
2. **Placeholders.** Working correctly — zero leftover `PLACEHOLDER_*` in any
   rewritten trace; no snippet-ID collisions (inserts use 6-hex prefixes, original
   generation uses 8-hex); original retrieved snippet text preserved verbatim.
3. **Staying within `<can_edit>` bounds.** Two problems: (a) `<can_edit>` tags leak
   into output — deemed acceptable (stripped later); (b) genuine out-of-bounds drift
   — the model edits identical text outside tagged spans (rec0: removed the wrong
   device descriptor in an untagged recap block).
4. **Do rewrites address critiques.** Yes — semantically strong. rec0 added the
   "Bottom line" takeaway, removed the erroneous device label, softened an
   overstated claim.

## Bugs found

- **B1 — malformed `<think>` insertions.** Model nested `<call_tool>`/`<tool_output>`
  in an unclosed `<think>`. Root cause: prompt prescribed "close `<think>` first"
  as a blanket rule (wrong — depends on whether the `<can_edit>` sits between blocks
  or mid-`<think>`); the real invariant is that think/call_tool/tool_output never nest.
- **B2 — malformed `<can_edit>` tags.** `insert_can_edit_tags` did not dedupe/merge
  overlapping or duplicate spans; two critiques sharing one span produced nested,
  word-splitting tags (`<can_edit><can_edit>…suppor</can_edit>ted…`), which the
  rewrite model then dropped entirely (rec1 trace: 0 tags out of 2 expected).
- **B3 — silent anchor-match failure.** `insert_can_edit_tags` locates spans by
  literal substring and silently `continue`s on no-match. Answer critiques carry
  `edit_span` anchors that include the `<answer>` wrapper (critiques are generated
  against the full trace), but the answer was tagged against the wrapper-less
  `original_answer` field → anchor never matched, span never tagged (rec0 takeaway).

## Fixes applied (verified on tagging side; pending pipeline re-run)

- **F1 (B1):** rewrote Step-1 prompt to state the non-nesting invariant and the two
  correct insertion cases (between-blocks vs mid-`<think>`); added
  `validate_trace_structure()` that flags nesting/imbalance after Step 2.
- **F2 (B2):** merge overlapping/duplicate/touching spans into disjoint intervals
  before inserting tags.
- **F3 (B3):** tag/rewrite the `<answer>…</answer>` block extracted from the trace
  (with wrapper) instead of the wrapper-less field; store the wrapped form as
  `original_answer` so original/rewritten stay consistent for DPO.

## Remaining gaps

- **G1 — out-of-bounds drift (open).** DPO runs on both trace AND answer, so edits
  must be confined to `<can_edit>` in both. Even with clean tags (rec0), the model
  edited outside them. Candidate fix: splice-revert (original outside tags + model
  inside), viable now that tags are well-formed; or per-span structured-edit redesign.
- **G2 — re-validation (open).** Re-run pipeline; confirm validator stays quiet,
  tags survive (esp. rec1 trace), anchors all match.
- **G3 — prevalence scan (open).** Run across all 50 records to measure how common
  each pattern is (esp. the stray-`</think>` artifact seen in the source of sample 3,
  which is an upstream DR Tulu generation issue, not a rewriter bug).
