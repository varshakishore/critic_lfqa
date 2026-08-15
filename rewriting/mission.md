# Mission: Critic-Guided Rewriting

## Goal

Train better long-form scientific QA models by generating high-quality DPO preference data. The core hypothesis is that localized, critique-guided edits produce cleaner training signal than global answer rewrites: by constructing preference pairs that diverge at a single specific span, the model receives unambiguous feedback about what to fix and where, rather than having to infer which of many differences between chosen and rejected actually matters.

## Sub-goal

Improve and evaluate the pipeline that rewrites long-form scientific QA answers using structured critiques. The pipeline should make minimal, targeted local edits — fixing only what the critique flags — rather than regenerating answers wholesale.

## Pipeline

**Step 0 — Answer generation** (`pg_dr_tulu.py`, `process_sample`)  
Call `http://localhost:8007/ask` for each question. Produces a trace (reasoning + searches + results) and a final answer.

**Step 1 — Critique generation** (`pg_dr_tulu.py`, `updated_prompt_v1`)  
Send the full trace to an LLM. Produces structured critiques identifying specific issues in the plan or answer, with `critique_span` (where the problem is) and `edit_span` (where to apply the fix).

**Step 2 — Trace rewriting** (`rewrite_answer_from_critiques.py`, Step 1+2)  
Tag editable spans in the trace with `<can_edit>` using `edit_span`. For critiques that require new searches, results for queries are pre-fetched from the Semantic Scholar snippet API. . The LLM rewrites only the tagged spans, inserting the reasoning, `<call_tool>` + `<tool_output>` blocks in the Dr Tulu format and `PLACEHOLDER_{id}` where new search results should go. Placeholders are then replaced with the real formatted results.

**Step 3 — Answer rewriting** (`rewrite_answer_from_critiques.py`, Step 3)  
Tag editable spans in the answer with `<can_edit>` using `edit_span`. LLM rewrites only those spans using the updated trace as evidence, adding `<cite id="...">` where appropriate.

## Answer Format

Input JSONL (`drtulu_answers_w_critiques.jsonl`):
```
question         str   — the scientific question
original_answer  str   — answer produced by the generation model
original_trace   str   — full trace string (<think>, <call_tool>, <tool_output>, <answer>)
critique         str   — JSON string with a "local" list of critique objects
```

Each critique object:
```json
{
  "critique_span": [["first few words", "last few words"], ...],
  "edit_span":     [["first few words", "last few words"], ...],
  "location":      "plan" | "answer" | "both",
  "issue":         "description of the problem",
  "tag":           "3-5 word label",
  "organization_related": true | false,
  "search_required":      true | false,
  "s2_search_queries":    [{"query": "...", "year": "...", "authors": [], "field_of_study": "..."}]
}
```

`edit_span` convention: same as `critique_span` for rewrites/deletes. For insertions, start == end (same anchor string) signals an insertion point immediately after that string.

Output JSONL (`drtulu_answers_w_critiques_rewritten.jsonl`):
```
question         str   — unchanged
original_answer  str   — unchanged
original_trace   str   — unchanged
rewritten_trace  str   — trace after Step 1+2 (new searches inserted, placeholders filled)
critique         str   — unchanged critique JSON string
rewritten        str   — answer after Step 3
```

## DPO Pair Construction

After rewriting, create one DPO pair per local edit (per critique with a non-empty `edit_span`):

- **prompt**: question + answer text up to the start of the edit span
- **chosen**: rewritten text from that point forward
- **rejected**: original text from that point forward

Pairs share the same prefix and diverge exactly at the edit location. This gives N pairs per answer (one per critique) with a localized preference signal — the model learns to prefer the improved span in context rather than receiving a noisy global signal over two answers that differ in many places at once.

Plan-level critiques (location: "plan" or "both") that add new searches are handled at the trace level and do not directly produce answer-level DPO pairs.

## Key Questions to Investigate
1. Read some of the DR Tulu traces and answers to understand the DR Tulu format. Check if the trace rewriting is working correctly and the new searches queries are being inserted in the expected format. 
2. Are placeholder searches being inserted and filled correctly?
3. Is the rewrite model staying within `<can_edit>` bounds or drifting?
4. Does the rewritten answer actually address the flagged critiques?
