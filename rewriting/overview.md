# Critique-Guided Rewriting — System Overview

_Last updated: 2026-08-26_

## Goal

Produce clean **DPO preference pairs** for long-form scientific QA (LFQA). The core
hypothesis: **localized, critique-guided edits** — where the chosen and rejected
answers diverge at a *single* critique-flagged span — give a cleaner training
signal than global answer rewrites. A pair that differs in exactly one place
teaches the model precisely *what* to fix and *where*.

The data is DR Tulu agentic QA: each record is a question, a multi-round
reasoning **trace** (interleaved thinking + retrieval), and a final cited
**answer**. We keep the DPO signal on both the trace and the answer, editing only
inside critique-flagged spans.

## Pipeline (4 stages)

```
generation → critique → rewrite → DPO pairs
```

1. **Generation** — DR Tulu produces the original trace + answer.
   Source data: HF `rl-research/dr-tulu-rl-data`, `sqa_1k` split (1000 records).
2. **Critique** (`pg_dr_tulu.py`, model `gpt-5.4`) — emits structured JSON
   critiques, each anchoring a verbatim span and describing the issue.
3. **Rewrite** (`rewrite_answer_from_critiques.py`) — applies the critiques as
   localized edits to the trace and answer, inserting new retrieval rounds where a
   critique calls for more evidence.
4. **DPO pairs** — (original, rewritten) become (rejected, chosen).

## Trace format (DR Tulu)

```
<think> reasoning </think>
<call_tool name="snippet_search" ...> query </call_tool>
<tool_output><snippet id="X-N">Title:… Snippet:…</snippet>…</tool_output>
[ bare reasoning ] </think>          ← 2nd+ rounds: reasoning is NOT re-opened
<call_tool …> … </call_tool>
<tool_output>…</tool_output>
…
<answer> … <cite id="…">…</cite> … </answer>
```

**Important convention:** only the *first* reasoning block is wrapped
`<think>…</think>`. Every reasoning block after a `<tool_output>` is emitted
**bare** (no opening `<think>`) and terminated by a lone `</think>`. This is why a
DR Tulu trace has roughly 2× as many `</think>` as `<think>` and is *not*
malformed — it's the native format.

## Critique JSON

```json
{"local": [{
  "critique_span": "…", "edit_span": [["start","end"], …],
  "location": "plan | answer | both",
  "issue": "…", "tag": "…",
  "organization_related": bool,
  "search_required": bool,
  "s2_search_queries": [ … ]
}]}
```

- `edit_span` entries are `[start, end]` **verbatim** anchors into the text.
  `start == end` signals an *insertion point* (not a replacement).
- The pipeline locates a span via `text.find(start[:40])` then
  `text.find(end[-40:], …)`.

## Rewrite pipeline — how it works

`rewrite_answer_from_critiques.py`, per record:

1. **Tag editable spans.** `compute_edit_spans` normalizes/merges the critique
   spans; `insert_can_edit_tags` wraps flagged regions in `<can_edit>…</can_edit>`
   so the model only edits inside them.
2. **In-place trace edit (Step 1).** For `plan`-location critiques, the model
   reproduces the trace with edits confined to the tagged spans. `splice_edits`
   (difflib `SequenceMatcher`) then keeps model text *only inside* editable spans
   and reverts everything else to the original — making the result immune to
   drift, dropped tags, and evidence corruption.
3. **Programmatic search insertion (Step 1b).** When a critique needs more
   evidence (`search_required`), the **pipeline** owns the structure: it runs the
   real Semantic Scholar snippet search, builds a well-formed
   `<think>reflection</think><call_tool>query</call_tool><tool_output>real
   results</tool_output>` round, and inserts it *after the current round's*
   `</tool_output>`. The model writes only the reflection prose (tag-free), so it
   can't corrupt structure. Insertion points snap to the enclosing complete
   reasoning block.
4. **Answer rewrite (Step 2).** The `<answer>` block is edited in place under the
   same tag-and-splice discipline.

### Backends & cost

- `REWRITE_BACKEND=gpt` (default) → OpenAI Responses API. `GPT_MODEL` default
  `gpt-5.6-luna` ($0.20 in / $1.20 out per 1M).
- `REWRITE_BACKEND=glm` → self-hosted GLM-5.2-FP8 (OpenAI-compatible
  chat.completions, `$0`).
- `gpt-5.4` = $2.50 / $15.00 (used for critique generation).
- Config: `PROGRAMMATIC_SEARCH=1`, `COST_LIMIT=100`, `MAX_WORKERS=20` (thread
  pool), resume-by-skipping-done + append. Output file name encodes model slug +
  `RUN_TAG` (currently `v3`).

## Robustness features (added iteratively)

- **Diff-based splice** — reverts all non-editable regions to original.
- **`_norm_span`** — coerces malformed spans (`["one string"]` / bare string →
  `(s, s)` insertion).
- **`parse_critique`** — repairs the common JSON failure (unescaped quotes inside
  `<cite …>`-style tags); recovers ~2 of 4 known parse failures.
- **`extract_answer_block`** — handles truncated / missing `</answer>`.
- **S2 retry/backoff** — exponential backoff, special-cases 429 (rate limit);
  504s are server-side timeouts on heavy snippet queries.
- **`validate_trace_structure`** — tag-balance checker that understands the DR
  Tulu round convention (bare reasoning resumes after `</tool_output>`; a lone
  `</think>` there is legitimate). Flags only genuine nesting/imbalance bugs.

## Current state (as of this session)

**Latest run:** `samples_1000/drtulu_answers_w_critiques_rewritten_gpt-5.6-luna_v3.jsonl`
— 996 records rewritten with `gpt-5.6-luna`, programmatic search insertion on.

**Quality audit — it looks good:**

| Check | Result |
|---|---|
| Answers `<answer>`-wrapped | 996 / 996 |
| Empty answers | 0 |
| Leftover placeholders | 0 |
| Answers identical to original (no edit) | 0 |
| Search rounds inserted (total) | 4500 |
| Source traces genuinely malformed | 12 / 996 |
| Rewritten traces flagged | 29 / 996 |
| **Structure problems introduced by rewriting** | **18 / 996 (1.8%)** |
| Reflections with hallucinated template tags | ~5 (e.g. rec 24, 247, 481, 525) |

**Key finding:** an initial scan showed 621/996 traces "malformed," but this was a
validator bug — it enforced strict XML nesting and flagged DR Tulu's native
closed-but-not-reopened `</think>` convention (609 source traces already
"failed"). After fixing `validate_trace_structure` to model the round convention,
the real picture is: **98.2% of rewritten traces are structurally clean**, and
rewriting introduces genuine problems in only 18 records (mostly a dangling
`</think>` when a reflection insertion goes sideways — e.g. luna emitting
JSON-array-style reflection text into the trace). The answer side is fully clean.

The 18 introduced-problem records: `24, 32, 41, 84, 341, 387, 479, 480, 512, 516,
550, 555, 566, 726, 779, 912, 938, 964`. They are individually re-runnable if we
want to patch them.

## Key files

| File | Role |
|---|---|
| `pg_dr_tulu.py` | Stage 1–2: DR Tulu answer generation + `gpt-5.4` critique generation. `ANSWER_ONLY=1` does generation only. |
| `rewrite_answer_from_critiques.py` | Stage 3: the rewrite pipeline (tag → splice → programmatic search → answer rewrite). |
| `generate_viewer.py` | Builds `trace_viewer.html` — side-by-side model-comparison viewer (old vs rewritten trace/answer, critiques, word-diff, inserted-round highlighting). |
| `samples_1000/` | 1000-record inputs and rewritten outputs (~100–200 MB JSONL). |
| `test_samples/` | Small (2–10 record) dev/validation samples. |
| `RESEARCH_CHALLENGE.md` | Write-up submitted to `allenai/asta-research-challenge` (PR #12). |

## Open items

- Patch or drop the 18 trace regressions + ~5 hallucinated-tag reflections.
- Step-1 latency: the in-place trace edit asks the model to reproduce the full
  trace (~17–40k tokens) to make small edits — slow on long traces. A per-span
  in-place rewrite was proposed but **not** implemented.
- Cost accounting under-counts records that fail/skip *after* LLM calls (proposed
  incremental fix, not implemented).
