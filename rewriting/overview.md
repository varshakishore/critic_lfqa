# Critique-Guided Rewriting — System Overview

_Last updated: 2026-08-28_

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
4. **DPO pairs** (`build_dpo_pairs.py`) — two arms from the same input:
   `--mode local` (default) emits one pair per change, `chosen` and `rejected`
   sharing a byte-identical prefix and diverging at that change; `--mode full`
   emits the whole-sample **baseline** the hypothesis is measured against.

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
| Prompt-echo leaks (luna copying its prompt into output) | 10 → **0** (repaired) |
| **Structure problems introduced by rewriting** | **0 / 996** (was 18; repaired) |
| Rewritten traces still flagged (inherited from source) | 11 / 996 |
| Reflections with hallucinated template tags | ~5 (e.g. rec 247, 481, 525) |

Two classes of rewriting-introduced trace damage were found and fixed:

1. **Prompt echo (10 records):** luna occasionally copied its Step-1 rewrite prompt
   (`--- Critiques to fix …`) into the trace; it survived splicing because it landed
   inside an editable span. **Fix:** `_strip_prompt_echo()` truncates any leaked
   scaffold before the diff in `splice_edits`, so difflib reverts that region to the
   original. Existing file patched (backup: `…_v3.jsonl.bak_preleak`).
2. **Tag-balance defects (11 records):** in-place edits (8) and search-round
   insertions (3) introduced stray/nested `<think>`/`</think>`/`</tool_output>` tags.
   **Fix:** `normalize_trace_structure()` applies minimal, tag-only repairs (never
   touches prose) and runs as a guard after Step 1b. Existing file patched.

After both fixes: **0 structure problems introduced by rewriting**; the only 11
still-flagged traces carry malformation inherited from the source DR Tulu traces.

3. **Snippet-id format/scheme mismatch:** inserted search rounds used a fabricated
   6-hex prefix (`hash(query) % 0xFFFFFF`, e.g. `1de5a6-0`) while DR Tulu's native
   snippet ids are 8-hex, unique per round (e.g. `ad1dd40e-0`). DR Tulu's actual
   `generate_snippet_id()` is `md5(uuid4())[:8]` — a **random** 8-hex per round.
   **Fix:** `new_snippet_id()` now matches it exactly (`md5(uuid4())[:8]`, random,
   generated once per round; `insert_search_rounds` resolves it by query via
   `search_results`). Existing file retrofitted twice — first 6-hex→8-hex, then to
   genuinely random ids — remapping snippet ids and answer cites together, single-
   pass so every cite still resolves. Verified: 0 six-hex remaining, 0 cross-record
   shared prefixes, dangling-cite count invariant (177 before = 177 after, so no
   regression). Backups: `…_v3.jsonl.bak_pre_idfix` (pre-retrofit).

**Separate pre-existing issue (not fixed — a DR Tulu data characteristic):** answers
cite snippet ids absent from their own trace — 177 in the rewritten answers, and
**350 in DR Tulu's own original answers** (baseline). Includes citing a snippet
index that was never retrieved (e.g. `-5` when the round returned indices 0–4).
Unrelated to id format; inherited from the source data.

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

## DPO pair construction (Stage 4)

`build_dpo_pairs.py` turns the Stage-3 output into preference pairs. Find where
original and rewritten diverge; find where they agree again; emit one pair:

```
prompt   = <DR Tulu prompt> + REWRITTEN text up to the divergence point
chosen   = rewritten segment  + agreed tail
rejected = original segment   + agreed tail
```

The prefix is the **rewritten** text, so every earlier fix is already applied and
each pair reads: *having done everything right up to here, prefer the corrected
continuation.* (`--trace-prefix`/`--answer-prefix original` conditions on the
unrewritten text instead.)

The **agreed tail** runs from the reconvergence point to the end of the sentence it
completes. Terminators are `.!?` plus closing tags (`</think>`, `</answer>`,
`</tool_output>`, `</call_tool>`, `</snippet>`) and newlines, since trace text is
not uniformly prose. It exists so pure insertions/deletions have a non-empty side,
and stopping at a sentence keeps it short — every tail token is loss mass on text
carrying no preference (`small_synthetic_exp` H7). Median tail **48 chars**.

Three things end the tail, whichever comes first: the sentence boundary, **the next
divergence** (it has to win, or the pair would differ in two places — so some tails
end mid-sentence), or `--tail-max-chars` (1500).

**Diffing** is two-level: lines first (a 100 kB trace is only ~350 lines), then
words inside each changed line-block. A single-level word diff over ~30k tokens is
O(n²) and takes minutes per record. Changes separated by fewer than `--merge-gap`
(default 40) identical chars are coalesced, so one reworded sentence doesn't
shatter into a dozen word-level pairs.

**Answer pairs additionally carry the whole rewritten trace** ahead of the answer
prefix. That is required, not a preference: Step 3 is called with
`trace=rewritten_trace`, so the rewritten answer cites snippet IDs that exist only
in rounds inserted at Step 1b. An original-trace prefix would make `chosen` cite
evidence absent from its own context.

| pairs | prefix |
|---|---|
| trace | DR Tulu prompt + **rewritten** trace up to the divergence |
| answer | DR Tulu prompt + **rewritten** trace (whole) + **rewritten** answer up to the divergence |

### Prompt prefix — `drtulu_prompt_template.txt`

The prefix must reproduce the prompt DR Tulu actually generated against, or the
policy is trained in a context it never sees at inference. That file holds it,
rendered byte-exactly (6,266 chars / 1,461 tokens), ending
`<|im_end|>\n<|im_start|>assistant\n` — the trace then opens with `<think>`.

```
dr-tulu/agent/workflows/query_direct.py  POST /ask -> AutoReasonSearchWorkflow
  auto_search_sft_s2_only_hamish.yaml (no prompt_version -> default "v20250907")
  dataset_name "sqav2" -> long_form
-> [system: unified_tool_calling_v20250907.yaml system_prompt,
    user:   question + "\n\n" + additional_instructions.long_form]
-> LLMToolClient._messages_to_prompt: Qwen/Qwen3-8B apply_chat_template(
       tokenize=False, add_generation_prompt=True)
```

Verified to match the RL prompt (same YAML, same md5, `train_dr_tulu.sh:81`) and
format-identical to SFT (LLaMA-Factory `template: qwen3`). Regenerate with
`--regen-prompt-template <path-to-dr-tulu/agent>`. EOS is `<|im_end|>` (151645).

### End-of-text

Each pair carries a `complete` flag — True only when its completion runs to
`</answer>`. TRL's `DPOTrainer.tokenize_row` appends EOS to *every* completion
unconditionally, which would teach the model to stop mid-answer, so training must
use `MaskedDPOTrainer` (`dpo_trainer_utils.py`) with
`remove_unused_columns=False`. Trace pairs are never `complete` — the sequence
continues into the answer.

### Snippet-ID namespaces

Useful for telling inserted evidence from DR Tulu's own. Both are
`{round}-{snippet_index}`; the prefixes can't collide because they differ in length:

| | generated by | shape |
|---|---|---|
| DR Tulu | `base.py:84` `str(uuid.uuid4())[:8]`, fresh per **tool call** | **8** hex, e.g. `ad1dd40e-0` |
| inserted rounds | `rewrite_answer_from_critiques.py:665` `hex(abs(hash(query)) % 0xFFFFFF)[2:].zfill(6)`, keyed on the **query string** | **6** hex, e.g. `3f2a1b-2` |

Checked on 200 records: original traces carry only 8-hex ids, rewritten carry both,
zero collisions.

### Latest build

996-record v3 input → **11,084 pairs**, record-level split (no question leakage):

| | pairs | |
|---|---|---|
| train / val / test | 10,104 / 473 / 507 | 1,299 MB |
| answer | 9,215 | 7,526 replace / 1,250 insert / 439 delete |
| trace | 1,869 | 973 replace / 858 insert / 38 delete |
| carrying mask spans | 678 (train) | 666 of 1,733 trace pairs; 0 answer pairs |
| `complete` | 379 (3.4%) | |

Median prefix 116k chars (~29k tok), chosen 397, rejected 256, tail 48. Skipped: 10
records with an empty `rewritten_trace`, 17 malformed answers, and trace pairs from
records where rewriting introduced structure problems.

### Tool-output masking

Answer pairs are clean generated text (0 / 8,371 contain `<tool_output>`). **Trace
pairs are not**: 666 / 1,733 do, because a programmatically inserted search round
*contains* its own retrieved snippets and the whole round is one diff block — ~90%
of trace-pair completion tokens are snippet text.

The policy never emits those tokens: `</call_tool>` is a hard stop sequence
(`tool_parsers.py:379`), generation halts there, and the harness splices the results
in. Both DR Tulu training stages exclude them — SFT via `use_span_masking: true,
mask_span_types: "tool_output"` (a DR Tulu **addition** to LLaMA-Factory:
`data/processor/span_masked.py` + `data/tokenizer_utils.py`, absent from upstream
v0.9.4 and HEAD), RL via `mask_tool_use=True` (`grpo_fast.py:910`).

So every pair carries `chosen_mask_spans` / `rejected_mask_spans` — character spans
dropped from the DPO log-prob sum. **The tokens stay in the sequence and are still
attended to**, so a later reflection can read the snippets it reacts to; only their
own log-probs leave `Σ_t log π(y_t | x, y_<t)`.

`MaskedDPOTrainer` applies them:

- **Char → token** via `tokenizer(..., return_offsets_mapping=True)` on the whole
  string, so ids are byte-identical to the unmasked path. Tokenizing keep/drop
  segments separately would change ids at every seam (BPE merges across the
  boundary — measured: 3,716 vs 3,714 tokens). Requires a fast tokenizer.
- A token overlapping a masked span at all is masked (**drop-dominant**); straddling
  tokens are ~2 per pair.
- TRL exposes no user-supplied loss mask: `loss_mask` is built from
  `completion_attention_mask`, the *same* tensor that becomes the attention mask, so
  a doctored attention mask would hide snippets from attention and break
  `flush_left`/`flush_right`. The fix needs a separate tensor, i.e. editing
  `concatenated_forward` (247 lines). Rather than vendor a copy that would silently
  rot on a TRL upgrade, the module **recompiles TRL's own source with one clause
  appended** after the `loss_mask` construction, and raises if the anchor text ever
  moves. Pinned to trl 0.24.0.

### Baseline — whole-sample pairs (`--mode full`)

The control arm: the "global answer rewrite" that localized edits are supposed to
beat. One pair per record, no diffing.

```bash
python build_dpo_pairs.py --mode full     # -> ..._dpo_pairs_full_{train,val,test}.jsonl
```

```
prompt   = the DR Tulu prompt alone (question, no generated text)
rejected = the full original sequence   (trace + answer)
chosen   = the full rewritten sequence  (rewritten trace + rewritten answer)
```

Same record schema as the local arm, so `MaskedDPOTrainer` / `load_pairs()` work
unchanged. Both sides end at `</answer>`, so `complete` is True (950 / 951) and both
take an EOS — unlike local pairs, which almost never do.

| | pairs | |
|---|---|---|
| train / val / test | 858 / 45 / 48 | 199 MB |
| `complete` | 950 (99.9%) | |

Median prompt 6,369 chars, chosen 119,328, rejected 79,133.

**Splits are aligned with the local arm** — same seed and record shuffle, so every
record lands in the same split in both datasets (verified: zero cross-dataset
leakage). The record *sets* differ slightly:

- **32 records in local but not the baseline.** Full mode needs *both* documents to
  pass their gates (a malformed answer poisons the whole `chosen`), while local mode
  still emits trace pairs from a record whose answer failed.
- **1 record (641) the other way** — its only two changes are a 48.9k-char trace
  insertion and a 23.6k-char answer rewrite, both over `--max-change-chars` (20k), so
  local mode emits nothing while the baseline still has a whole-sample pair.

Intersect the record ids first if you need the arms exactly matched.

**Run both arms with masking on.** 81.5% of the baseline's chosen tokens are tool
output; unmasked, it would be ~23k tokens of loss per side, nearly all retrieved
snippets — the comparison would measure the masking rather than the hypothesis.
Effective lengths after masking:

| | total tokens | unmasked |
|---|---|---|
| chosen | 23,101 | 4,267 |
| rejected | — | 3,715 |

Even masked the arms differ ~10× in tokens per pair (~4.3k vs ~400) and 951 vs
11,084 pairs. That asymmetry *is* the hypothesis, not a flaw — but report
`max_length`, epochs, and effective tokens seen per arm rather than assuming they
are comparable.

## Key files

| File | Role |
|---|---|
| `pg_dr_tulu.py` | Stage 1–2: DR Tulu answer generation + `gpt-5.4` critique generation. `ANSWER_ONLY=1` does generation only. |
| `rewrite_answer_from_critiques.py` | Stage 3: the rewrite pipeline (tag → splice → programmatic search → answer rewrite). |
| `build_dpo_pairs.py` | Stage 4: builds the DPO preference pairs. `--mode local` (per-change) or `--mode full` (whole-sample baseline). |
| `dpo_trainer_utils.py` | `MaskedDPOTrainer`: EOS only when `complete`, tool-output spans excluded from the loss. Plus `load_pairs()`. |
| `drtulu_prompt_template.txt` | The exact DR Tulu inference prompt, `{question}` placeholder. |
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
- Snippet-ID generation for inserted rounds is being reworked (`hash()` is salted
  per process, so 6-hex prefixes are not stable across runs); the pair data will be
  rebuilt afterwards.
- Masking the **agreed tail** as well (H7 on the 9,215 answer pairs, where the tail
  is ~10% of completion tokens) — the machinery exists, it is a second span list;
  **not implemented**, since masking was scoped to tool outputs.
- No training script yet — `small_synthetic_exp/run_dpo_training.py` is em-dash
  specific (Qwen3-4B, connector-rate callback) and needs adapting to DR-Tulu-8B,
  `remove_unused_columns=False`, and `load_pairs()`.
- The two arms cover slightly different record sets (32 / 1, see above); a
  `--restrict-records` flag to force an exact intersection is **not** implemented.
- `--question-template` defaults to the checked-in `drtulu_prompt_template.txt`;
  re-render it if the DR Tulu prompts or `prompt_version` default change.
