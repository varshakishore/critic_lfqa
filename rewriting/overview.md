# Critique-Guided Rewriting — System Overview

_Last updated: 2026-10-30_

> **Status in one paragraph.** The data pipeline works and the rewrites are genuinely
> better answers — a pairwise `gpt-5.4` judge prefers `chosen` over `rejected` in 89%
> of judgments. But **DPO does not transfer that quality**: neither arm (local or
> full, from the SFT or the RL checkpoint) beats its base model behaviourally. The
> local arm reliably learns *formatting* decided at the divergence point (`<think>`
> re-opening, answer length) but not the content of the fixes. See
> [Behavioural eval](#behavioural-eval-eval) and [Open items](#open-items).

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

1. **Generation** — **`rl-research/DR-Tulu-SFT-8B`** (not the RL model) produces the
   original trace + answer, through `dr-tulu/agent/workflows/query_direct.py`
   (S2 search only, ≤5 tool calls, temperature 0.7).
   Source data: HF `rl-research/dr-tulu-rl-data`, `sqa_1k` split (1000 records).
   Confirmed empirically: fresh SFT generations under that config match the stored
   originals on every format statistic, while DR-Tulu-8B does not (see
   [Trace format](#trace-format-dr-tulu)). The SFT model is also DR Tulu's RL init
   (`train_dr_tulu.sh:1`), and `sqa_1k` is in DR Tulu's RL data, so DR-Tulu-8B has
   been RL-trained on these exact questions.
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
[<think>] reasoning </think>        ← 2nd+ rounds: <think> re-opened or not, see below
<call_tool …> … </call_tool>
<tool_output>…</tool_output>
…
<answer> … <cite id="…">…</cite> … </answer>
```

**Two styles, and which model uses which.** A reasoning block after a
`<tool_output>` is either re-opened with `<think>` or emitted **bare** and closed by
a lone `</think>`. Both parse, and a bare block is *not* malformed — but bare is the
**SFT model's habit, not "the native format"** as this file used to say:

| rounds that re-open `<think>` after `</tool_output>` | |
|---|---|
| original traces = DR-Tulu-SFT-8B (996 records) | 46% |
| DR-Tulu-SFT-8B, fresh eval generations | 43% |
| DR-Tulu-8B (RL), fresh eval generations | **95%** |
| rewritten traces (`chosen`) | 71% |

SFT picks one style per trace (≈46% never re-open, ≈35% always). The rewrites sit at
71% because Step 1b builds every inserted round from a template that opens
`<think>` — a rewrite-pipeline artifact, not a critique fix. The two checkpoints
differ in answer style too: SFT opens 63% of answers with a heading and 0% with
"Below is…"; the RL model 13% and 53%.

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

## Rewrite run and quality audit (Stage 3)

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
validator bug — it enforced strict XML nesting and flagged the SFT model's
closed-but-not-reopened `</think>` style (609 source traces already
"failed"). After fixing `validate_trace_structure` to model the round convention,
the real picture is: **98.2% of rewritten traces are structurally clean**, and
rewriting introduces genuine problems in only 18 records (mostly a dangling
`</think>` when a reflection insertion goes sideways — e.g. luna emitting
JSON-array-style reflection text into the trace). The answer side is fully clean.

The 18 introduced-problem records: `24, 32, 41, 84, 341, 387, 479, 480, 512, 516,
550, 555, 566, 726, 779, 912, 938, 964`. They are individually re-runnable if we
want to patch them.

## DPO pair construction (Stage 4)

**Input lineage.** Both arms build from the twice-repaired Stage-3 output:

```
…_v3.jsonl              original rewrite run
   ↓  snippet-id rework + trace-regression repairs
…_v3_mod.jsonl          ids now md5(uuid4())[:8] like DR Tulu's; 18 regressions + 10
                        prompt-echo leaks fixed (975/996 records differ only in ids)
   ↓  repair_citations.py
…_v3_mod_fixed.jsonl    37 resolvable dangling <cite> ids repaired  ← DEFAULT_INPUT
```

`build_dpo_pairs.py` turns that into preference pairs. Find where original and
rewritten diverge; find where they agree again; emit one pair:

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
`</answer>`. TRL appends EOS to *every* non-conversational completion (in 1.13 via
`add_eos` inside `_prepare_dataset`), which would teach the model to stop
mid-answer, so training must use `MaskedDPOTrainer` (`dpo_trainer_utils.py`) with
`remove_unused_columns=False`. Trace pairs are never `complete` — the sequence
continues into the answer. Only 3.6% of local pairs take an EOS, versus 100% of
baseline pairs.

### Snippet IDs — no longer distinguishable by shape

Inserted rounds originally used a 6-hex prefix (`hash(query) % 0xFFFFFF`) while DR
Tulu's own are 8-hex, so the two were separable by length. The `_mod` rework moved
inserted rounds to `md5(uuid4())[:8]` to match DR Tulu exactly, which is correct but
**removes that signal**: both namespaces are now 8 hex + `-` + index.

To tell inserted evidence from native, set-difference the traces instead:

```python
inserted = snippet_ids(rewritten_trace) - snippet_ids(original_trace)
```

Corpus-wide that is 22,572 inserted snippets. `hash()` was also salted per process,
so the old 6-hex prefixes were not stable across runs; the uuid4 scheme fixes that
too. Zero collisions with native ids after the change.

### Citation integrity

Checked with the same set-difference. Counting `<cite id=…>` values with no matching
`<snippet id=…>` anywhere in the record's trace:

| | original answers | rewritten answers |
|---|---|---|
| dangling citations | 953 (169 records) | 698 → **661** after repair |

Of the rewritten ones, **621 are inherited** from DR Tulu's own answer and **77 were
introduced** by the rewrite (37 of those repaired by `repair_citations.py`, leaving
40). The rewrite *removes* 332 inherited danglers, so it fixes 4.3× more broken
citations than it creates — and those 332 are real preference signal, which is why
inherited danglers are never "repaired" on one side only.

### Rewritten-side gate

A record is skipped **entirely** when its rewritten output has a format or citation
defect, because every pair from it puts that output in `chosen` — the training
target. **Absolute, not a delta** against the source: a malformed `chosen` teaches
malformed output whether the rewriter caused the defect or inherited it. The
ORIGINAL is never gated; `rejected` is the dispreferred side, so its flaws are what
the preference points away from.

| rejected for | records |
|---|---|
| `dangling_cite` | 105 (+6 in combination) |
| `malformed_cite_tag` (a `<` or `>` inside the quoted id — the quote never closed) | 24 (+2) |
| `answer_wrapper` | 14 (+3) |
| `trace_structure` | 10 (+1) |
| `empty_trace` | 10 |
| **records kept after the gate** | **827 / 996 (83%)** |
| **…and after the 32,768-token length filter** | **647 / 996 (65%)** |

Whitespace inside a cite value is fine — `id="a, b"` is a legitimate multi-id
citation, and treating it as malformed would cost 11 records for nothing. Disable
the gate with `--no-gate-rewritten`.

Note that 85 of the 111 citation rejections are records where the *original* is at
fault and the rewrite merely failed to fix it. Gating only on *introduced* danglers
would keep those (~+70 records); that is a one-line change if wanted.

### Stray `<answer>` tags

85 rewritten answers (10%) carry a premature `</answer>` mid-answer — typically ~5%
of the way in, right after a heading — and then keep writing. The originals have
**zero**, so it is purely rewrite noise. It matters because `</answer>` is where
generation *stops* at inference, so a `chosen` containing one teaches the model to
truncate itself.

`strip_stray_answer_tags` keeps the first `<answer>` and the last `</answer>` and
drops the rest, which covers all four observed variants — (1,2) ×82, (1,3), (1,4),
and one (2,1) stray *opener*. Applied at load time, before the gate and the diff,
and reported in the build summary. `--keep-stray-answer-tags` disables it.

Because the normalisation lives in the builder rather than the source file,
`validate_pairs.py` imports and applies the same function — otherwise all 85 records
would fail reconstruction against the unnormalised source.

### Context-length filter

Qwen3-8B (and DR-Tulu-8B, unchanged) reports `max_position_embeddings = 40,960`
with `rope_type: default` — no YaRN — but was natively trained to **32,768**.
Sequences here routinely exceed both: local prompts run to 46k tokens, baseline
`prompt + chosen` to 50k.

`--max-sequence-tokens` (default **32,768**) drops any record whose full sequence
exceeds the budget, rather than truncating at training time. Truncation would be
badly behaved here:

- It is **one-sided**. `rejected` tops out at 30,537 tokens, so any cap above that
  clips `chosen` only — and `chosen` is already +51% longer, so it would selectively
  shrink the most length-skewed pairs (the truncated 10% have a chosen−rejected gap
  of 14,476 tokens versus 7,740 for the rest).
- It **eats the answer**. The `<answer>` block is the last ~10% of a baseline
  sequence and TRL truncates completions from the right.

Dropping is symmetric and keeps every surviving pair intact. Cost:

| cap | records lost | records kept | local pairs |
|---|---|---|---|
| 40,960 (configured max) | 23 | 804 (97%) | 9,800 |
| **32,768 (natively trained)** | **224** | **647 (78%)** | **7,586** |

Records are *marked*, not removed from the list: `record_idx` must stay the source
line number (the validator indexes by it) and `len(records)` drives the split
shuffle, so filtering the list would silently reshuffle every split.

Because nothing is truncated, and because trl 1.13 dropped the separate prompt and
completion caps in favour of a single `max_length`, no truncation happens at
training time.

### Latest build

647 records (after gate + length filter) → **7,586 local pairs** and **647
baseline pairs**. **Train/test only** — `--val-frac` defaults to 0, and the builder
writes only splits that have records.

| | local | baseline |
|---|---|---|
| train / test | 7,260 / 326 | 616 / 31 |
| records | 616 / 31 | 616 / 31 |
| size | 733 MB / 33 MB | 114 MB / 5.6 MB |
| answer / trace | 5,774 / 1,812 | — |
| `complete` (EOS-eligible) | 274 (3.6%) | 647 (100%) |
| prompt tokens (median) | 19,938 | ~1,480 |
| unmasked completion (median) | 91 chosen / 55 rejected | 4,770 / 4,134 |
| gradient-bearing tokens / epoch | 2.68 M | ~5.1 M |

The arms are matched: identical records in identical splits, zero cross-arm mismatch.

**Signal density is the point.** Only ~15% of the baseline's `chosen` sequence is
critique-flagged edit — the rest is text it is told to prefer for no critique-related
reason — against ~90% for the local arm. So the local arm has *more* critique-relevant
signal per epoch from fewer tokens. Do not token-match the arms; tune each separately
and compare bests.

**Eval is on `test`.** With no val split, anything chosen by watching the eval curve
(epochs, β, checkpoint) makes that number optimistic. For a two-arm comparison under
identical settings that is usually acceptable, but the absolute figures are not clean
held-out estimates.

### Tool-output masking

Answer pairs are clean generated text (0 contain `<tool_output>`). **Trace pairs are
not**: a programmatically inserted search round *contains* its own retrieved snippets
and the whole round is one diff block, so ~90% of trace-pair completion tokens are
snippet text (~81% of the baseline's).

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
- **The cap must measure trainable content.** `--max-change-chars` originally
  counted raw characters, which silently dropped the *largest* search insertions —
  a 30,000-char block that is 90.8% masked has only 2,749 trainable chars. Fixing it
  to count post-mask length recovered **791 trace pairs (+49%)**.
- **Ragged blocks.** A change can begin inside the preceding tool output, so the
  completion opens with an orphan `</snippet></tool_output>` and there is no opener
  to anchor on. `tool_output_spans` masks a leading orphan closer from position 0,
  and an unclosed opener to end-of-string. Before that fix, 541 pairs (5.8%) leaked
  tool-output tags; now 15, all bare `</snippet>` with no closer or snippet ids the
  model wrote inline in prose.
- **trl 1.13 makes this clean.** `_compute_loss` reads the per-token mask straight
  from the batch (`completion_mask = inputs["completion_mask"]`) and the reference
  path reuses the *same* tensor, so policy and reference log-probs cannot disagree
  about what was masked. No source patching. (On 0.24 the mask was derived from
  `completion_attention_mask` — the same tensor used for attention — which forced a
  247-line recompile of `concatenated_forward`. That approach is gone.)

### Baseline — whole-sample pairs (`--mode full`)

The control arm: the "global answer rewrite" that localized edits are supposed to
beat. One pair per record, no diffing.

```bash
python build_dpo_pairs.py --mode full     # -> ..._dpo_pairs_full_{train,test}.jsonl
```

```
prompt   = the DR Tulu prompt alone (question, no generated text)
rejected = the full original sequence   (trace + answer)
chosen   = the full rewritten sequence  (rewritten trace + rewritten answer)
```

Same record schema as the local arm, so `MaskedDPOTrainer` / `load_pairs()` work
unchanged. Both sides end at `</answer>`, so `complete` is True for **all 647** and
both take an EOS — unlike local pairs, of which only 3.6% do.

| | pairs | |
|---|---|---|
| train / test | 616 / 31 | 120 MB |
| `complete` | 647 (100%) | |

Length asymmetry, the thing most likely to make this control misleading:

| | chosen | rejected | difference |
|---|---|---|---|
| characters | 118,234 | 78,245 | +39,705 (+51%) |
| tokens, raw | 24,941 | 16,579 | +8,358 (+51%) |
| tokens, unmasked | 4,770 | 4,134 | +686 (+13%) |

Raw, `chosen` is shorter in only **3%** of pairs — about as systematic a confound as
exists, and DPO is length-sensitive. Masking cuts it to +13% (shorter in 17%), which
is why the baseline must be run masked or it will learn "longer" and look like it
learned "searches more". `--loss-type ipo` normalises by length if you want it
tighter; worth running the baseline both ways.

**Splits are aligned with the local arm** — same seed and record shuffle, and the
record sets are now **identical**, since both the gate and the length filter reject a
record outright rather than letting one document through. Zero mismatch, zero
cross-dataset leakage.

**Run both arms with masking on.** ~81% of the baseline's chosen tokens are tool
output; unmasked, the comparison would measure the masking rather than the
hypothesis.

### Environment and the trl 1.13 port

Training runs under `/usr/local/bin/python3` (NOT the miniconda env — several
version-specific checks were made against the wrong interpreter before this was
noticed):

```
trl 1.13.0   transformers 5.17.0   accelerate 1.15.0
torch 2.13.0+cu129   deepspeed 0.19.7   liger_kernel 0.8.3
8 GPUs, 178 GiB each
```

1.13 rewrote the DPO internals: `tokenize_row`, `concatenated_forward`,
`concatenated_inputs`, `dpo_loss`, `use_logits_to_keep`, `max_prompt_length` and
`max_completion_length` are all gone. `MaskedDPOTrainer` now needs only two
overrides, both verified end-to-end in the real environment:

* `_prepare_dataset` — tokenizes, gates EOS on `complete`, and emits
  `chosen_loss_mask` / `rejected_loss_mask` aligned 1:1 with the completion ids.
* `MaskedDataCollatorForPreference` — substitutes that mask for TRL's
  `[1] * len(chosen_ids)`, mirroring the parent's slicing and padding exactly.

**A real bug this port exposed — in stock TRL too.** Our prompts end at an
arbitrary character (the divergence point), so the last prompt token merges with
the first completion token in **50% of sides**:

```
prompt ends '>'  +  completion '\n<think>I…'   ->  one token '>\n<think>I'
```

TRL derives the completion as `tokenize(prompt+completion)[len(tokenize(prompt)):]`,
which skips the merged token — and since the collator rebuilds the sequence as
`prompt_ids + completion_ids`, that text is silently dropped. TRL warns and slices
anyway. `tokenize_pair` instead splits at the longest token prefix common to the
prompt and *both* full sequences, so concatenation is exact and the two completions
still share an identical prompt. Verified 0/480 mismatches, was ~50%.

**Config drift.** `transformers 5` removed `warmup_ratio` from `TrainingArguments`
(it is `warmup_steps` now). `run_dpo_training.py` filters its kwargs against the
installed `DPOConfig.__init__` signature and prints anything dropped, so a rename
degrades to a warning rather than a `TypeError`; `--warmup-ratio` is converted to a
step count locally.

### Training — `run_dpo_training.py`

```bash
python run_dpo_training.py --arm full --dry-run      # token + step budget, no GPU

# NOTE stage3_*_accelerate.conf are DEEPSPEED json configs despite the name, so they
# go to --deepspeed, NOT to `accelerate launch --config_file`.
torchrun --nproc_per_node 8 run_dpo_training.py --arm full \
    --model rl-research/DR-Tulu-SFT-8B \
    --deepspeed ../stage3_offloading_accelerate.conf --use-liger-kernel \
    --grad-accum 2 --epochs 2 --output-dir dpo_out_full_sft_b0.05_ep2

torchrun --nproc_per_node 8 run_dpo_training.py --arm local \
    --model rl-research/DR-Tulu-SFT-8B \
    --deepspeed ../stage3_offloading_accelerate.conf --use-liger-kernel \
    --grad-accum 8 --epochs 1 --output-dir dpo_out_local_sft_b0.05_ep1
```

**Pass `--model rl-research/DR-Tulu-SFT-8B`.** The script still defaults to
DR-Tulu-8B, but the pairs are SFT generations, so only the SFT checkpoint makes
`rejected` on-policy. The reference model follows `--model`. Recommended env:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; wandb must be logged in (or
`--no-wandb`). A Liger warning about the `_ascend` rope backend is harmless (Ascend
NPU code path).

`--arm` selects the dataset, `max_length` and the output dir (`dpo_out_<arm>/`).
Uses `MaskedDPOTrainer`, sets `remove_unused_columns=False`, asserts a fast
tokenizer, warns if prompts stop carrying the DR Tulu template, and prints a banner
of the memory-relevant settings plus the step budget before training.

Defaults: DR-Tulu-8B (override, see above), β=0.05, lr=5e-7, 2 epochs, batch 1 × grad-accum 8, bf16,
gradient checkpointing, `max_length=32768`, `attn_implementation=flash_attention_2`
(falls back to sdpa), `--eval-split test`, wandb project `dpo-critique-rewrite`,
checkpoints each epoch keeping 2.

**Reference model.** Under ZeRO-3, TRL cannot deep-copy the policy
(`create_reference_model` raises), so the script instantiates one explicitly and
passes `ref_model=`. `DPOConfig` is built **before** `from_pretrained` so
`HfTrainerDeepSpeedConfig` activates `deepspeed.zero.Init` and weights shard as they
load — otherwise each rank materialises a full 16 GB copy, twice over.
`prepare_deepspeed` then shards the reference at the same ZeRO stage, so it costs
~params/8 per GPU rather than 16 GB.

#### Memory — what actually binds

The constraint is the log-prob computation, not the models:

```
2 rows x 32,768 positions x 151,936 vocab x 4 B (fp32 log_softmax) = 37.1 GiB
```

That single tensor caused every OOM. Things that do **not** fix it: `--precompute-ref`
(removes only the reference's copy; also rejected with ZeRO-3 on 0.24, allowed on
1.13 but mutually exclusive with Liger), LoRA (leaves the tensor untouched, and is
out of scope per `small_synthetic_exp/mission.md`), multi-node (ZeRO-3 shards model
states across the world, but activations and logits are per-GPU), and larger batches
(Liger chunks over the batch dimension, and every chunk holds one chosen + one
rejected row, so the per-chunk peak is fixed).

What made it run: **`--use-liger-kernel`** plus the **offloading** DeepSpeed config,
on 178 GiB GPUs. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is worth setting.
A genuine fix for smaller cards would be a sequence-chunked loss (walk the sequence
in slices, accumulate `Σ_t log π(y_t)`, free as you go — ~5 GiB at 4k slices);
designed, not implemented.

#### Step budget — check it before launching

The baseline arm is only 616 pairs, so the effective batch matters enormously:

| grad_accum | effective batch | steps/epoch | 3 epochs |
|---|---|---|---|
| 8 (default) | 64 | 9 | 28 |
| **2** | **16** | **38** | **115** |

`--warmup-ratio` (0.05) is converted to steps locally and the banner warns when
warmup exceeds half the run. The first completed run used grad_accum 8 with a fixed
20-step warmup against ~20 total steps — **the learning rate ramped for the entire
run**, so its numbers should not be read as a result.

#### Runs

| output dir | init | arm | steps | eval loss / acc / margin | note |
|---|---|---|---|---|---|
| `dpo_out_full` | RL | full | 20 | 0.080 / 1.00 / 2.76 | LR warmup spanned the whole run — **not a result** |
| `dpo_out_full_b0.05_ep2` | RL | full | 78 (ga 2, 2 ep) | 0.000 / 1.00 / 13.17 | init/data mismatch |
| `dpo_out_local_b0.05_ep1` | RL | local | 114 (ga 8, 1 ep) | 0.288 / 0.95 / 2.17 | init/data mismatch |
| `dpo_out_full_sft_b0.05_ep2` | **SFT** | full | 78 | 0.000 / 1.00 / 13.66 | on-policy `rejected` |
| `dpo_out_local_sft_b0.05_ep1` | **SFT** | local | 114 | 0.290 / 0.95 / 1.99 | on-policy `rejected` |

The three RL-init runs trained DR-Tulu-8B on SFT-generated pairs, so `rejected` was
another model's text. Re-running from SFT barely changed the curves:

- **The full arm saturates** (loss ≈ 0, acc 1.00 by step ~30) from either init:
  whole luna sequences vs whole DR Tulu sequences are trivially separable.
- **Margins come from raising `chosen`, not lowering `rejected`.** Full/SFT ends at
  reward(chosen) +10.8 vs reward(rejected) −2.8 (β = 0.05, i.e. +217 vs −57 nats of
  log-ratio); local/SFT +1.8 vs −0.2 (+36 vs −4 nats).
- **`logps/chosen` < `logps/rejected` in absolute terms is expected** and not a
  sign of undertraining: the margin compares *changes* against the reference, and
  `chosen` is both longer and off-policy.

**The off-policy gap.** With an on-policy `rejected` (SFT init), per unmasked token
`chosen` ≈ −0.87 vs `rejected` ≈ −0.29 nats — `chosen` (luna's text) is ~3x less
likely than the model's own. The old "−0.97 vs −0.35, rejected is DR Tulu's own
text" figure was measured against the RL model and is superseded.

**How far the policy moved on its own samples** (`eval/onpolicy_shift.py`: base
model's own eval generations scored under base and under each trained policy,
prompt + tool output masked):

| policy | KL/token trace | KL/token answer | greedy token differs (answer) |
|---|---|---|---|
| full, SFT init | 0.015 | 0.016 | 2.1% |
| full, RL init | 0.013 | 0.015 | 2.3% |
| local, SFT init | 0.030 | **0.044** | **4.0%** |
| local, RL init | 0.038 | 0.053 | 4.7% |

Local moves the model's own text 2–3x more than full, concentrated in the answer,
from about half the gradient-bearing tokens — localization works at the level of
distribution shift. Init barely matters. Movement is not improvement, though: see
below.

### Inspecting and validating

```bash
python inspect_pairs.py --doc answer --n 8          # colourised single pair view
python inspect_pairs.py --doc trace --kind insert --masked-only --n 5
python inspect_pairs.py --record 0 --doc trace
python validate_pairs.py                            # re-derive every pair from source
```

`inspect_pairs.py` renders what flat JSONL hides: the divergence, the agreed tail
(dim), and masked spans collapsed to a placeholder — without which a 14k-char trace
insertion is unreadable. Filters: `--doc --kind --tag --record --complete --untagged
--masked-only`.

`validate_pairs.py` re-derives each pair from the source instead of trusting stored
fields, so any builder bug surfaces as a mismatch. Current state:

```
reconstruct · prefix · shared tail · non-degenerate · mask spans ·
complete flag · trace-never-complete · split leakage · arm alignment    ALL PASS
single locus                               22 / 7,586 differ in two regions (0.29%)
```

Those 22 are settings, not bugs: sub-edits merged by `--merge-gap 40`, and
are blocks over `_REFINE_LIMIT` (20,000) where word-level refinement is skipped and
the coarse line-level block is emitted whole.

**Known metadata bug (training unaffected).** `attribute()` (`build_dpo_pairs.py:399`)
labels a pair with a critique only if the change lies inside that critique's
`edit_span`, but Step 1b inserts the search round after the round's
`</tool_output>` — usually outside the span. So 589 of the 598 untagged local trace
pairs are search insertions, and only 44% of search critiques appear in any pair's
`critique_tags`. Counting them, search is ~15% of local pairs, matching the full
arm. Fix (deferred): attribute inserted rounds by matching the `<call_tool>` query to
the critique's `s2_search_queries`. `eval/score_eval.py` works around it.

## Behavioural eval (`eval/`)

All on the 31 test questions, 3 samples each at temperature 0.7, through the real
DR Tulu tool-calling workflow.

```bash
# 1. generate (one GPU; needs S2_API_KEY). Serves each checkpoint as "policy".
python -u eval/gen_eval_answers.py --models base_sft=rl-research/DR-Tulu-SFT-8B \
    local_sft_b0.05_ep1=$PWD/dpo_out_local_sft_b0.05_ep1
# 2. pairwise judge (needs OPENAI_API_KEY) — the primary metric
python -u eval/pairwise_judge.py --pairs base_sft:local_sft_b0.05_ep1
# 3. critique-based stats (secondary; see why below)
python -u eval/score_eval.py --models base_sft local_sft_b0.05_ep1
```

`gen_eval_answers.py` imports `AutoReasonSearchWorkflow` without modifying `dr-tulu`,
pointed at `eval/drtulu_eval.yaml` (a copy of `auto_search_sft_s2_only_hamish.yaml`
with `search_agent_model_name: "policy"`); it refuses to start if port 30001 is
already serving a model. The "Unclosed client session" aiohttp warnings it prints
come from litellm and are harmless. `make_reference_generations.py` exports the test
pairs as two pseudo-models, `original` (= `rejected`) and `rewrite_luna`
(= `chosen`), asserted byte-identical to the stored full-arm pairs. Output files are
resumable; when piping to `tee`, `mkdir -p` the log's directory first.

### The critic is not a usable metric

The first eval counted `gpt-5.4` critiques (same prompt as training). It failed:

- **Ceiling check:** `rewrite_luna` gets 10.26 critiques per answer vs 10.03 for its
  own `original` — the critic emits ~10 critiques whatever the quality.
- **Test-retest** (`critic_retest.py`, same answer critiqued twice): only **55%** of
  flagged spans recur; total-count r = 0.56; the "search" category r = 0.05 (noise).
  Critic noise is as large as the difference between two samples of a question.
- **One category does carry signal:** "unsupported" (overclaiming, unsupported
  inference) — rewrite 1.65 vs original 2.29 per answer, −28% [−1.13, −0.13], with
  retest r = 0.62. Elsewhere the critic finds *new* problems in the rewrites
  (organization +64%, detail +125%): it re-spends a fixed budget.

Tags are free text (6,972 distinct over 10,720 training critiques), grouped into 10
categories by `critique_categories.py` (keyword rules, 87% coverage).

### Pairwise judge — the primary metric

`pairwise_judge.py`: the team's standard pairwise prompt (A / B / Tie / Both Bad),
`gpt-5.4`, both orders, answers only (no trace, so it cannot check that a snippet
supports its claim). Score = B wins 1, A wins 0, tie 0.5, averaged per question,
bootstrapped over questions.

| B vs A | B preference [95% CI] | questions B better/tie/worse | order consistency | first-shown wins |
|---|---|---|---|---|
| **rewrite_luna vs original** | **0.887** [0.782, 0.968] | 27 / 2 / 2 | 90% | 45% |
| full_sft vs base_sft | 0.457 [0.376, 0.532] | 10 / 6 / 15 | 73% | 56% |
| local_sft vs base_sft | 0.425 [0.333, 0.511] | 9 / 8 / 14 | 75% | 58% |

Validation: repeat agreement 97% on identical prompts; answer length does not drive
verdicts (longer answer wins 42–55% of decisive judgments). Caveat: judge and
rewriter are both GPT-5-family, so some of the 0.887 may be style self-preference.

### What the trained models did and did not learn

| | base_sft | full_sft | local_sft | rewrite_luna |
|---|---|---|---|---|
| judge vs base_sft | — | 0.457 | 0.425 | (0.887 vs original) |
| "unsupported" critiques / answer | 2.34 | 2.35 | 2.33 | 1.65 |
| rounds re-opening `<think>` | 43% | 47% | **69%** | 71% |
| answer chars | 10,920 | 10,363 | **9,432** | 9,422 |

- **Content did not transfer.** Neither arm moves "unsupported" or the judge.
- **The local arm learns what is decided at the divergence point.** 257 local trace
  pairs diverge right after `</tool_output>` with `chosen` opening `<think>` and
  `rejected` bare (none the other way); local adopts the style (43% → 69%), full
  does not (47%). It also adopts the training data's answer length. Both are
  rewrite-pipeline signals, not critique fixes.
- The RL-init arms behaved the same way (no gain; critique count +0.5/answer purely
  from longer answers).

## Key files

| File | Role |
|---|---|
| `pg_dr_tulu.py` | Stage 1–2: DR Tulu answer generation + `gpt-5.4` critique generation. `ANSWER_ONLY=1` does generation only. |
| `rewrite_answer_from_critiques.py` | Stage 3: the rewrite pipeline (tag → splice → programmatic search → answer rewrite). |
| `build_dpo_pairs.py` | Stage 4: builds the DPO preference pairs. `--mode local` (per-change) or `--mode full` (whole-sample baseline). |
| `dpo_trainer_utils.py` | `MaskedDPOTrainer` (trl **1.13**): EOS only when `complete`, tool-output spans excluded from the loss, `tokenize_pair` fixing the prompt/completion token seam. Plus `load_pairs()`. Warns if the installed trl differs from 1.13.0. |
| `run_dpo_training.py` | Stage 5: DPO training, `--arm local` or `--arm full`. `--dry-run` prints the token budget. |
| `inspect_pairs.py` | Terminal viewer for individual pairs (divergence, agreed tail, masked spans). |
| `validate_pairs.py` | Re-derives every pair from source and checks all invariants. Run after each build. |
| `repair_citations.py` | One-off: repaired 37 resolvable dangling `<cite>` ids into `…_mod_fixed.jsonl`. |
| `drtulu_prompt_template.txt` | The exact DR Tulu inference prompt, `{question}` placeholder. |
| `generate_viewer.py` | Builds `trace_viewer.html` — side-by-side model-comparison viewer (old vs rewritten trace/answer, critiques, word-diff, inserted-round highlighting). |
| `eval/gen_eval_answers.py` | Generates eval traces per checkpoint through the DR Tulu workflow (vLLM + MCP), `eval/generations/<model>.jsonl`. |
| `eval/drtulu_eval.yaml` | Copy of DR Tulu's `auto_search_sft_s2_only_hamish.yaml` with model name `policy`. |
| `eval/make_reference_generations.py` | Exports the test pairs as `original` / `rewrite_luna` pseudo-models. |
| `eval/pairwise_judge.py` | **Primary metric**: pairwise `gpt-5.4` judge, both orders, `eval/pairwise/`. |
| `eval/score_eval.py` | Critique-based stats (counts, categories, structure, dose-response), `eval/critiques/`. |
| `eval/critique_categories.py` | Free-text critique tags → 10 categories. |
| `eval/critic_retest.py` | Critic test-retest reliability. |
| `eval/onpolicy_shift.py` | KL / greedy-token shift of each policy on the base model's own samples. |
| `samples_1000/` | 1000-record inputs and rewritten outputs (~100–200 MB JSONL). |
| `test_samples/` | Small (2–10 record) dev/validation samples. |
| `RESEARCH_CHALLENGE.md` | Write-up submitted to `allenai/asta-research-challenge` (PR #12). |

## Open items

- Step-1 latency: the in-place trace edit asks the model to reproduce the full
  trace (~17–40k tokens) to make small edits — slow on long traces. A per-span
  in-place rewrite was proposed but **not** implemented.
- Cost accounting under-counts records that fail/skip *after* LLM calls (proposed
  incremental fix, not implemented).
- **Next experiment — SFT on the rewrites as a control.** Fine-tune DR-Tulu-SFT-8B
  directly on `rewrite_luna` sequences (tool output masked) and judge it against
  base_sft. If it wins, the quality is transferable and plain DPO is the
  bottleneck (then try DPO + an NLL term on `chosen`, or SFT → DPO, still local vs
  full). If it does not, ~600 records is too little to transfer it.
- **Judge with evidence.** Append each answer's cited snippets so the judge can
  check support — the one place the critic sees the rewrite improve. Re-validate on
  `original:rewrite_luna` first.
- **Cross-family judge.** Re-run `original:rewrite_luna` with a non-GPT judge to bound
  style self-preference before trusting any arm win.
- **Strip pipeline artifacts from pairs.** The 257 `<think>`-only trace pairs carry
  no critique content but are what the local arm learns most readily.
- **Search-round tag attribution** (see Stage 4) — metadata only.
- **test is 31 records.** Effective sample size is governed by records, not pairs
  (~11.7 pairs per record), so per-arm eval will be noisy — an effect must be large
  to clear it. Relaxing `--max-sequence-tokens` to 40,960 would give 804 records
  instead of 647 at the cost of using RoPE positions above Qwen3's trained range.
- 40 introduced dangling citations remain unrepairable (ambiguous truncations,
  out-of-range indices); their records are dropped by the gate rather than patched.
- Masking the **agreed tail** as well (H7 on the answer pairs) — the machinery
  exists, it is a second span list; not implemented, masking was scoped to tool output.
- **Sequence-chunked DPO loss.** The 37 GiB fp32 log-softmax tensor is the only
  thing standing between this and an 80 GiB card; chunking the sequence would cut it
  to ~5 GiB. Designed, not implemented — it is custom numerics in the loss path and
  would need validating against the stock loss first.
- **The off-policy gap.** `chosen` is ~3x less likely per token than an on-policy
  `rejected` (−0.87 vs −0.29), because luna wrote it. The leading explanation for
  why content does not transfer; on-policy rewrites (the policy applying the
  critiques itself) would remove it.
- `run_dpo_training.py` still defaults `--model` to DR-Tulu-8B; consider changing
  the default to the SFT checkpoint so the mismatch cannot recur.
- `--question-template` defaults to the checked-in `drtulu_prompt_template.txt`;
  re-render it if the DR Tulu prompts or `prompt_version` default change.
