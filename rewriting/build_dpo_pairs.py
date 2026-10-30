"""Build localized DPO preference pairs from critique-guided rewrites.

Input is the Stage-3 output of `rewrite_answer_from_critiques.py`, one JSON
object per record with:

    question         the scientific question
    original_trace   full DR Tulu trace, INCLUDING the <answer>…</answer> block
    rewritten_trace  trace after Step 1a/1b, WITHOUT the answer block
    original_answer  the <answer>…</answer> block extracted from original_trace
    rewritten        the rewritten <answer>…</answer> block
    critique         critique JSON string (used only for provenance labels)

For every *change* between original and rewritten we emit one preference pair
whose `chosen` and `rejected` share an identical prefix and diverge at exactly
that change:

    prompt   = <DR Tulu prompt> + REWRITTEN text up to the divergence point
    rejected = original segment   + agreed tail
    chosen   = rewritten segment  + agreed tail

The prefix is the *rewritten* text, so every earlier fix is already applied and
the pair reads: having done everything right up to here, prefer the corrected
continuation over the original one. (`--trace-prefix`/`--answer-prefix original`
switches back to conditioning on the unrewritten text.)

The **agreed tail** runs from the point where the two versions reconverge to the
end of the sentence it completes (`common_tail`; closing tags and newlines count
as terminators, since trace text is not uniformly prose). It exists so pure
insertions and deletions have a non-empty completion on both sides, and so the
pair reads as a real continuation rather than a bare span. Stopping at a sentence
boundary keeps it short — every tail token is loss mass on text that carries no
preference (`small_synthetic_exp/research_state.md` H7).

Answer pairs additionally carry the whole REWRITTEN TRACE in the prefix. That is
required, not a preference: Step 3 of the rewriter is called with
`trace=rewritten_trace` (`step3_prompt`), so the rewritten answer cites snippet
IDs that exist only in rounds inserted at Step 1b; an original-trace prefix would
make `chosen` cite evidence absent from its own context.

REWRITTEN-SIDE GATE
-------------------
A record is skipped entirely when its REWRITTEN output has a format or citation
defect, because every pair from that record puts that output in `chosen` — the
training target. This is ABSOLUTE, not a delta against the source: a malformed
`chosen` teaches malformed output whether the rewriter caused the defect or
inherited it from a bad source trace. Checks: empty trace, `validate_trace_structure`
problems, leaked critique-prompt text, unfilled `PLACEHOLDER_*`, missing
`<answer>` wrapper, unterminated `<cite id="`, and any `<cite>` id with no matching
`<snippet id=>` in the trace.

The ORIGINAL is deliberately never gated. `rejected` is the dispreferred side, so
its flaws are what the preference points away from — DR Tulu's own answers carry
953 dangling citations, and the rewrite removing 332 of them is real signal.
Disable with `--no-gate-rewritten`.

TOOL OUTPUT
-----------
An inserted search round contains the real `<tool_output>` snippets it retrieved,
and the whole round is one diff block, so those snippets land in `chosen` (~90% of
trace-pair completion tokens). The policy never emits them — `</call_tool>` is a
hard stop sequence and the harness splices the results in — and both DR Tulu
training stages exclude them (SFT `mask_span_types: "tool_output"`, RL
`mask_tool_use=True`). Each pair therefore carries `chosen_mask_spans` /
`rejected_mask_spans`: character spans to drop from the DPO log-prob sum. The
tokens stay in the sequence and are still attended to. Apply them by training with
`MaskedDPOTrainer` from `dpo_trainer_utils.py`.

END-OF-TEXT
-----------
Each pair carries a `complete` flag: True when its completion runs to the true
end of the sequence (`</answer>`), False when it stops mid-sequence. TRL's
`DPOTrainer.tokenize_row` appends EOS to *every* completion unconditionally,
which would teach the model to stop in the middle of an answer. Honor the flag
one of two ways:

  1. Train with `MaskedDPOTrainer` from `dpo_trainer_utils.py` (recommended) — it
     appends EOS only when `complete` is True, and applies the mask spans above.
     Requires `remove_unused_columns=False` so the columns reach tokenize_row.
  2. Bake the EOS string in here with `--eos-token '<|im_end|>'` (or
     `--tokenizer <hf-id>` to resolve it automatically). You STILL need the
     trainer subclass, otherwise TRL adds a second EOS to every completion.

CONTEXT MATCH
-------------
The prefix must reproduce the prompt DR Tulu actually generated against, or the
policy is trained in a context it never sees at inference — the single biggest
confound in the list-marker experiments. `drtulu_prompt_template.txt` is that
prompt, rendered byte-exactly and used by default:

    dr-tulu/agent/workflows/query_direct.py  POST /ask -> AutoReasonSearchWorkflow
      configuration auto_search_sft_s2_only_hamish.yaml (no prompt_version ->
      SearchAgent default "v20250907"), dataset_name "sqav2" -> long_form
    -> messages = [system: unified_tool_calling_v20250907.yaml system_prompt,
                   user:   question + "\n\n" + additional_instructions.long_form]
    -> LLMToolClient._messages_to_prompt: Qwen/Qwen3-8B apply_chat_template(
           tokenize=False, add_generation_prompt=True)

which ends `<|im_end|>\n<|im_start|>assistant\n`; the trace then opens with
`<think>`. `full_traces.generated_text` is exactly the assistant continuation
(assistant content + formatted `<tool_output>` blocks, client.py:1070-1131), and
because the trace contains `<answer>`, the workflow returns the SearchAgent's own
trace rather than a separate AnswerAgent call (auto_search_sft.py:693) — so
trace + answer is one continuous generation from this prompt.

Regenerate with `--regen-prompt-template <path-to-dr-tulu/agent>` if the DR Tulu
prompts change. EOS for this stack is `<|im_end|>` (Qwen3-8B, id 151645).

Usage
-----
    python build_dpo_pairs.py                       # local arm, writes 3 splits
    python build_dpo_pairs.py --mode full           # whole-sample baseline arm
    python build_dpo_pairs.py --merge-gap 80 --no-mask-tool-output
    python build_dpo_pairs.py --max-prefix-chars 24000 --tokenizer Qwen/Qwen3-8B
    python build_dpo_pairs.py --regen-prompt-template ../../dr-tulu/agent
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import random
import re
import sys
from collections import Counter
from difflib import SequenceMatcher

DEFAULT_INPUT = "samples_1000/drtulu_answers_w_critiques_rewritten_gpt-5.6-luna_v3_mod_fixed.jsonl"
FALLBACK_QUESTION_TEMPLATE = "Question: {question}\n\n"
PROMPT_TEMPLATE_FILE = "drtulu_prompt_template.txt"
HELPER_SOURCE = "rewrite_answer_from_critiques.py"


# ── borrow helpers from the rewriter without running it ───────────────────────
# rewrite_answer_from_critiques.py does real work at import time (loads a 111 MB
# JSONL, constructs API clients), so we lift just the pure functions we need out
# of its AST. Extracting rather than copying keeps span attribution in lockstep
# with the logic that actually produced the rewrites.
def _load_rewriter_helpers(path: str) -> dict:
    wanted = {
        "_norm_span",
        "compute_edit_spans",
        "parse_critique",
        "extract_answer_block",
        "validate_trace_structure",
    }
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    missing = wanted - {n.name for n in defs}
    if missing:
        sys.exit(f"{path}: expected function(s) not found: {sorted(missing)}")
    ns: dict = {"re": re, "json": json}
    exec(compile(ast.Module(body=defs, type_ignores=[]), "<rewriter-helpers>", "exec"), ns)
    return ns


# ── diffing ───────────────────────────────────────────────────────────────────
_WORD = re.compile(r"\s+|\w+|[^\w\s]")
_REFINE_LIMIT = 20_000  # don't word-diff a block this large; it's a bulk insert


def _offsets(seq) -> list[int]:
    return [0] + list(itertools.accumulate(len(t) for t in seq))


def char_blocks(a: str, b: str) -> list[tuple[int, int, int, int]]:
    """Character-level change blocks between `a` and `b`, as (a0, a1, b0, b1).

    Two-level diff: lines first (a 100 kB trace is only ~350 lines, so this is
    fast), then words inside each changed line-block to tighten the spans. A
    single-level word diff on the whole document is O(n^2) on 30k tokens and
    takes minutes per record.
    """
    A, B = a.splitlines(keepends=True), b.splitlines(keepends=True)
    oa, ob = _offsets(A), _offsets(B)
    out: list[tuple[int, int, int, int]] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        sa, ea, sb, eb = oa[i1], oa[i2], ob[j1], ob[j2]
        if (ea - sa) + (eb - sb) > _REFINE_LIMIT:
            out.append((sa, ea, sb, eb))
            continue
        WA, WB = _WORD.findall(a[sa:ea]), _WORD.findall(b[sb:eb])
        wa, wb = _offsets(WA), _offsets(WB)
        for tag2, k1, k2, l1, l2 in SequenceMatcher(None, WA, WB, autojunk=False).get_opcodes():
            if tag2 == "equal":
                continue
            out.append((sa + wa[k1], sa + wa[k2], sb + wb[l1], sb + wb[l2]))
    return out


def merge_blocks(blocks, gap: int):
    """Coalesce blocks separated by fewer than `gap` identical characters.

    Without this, one reworded sentence shatters into a dozen word-level pairs
    that each teach a fragment of the same edit.
    """
    merged: list[list[int]] = []
    for s1, e1, s2, e2 in blocks:
        if merged and (s1 - merged[-1][1]) < gap and (s2 - merged[-1][3]) < gap:
            merged[-1][1], merged[-1][3] = e1, e2
        else:
            merged.append([s1, e1, s2, e2])
    return [tuple(m) for m in merged]


# End of the sentence (or structural block) in which the two versions have
# reconverged. Trace text is not uniformly prose, so closing tags and line breaks
# count as terminators alongside sentence punctuation.
_SENT_END = re.compile(
    r"""[.!?][)"'\]]*(?=\s|$)          # sentence punctuation, incl. trailing quote/bracket
      | </think>|</answer>|</tool_output>|</call_tool>|</snippet>
      | \n""",
    re.VERBOSE,
)


def common_tail(common: str, args) -> str:
    """Take the agreed text up to the end of the sentence it completes.

    `common` is the text after the change, identical in both versions. Searching
    from offset 1 guarantees a non-empty tail even when the change ends exactly on
    a boundary — otherwise a pure insertion would leave `rejected` empty.
    """
    if args.tail_to_end:
        return common
    # Walk terminators until the tail is long enough to be a real agreed
    # continuation. With min_tail_chars=0 this stops at the first one, which is
    # often 2 chars ('.\n') when the change ends just before the sentence-final
    # punctuation — maximal signal density, minimal shared suffix.
    pos = 1
    while True:
        m = _SENT_END.search(common, pos)
        if m is None or m.end() > args.tail_max_chars:
            break
        if m.end() >= args.min_tail_chars:
            return common[:m.end()]
        pos = m.end() + 1
    return common[:args.tail_max_chars]


def trainable_len(text: str, args) -> int:
    """Characters that will actually carry gradient — raw length minus masked spans."""
    if not args.mask_tool_output:
        return len(text)
    return len(text) - sum(e - s for s, e in tool_output_spans(text))


_TOOL_OPEN = re.compile(r"<tool_output>")
_TOOL_CLOSE = re.compile(r"</tool_output>")


def tool_output_spans(text: str) -> list[list[int]]:
    """Character spans of <tool_output>…</tool_output> within `text`, tags included.

    Two ragged cases are handled, since a change block can start or end inside a
    tool output: an unclosed block runs to end-of-string, and a leading orphan
    '</snippet></tool_output>' is masked from position 0.
    """
    spans, pos = [], 0
    # A change block can begin *inside* the preceding tool output, so the segment
    # opens with an orphan '</snippet></tool_output>'. Those tags are environment
    # text the policy never emits, so mask them too.
    orphan = _TOOL_CLOSE.search(text)
    if orphan and (_TOOL_OPEN.search(text) is None
                   or orphan.start() < _TOOL_OPEN.search(text).start()):
        spans.append([0, orphan.end()])
        pos = orphan.end()
    while True:
        o = _TOOL_OPEN.search(text, pos)
        if o is None:
            return spans
        c = _TOOL_CLOSE.search(text, o.end())
        end = c.end() if c else len(text)
        spans.append([o.start(), end])
        pos = end


_ANSWER_TAG = re.compile(r"</?answer>")


def strip_stray_answer_tags(text: str) -> str:
    """Keep only the first <answer> and the last </answer>; drop any others.

    The rewrite model sometimes emits a premature `</answer>` mid-answer (85 records,
    typically ~5% of the way in, right after a heading) and then carries on writing.
    The original answers never do this, so it is purely rewrite noise — but
    `</answer>` is where generation STOPS at inference, so a `chosen` containing one
    teaches the model to truncate its own answer. Also handles the mirror case of a
    stray second `<answer>`.
    """
    tags = list(_ANSWER_TAG.finditer(text))
    if len(tags) <= 2:
        return text
    first_open = next((m for m in tags if m.group(0) == "<answer>"), None)
    last_close = next((m for m in reversed(tags) if m.group(0) == "</answer>"), None)
    keep = {m.span() for m in (first_open, last_close) if m is not None}
    out, prev = [], 0
    for m in tags:
        if m.span() in keep:
            continue
        out.append(text[prev:m.start()])
        prev = m.end()
    out.append(text[prev:])
    return "".join(out)


def oversize_records(records, template, max_tokens, tokenizer_name):
    """Record indices whose full sequence cannot fit `max_tokens` for the target model.

    Applied identically to both arms so they stay matched. The bound is
    `prompt_header + max(|original sequence|, |rewritten sequence|)`: every pair's
    prompt+completion is a slice of one of those two documents preceded by the
    header, so a record that fits this bound fits in either arm.

    Qwen3-8B reports max_position_embeddings 40,960 but was natively trained to
    32,768; positions above that are RoPE extrapolation. Filtering here rather than
    truncating at training time keeps every surviving pair intact — truncation would
    be one-sided (the rewritten side is ~51% longer) and would eat the `<answer>`
    block, which sits at the end of the sequence.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    n_tok = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
    head_tokens = n_tok(template.replace("{question}", "")) if template else 0
    bad = set()
    for i, rec in enumerate(records):
        m = re.search(r"<answer>", rec["original_trace"])
        if not m:
            continue
        pre = rec["original_trace"][:m.start()].rstrip()
        joiner = rec["original_trace"][len(pre):m.start()] or "\n"
        original = pre + joiner + rec["original_answer"]
        rewritten = rec["rewritten_trace"] + joiner + rec["rewritten"]
        need = head_tokens + n_tok(rec["question"]) + max(n_tok(original), n_tok(rewritten))
        if need > max_tokens:
            bad.add(i)
    return bad


# ── rewritten-side quality gates ──────────────────────────────────────────────
# `chosen` is the training target, so a defect there teaches the model to produce
# it — regardless of whether the rewriter caused the defect or inherited it from a
# malformed source trace. These gates are therefore ABSOLUTE on the rewritten side.
# The original is deliberately NOT gated: `rejected` is the dispreferred side, and
# its flaws are exactly what the preference is supposed to point away from.
_SNIPPET_ID = re.compile(r"<snippet id=([^\s>]+?)>")
_CITE_ANY = re.compile(r"<cite\s+ids?=")
# A '<' or '>' inside the value means the quote never closed. Whitespace is fine —
# `id="a, b"` is a legitimate multi-id citation.
_CITE_OK = re.compile(r'<cite\s+ids?="([^"<>]*)"\s*>')
_PLACEHOLDER = re.compile(r"PLACEHOLDER_[A-Za-z0-9]+")
_PROMPT_LEAK = re.compile(r"Critiques to fix \(edit only the flagged spans\)")


def cited_ids(text: str) -> set:
    out = set()
    for m in _CITE_OK.finditer(text):
        out |= {x.strip() for x in m.group(1).split(",") if x.strip()}
    return out


def rewritten_defects(rec, rewritten_trace, rewritten_answer, validate) -> list[str]:
    """Reasons the rewritten output is unusable as `chosen`. Empty list = clean."""
    bad = []
    if not rewritten_trace.strip():
        bad.append("empty_trace")
    else:
        if validate(rewritten_trace):
            bad.append("trace_structure")
        if _PROMPT_LEAK.search(rewritten_trace):
            bad.append("prompt_leak")
        if _PLACEHOLDER.search(rewritten_trace) or _PLACEHOLDER.search(rewritten_answer):
            bad.append("placeholder")
    if not (rewritten_answer.lstrip().startswith("<answer>")
            and rewritten_answer.strip().endswith("</answer>")):
        bad.append("answer_wrapper")
    if len(_CITE_ANY.findall(rewritten_answer)) != len(_CITE_OK.findall(rewritten_answer)):
        bad.append("malformed_cite_tag")
    known = set(_SNIPPET_ID.findall(rewritten_trace)) | set(_SNIPPET_ID.findall(rec["original_trace"]))
    if cited_ids(rewritten_answer) - known:
        bad.append("dangling_cite")
    return bad


# ── pair construction ─────────────────────────────────────────────────────────
def classify(orig_seg: str, rew_seg: str) -> str:
    if not orig_seg.strip():
        return "insert"
    if not rew_seg.strip():
        return "delete"
    return "replace"


def attribute(spans, s1: int, e1: int, labels) -> list[str]:
    """Critique tags whose edit span covers this change (original coordinates)."""
    hits = []
    for (ss, se), lab in zip(spans, labels):
        if s1 == e1:
            if ss <= s1 <= se:
                hits.append(lab)
        elif ss < e1 and s1 < se:
            hits.append(lab)
    return hits


def build_pairs_for_doc(orig: str, rew: str, prefix_head: str, args, *,
                        doc: str, prefix_mode: str, spans, labels, doc_is_terminal: bool):
    """Emit one pair per change between `orig` and `rew`.

    `prefix_head` is everything that precedes this document in the sequence
    (question header, plus the rewritten trace for answer docs).
    `doc_is_terminal` marks the document that ends the sequence, so a tail that
    reaches its end can be flagged `complete`.
    """
    blocks = merge_blocks(char_blocks(orig, rew), args.merge_gap)
    pairs = []
    for idx, (s1, e1, s2, e2) in enumerate(blocks):
        orig_seg, rew_seg = orig[s1:e1], rew[s2:e2]
        if orig_seg == rew_seg:
            continue
        if not (orig_seg + rew_seg).strip():
            continue  # whitespace-only churn
        # Measure the change by its TRAINABLE content, not raw length: a multi-round
        # search insertion is ~90% <tool_output>, which is masked out of the loss, so
        # a raw-character cap silently drops pairs whose real content is small.
        size = max(trainable_len(orig_seg, args), trainable_len(rew_seg, args))
        if size < args.min_change_chars or size > args.max_change_chars:
            continue

        # Text after the change, clipped at the next change. Identical in both
        # versions by construction — verify, since a bad diff here would silently
        # make the pair differ in two places.
        nxt1 = blocks[idx + 1][0] if idx + 1 < len(blocks) else len(orig)
        nxt2 = blocks[idx + 1][2] if idx + 1 < len(blocks) else len(rew)
        common_o, common_r = orig[e1:nxt1], rew[e2:nxt2]
        if common_o != common_r:
            continue
        common = common_o

        tail = common_tail(common, args)
        if not tail and not orig_seg.strip():
            continue  # pure insertion with no tail -> empty `rejected`
        if not tail and not rew_seg.strip():
            continue  # pure deletion with no tail -> empty `chosen`

        reached_doc_end = (e1 + len(tail) == len(orig)) and (e2 + len(tail) == len(rew))
        complete = bool(doc_is_terminal and reached_doc_end)

        context = (rew[:s2] if prefix_mode == "rewritten" else orig[:s1])
        prefix = prefix_head + context
        if args.max_prefix_chars and len(prefix) > args.max_prefix_chars:
            # Left-truncate, then snap forward to a line start so the prompt does
            # not begin mid-token/mid-tag.
            cut = len(prefix) - args.max_prefix_chars
            nl = prefix.find("\n", cut)
            prefix = prefix[nl + 1:] if nl != -1 else prefix[cut:]

        chosen, rejected = rew_seg + tail, orig_seg + tail
        # Retrieved evidence the policy never emits: masked out of the DPO
        # log-prob sum at training time (see dpo_trainer_utils.MaskedDPOTrainer).
        chosen_mask = tool_output_spans(chosen) if args.mask_tool_output else []
        rejected_mask = tool_output_spans(rejected) if args.mask_tool_output else []
        if args.eos_token and (args.eos_policy == "always" or
                               (args.eos_policy == "complete_only" and complete)):
            chosen += args.eos_token
            rejected += args.eos_token

        pairs.append({
            "prompt": prefix,
            "chosen": chosen,
            "rejected": rejected,
            "complete": complete,
            "chosen_mask_spans": chosen_mask,
            "rejected_mask_spans": rejected_mask,
            "doc": doc,
            "change_idx": idx,
            "kind": classify(orig_seg, rew_seg),
            "critique_tags": attribute(spans, s1, e1, labels),
            "orig_span": [s1, e1],
            "rew_span": [s2, e2],
            "prefix_chars": len(prefix),
            "chosen_chars": len(chosen),
            "rejected_chars": len(rejected),
            "tail_chars": len(tail),
        })
    return pairs


def build_full_pair(rec, orig_pre, joiner, args, *, labels, trace_ok, answer_ok):
    """Baseline: one pair per record over the WHOLE sample.

    prompt   = the DR Tulu prompt alone (question, no generated text)
    rejected = the full original sequence   (trace + answer)
    chosen   = the full rewritten sequence  (rewritten trace + rewritten answer)

    This is the global-rewrite control the localized-edit hypothesis is measured
    against: identical records, identical split, identical prompt — the only
    difference is that chosen/rejected diverge everywhere instead of at one span.
    Both sides run to `</answer>`, so both are `complete` and take an EOS.
    """
    if not (trace_ok and answer_ok):
        return None, "doc_rejected"
    original = orig_pre + joiner + rec["original_answer"]
    rewritten = rec["rewritten_trace"] + joiner + rec["rewritten"]
    if not original.strip() or not rewritten.strip() or original == rewritten:
        return None, "degenerate"

    head = args.question_template.replace("{question}", rec["question"].strip()) \
        if args.question_template else ""
    prefix = head
    if args.max_prefix_chars and len(prefix) > args.max_prefix_chars:
        cut = len(prefix) - args.max_prefix_chars
        nl = prefix.find("\n", cut)
        prefix = prefix[nl + 1:] if nl != -1 else prefix[cut:]

    complete = (original.rstrip().endswith("</answer>")
                and rewritten.rstrip().endswith("</answer>"))
    chosen, rejected = rewritten, original
    if args.eos_token and (args.eos_policy == "always"
                           or (args.eos_policy == "complete_only" and complete)):
        chosen += args.eos_token
        rejected += args.eos_token

    return {
        "prompt": prefix,
        "chosen": chosen,
        "rejected": rejected,
        "complete": complete,
        "chosen_mask_spans": tool_output_spans(chosen) if args.mask_tool_output else [],
        "rejected_mask_spans": tool_output_spans(rejected) if args.mask_tool_output else [],
        "doc": "full",
        "change_idx": 0,
        "kind": "full",
        "critique_tags": sorted(set(labels)),
        "orig_span": [0, len(original)],
        "rew_span": [0, len(rewritten)],
        "prefix_chars": len(prefix),
        "chosen_chars": len(chosen),
        "rejected_chars": len(rejected),
        "tail_chars": 0,
    }, None


def process_record(rec, idx, H, args):
    """Return (pairs, skip_reason_or_None) for one input record."""
    question = rec["question"].strip()
    orig_trace = rec["original_trace"]
    rew_trace = rec["rewritten_trace"]
    orig_ans = rec["original_answer"]
    rew_ans = rec["rewritten"]

    m = re.search(r"<answer>", orig_trace)
    if not m:
        return [], "no_answer_tag"
    orig_pre = orig_trace[:m.start()].rstrip()
    joiner = orig_trace[len(orig_pre):m.start()] or "\n"

    if not orig_pre or not rew_trace:
        return [], "empty_trace"
    if len(rew_trace) < args.min_trace_ratio * len(orig_pre):
        return [], "trace_truncated"

    # One gate for the whole record: any defect in the rewritten output disqualifies
    # it, because every pair from this record puts that output in `chosen`.
    if args.gate_rewritten:
        defects = rewritten_defects(rec, rew_trace, rew_ans, H["validate_trace_structure"])
        if defects:
            return [], "reject:" + "+".join(defects)
    answer_pairs_allowed = trace_pairs_allowed = True

    critiques = H["parse_critique"](rec.get("critique", "[]"))
    if critiques is None:
        all_crit = []
    else:
        all_crit = critiques if isinstance(critiques, list) else critiques.get("local", [])
    plan_crit = [c for c in all_crit if c.get("location") in ("plan", "both")]
    ans_crit = [c for c in all_crit if c.get("location") in ("answer", "both")]

    def spans_and_labels(text, crits):
        # compute_edit_spans merges overlapping critiques, so re-derive per
        # critique to keep a label attached to each span.
        spans, labels = [], []
        for c in crits:
            for s, e in H["compute_edit_spans"](text, [c]):
                spans.append((s, e))
                labels.append(c.get("tag") or (c.get("issue", "")[:60]))
        return spans, labels

    # str.replace, not str.format: the DR Tulu system prompt contains literal
    # braces (\boxed{}, \boxed{November 16, 2001}) that .format would choke on.
    if args.mode == "full":
        pair, why = build_full_pair(rec, orig_pre, joiner, args,
                                    labels=[c.get("tag") or "" for c in all_crit],
                                    trace_ok=trace_pairs_allowed,
                                    answer_ok=answer_pairs_allowed)
        if pair is None:
            return [], why
        pair["record_idx"] = idx
        pair["question"] = question
        return [pair], None

    head = args.question_template.replace("{question}", question) if args.question_template else ""
    pairs = []

    if trace_pairs_allowed:
        s, l = spans_and_labels(orig_pre, plan_crit)
        pairs += build_pairs_for_doc(
            orig_pre, rew_trace, head, args,
            doc="trace", prefix_mode=args.trace_prefix, spans=s, labels=l,
            doc_is_terminal=False,   # the sequence continues into <answer>
        )

    if answer_pairs_allowed:
        s, l = spans_and_labels(orig_ans, ans_crit)
        answer_head = head + rew_trace + joiner
        pairs += build_pairs_for_doc(
            orig_ans, rew_ans, answer_head, args,
            doc="answer", prefix_mode=args.answer_prefix, spans=s, labels=l,
            doc_is_terminal=True,
        )

    if args.max_pairs_per_record and len(pairs) > args.max_pairs_per_record:
        pairs = pairs[:args.max_pairs_per_record]

    for p in pairs:
        p["record_idx"] = idx
        p["question"] = question
    skip = None
    if not trace_pairs_allowed and not answer_pairs_allowed:
        skip = "both_docs_rejected"
    return pairs, skip


# ── cli ───────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--out-prefix", default=None,
                   help="output path prefix (default: derived from --input)")
    p.add_argument("--limit", type=int, default=0, help="process only the first N records")
    p.add_argument("--mode", choices=["local", "full"], default="local",
                   help="'local': one pair per critique-flagged change (default). "
                        "'full': BASELINE — one pair per record over the whole sample, "
                        "prompt = question only, rejected/chosen = the entire original / "
                        "rewritten sequence")

    g = p.add_argument_group("change detection")
    g.add_argument("--merge-gap", type=int, default=40,
                   help="coalesce changes separated by fewer than this many identical chars")
    g.add_argument("--min-change-chars", type=int, default=2)
    g.add_argument("--max-change-chars", type=int, default=20000,
                   help="skip changes with more TRAINABLE chars than this (masked "
                        "<tool_output> does not count toward it)")
    g.add_argument("--max-pairs-per-record", type=int, default=0, help="0 = unlimited")

    g = p.add_argument_group("completion shape")
    g.add_argument("--min-tail-chars", type=int, default=0,
                   help="keep extending past sentence boundaries until the agreed tail "
                        "is at least this long. 0 (default) stops at the first boundary, "
                        "which is often 2 chars; a larger value buys a readable shared "
                        "continuation at the cost of loss mass on non-preference tokens")
    g.add_argument("--tail-max-chars", type=int, default=1500,
                   help="cap on the agreed tail when no sentence boundary is found")
    g.add_argument("--tail-to-end", action="store_true",
                   help="extend the tail to the next change / end of document instead "
                        "of stopping at the sentence boundary")
    g.add_argument("--mask-tool-output", dest="mask_tool_output",
                   action="store_true", default=True,
                   help="emit char spans of <tool_output> blocks for the trainer to "
                        "exclude from the loss (default: on)")
    g.add_argument("--no-mask-tool-output", dest="mask_tool_output",
                   action="store_false")

    g = p.add_argument_group("prefix")
    g.add_argument("--trace-prefix", choices=["original", "rewritten"], default="rewritten")
    g.add_argument("--answer-prefix", choices=["original", "rewritten"], default="rewritten",
                   help="form of the ANSWER text before the change; the trace part of an "
                        "answer prefix is always the rewritten trace")
    g.add_argument("--max-prefix-chars", type=int, default=0,
                   help="0 = full prefix. Traces are ~100 kB, so the full-prefix file is large")
    g.add_argument("--question-template", default=None,
                   help="literal template with a {question} placeholder; overrides "
                        "--question-template-file. '' omits the question entirely")
    g.add_argument("--question-template-file", default=PROMPT_TEMPLATE_FILE,
                   help="file holding the rendered DR Tulu prompt (default: %(default)s)")

    g = p.add_argument_group("eos")
    g.add_argument("--eos-policy", choices=["complete_only", "always", "never"],
                   default="complete_only")
    g.add_argument("--eos-token", default="",
                   help="literal EOS string to bake into completions, e.g. '<|im_end|>'")
    g.add_argument("--tokenizer", default=None,
                   help="HF id to resolve --eos-token from")

    g = p.add_argument_group("filters / splits")
    g.add_argument("--min-trace-ratio", type=float, default=0.5)
    g.add_argument("--max-sequence-tokens", type=int, default=32768,
                   help="drop records whose full sequence exceeds this many tokens, so "
                        "nothing is truncated at training time. 0 disables. Applied "
                        "identically to both arms (default: %(default)s, Qwen3-8B's "
                        "natively-trained context)")
    g.add_argument("--length-tokenizer", default="Qwen/Qwen3-8B",
                   help="tokenizer used by --max-sequence-tokens")
    g.add_argument("--strip-stray-answer-tags", dest="strip_stray_answer_tags",
                   action="store_true", default=True,
                   help="drop premature </answer> (and stray <answer>) from the rewritten "
                        "answer, keeping the first opener and last closer (default: on)")
    g.add_argument("--keep-stray-answer-tags", dest="strip_stray_answer_tags",
                   action="store_false")
    g.add_argument("--gate-rewritten", dest="gate_rewritten",
                   action="store_true", default=True,
                   help="skip a record entirely when its REWRITTEN output has a format "
                        "or citation defect (default: on). The original is never gated")
    g.add_argument("--no-gate-rewritten", dest="gate_rewritten", action="store_false")
    g.add_argument("--val-frac", type=float, default=0.0,
                   help="0 (default) writes only train/test")
    g.add_argument("--test-frac", type=float, default=0.05)
    g.add_argument("--seed", type=int, default=42)

    p.add_argument("--regen-prompt-template", metavar="DR_TULU_AGENT_DIR", default=None,
                   help="re-render drtulu_prompt_template.txt from a dr-tulu/agent "
                        "checkout and exit")
    return p.parse_args(argv)


def regen_prompt_template(agent_dir: str, out_path: str) -> None:
    """Re-render the DR Tulu prompt from a dr-tulu/agent checkout.

    Mirrors SearchAgent.prompt (workflows/auto_search_sft.py) for
    dataset_name="sqav2" and the v20250907 prompt version that
    auto_search_sft_s2_only_hamish.yaml leaves at its default, then applies the
    Qwen3-8B chat template exactly as LLMToolClient._messages_to_prompt does.
    """
    import yaml
    from transformers import AutoTokenizer

    yaml_path = os.path.join(
        agent_dir, "dr_agent", "shared_prompts", "unified_tool_calling_v20250907.yaml")
    with open(yaml_path) as f:
        prompts = yaml.safe_load(f)

    sentinel = "\x00QUESTION\x00"
    messages = [
        {"role": "system", "content": prompts["system_prompt"]},
        {"role": "user",
         "content": sentinel + "\n\n" + prompts["additional_instructions"]["long_form"]},
    ]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if rendered.count(sentinel) != 1 or "{question}" in rendered:
        sys.exit("unexpected rendering: sentinel not unique, or literal {question} in prompt")
    with open(out_path, "w") as f:
        f.write(rendered.replace(sentinel, "{question}"))
    print(f"wrote {out_path} ({len(rendered)} chars) from {yaml_path}")


def resolve_question_template(args, here: str) -> str:
    if args.question_template is not None:
        return args.question_template
    path = args.question_template_file
    if not os.path.isabs(path):
        path = os.path.join(here, path)
    if os.path.exists(path):
        with open(path) as f:
            template = f.read()
        print(f"question template: {path} ({len(template)} chars)")
        return template
    print(f"WARNING: {path} not found — falling back to a plain header, which does NOT "
          f"match the DR Tulu inference prompt. Run --regen-prompt-template <dr-tulu/agent>.")
    return FALLBACK_QUESTION_TEMPLATE


def main(argv=None):
    args = parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    if args.regen_prompt_template:
        regen_prompt_template(args.regen_prompt_template,
                              os.path.join(here, PROMPT_TEMPLATE_FILE))
        return
    args.question_template = resolve_question_template(args, here)

    if args.tokenizer and not args.eos_token:
        from transformers import AutoTokenizer
        args.eos_token = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True).eos_token or ""
        print(f"resolved --eos-token from {args.tokenizer}: {args.eos_token!r}")
    if args.eos_policy != "never" and not args.eos_token:
        print("note: no --eos-token given; pairs carry a `complete` flag only. "
              "Train with EOSAwareDPOTrainer (dpo_trainer_utils.py) to honor it.")

    H = _load_rewriter_helpers(os.path.join(here, HELPER_SOURCE))

    suffix = "_dpo_pairs" if args.mode == "local" else "_dpo_pairs_full"
    out_prefix = args.out_prefix or args.input.replace(".jsonl", suffix)

    records = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if args.limit:
        records = records[:args.limit]
    print(f"Loaded {len(records)} records from {args.input}")

    oversize = set()
    if args.max_sequence_tokens:
        drop = oversize_records(records, args.question_template,
                                args.max_sequence_tokens, args.length_tokenizer)
        if drop:
            print(f"dropping {len(drop)} records whose full sequence exceeds "
                  f"{args.max_sequence_tokens:,} tokens")
        # Marked, NOT removed: record_idx must stay the source line number (the
        # validator indexes the source by it) and len(records) drives the split
        # shuffle, so filtering the list would silently reshuffle every split.
        oversize = drop

    if args.strip_stray_answer_tags:
        n_fixed = 0
        for rec in records:
            cleaned = strip_stray_answer_tags(rec["rewritten"])
            if cleaned != rec["rewritten"]:
                rec["rewritten"] = cleaned
                n_fixed += 1
        if n_fixed:
            print(f"stripped stray <answer>/</answer> tags in {n_fixed} rewritten answers")

    rng = random.Random(args.seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    n_test = int(len(order) * args.test_frac)
    n_val = int(len(order) * args.val_frac)
    split_of = {}
    for i in order[:n_test]:
        split_of[i] = "test"
    for i in order[n_test:n_test + n_val]:
        split_of[i] = "val"
    for i in order[n_test + n_val:]:
        split_of[i] = "train"

    splits = [s for s in ("train", "val", "test") if s in set(split_of.values())]
    files = {s: open(f"{out_prefix}_{s}.jsonl", "w") for s in splits}
    stats = Counter()
    per_doc = Counter()
    lens = {"prefix": [], "chosen": [], "rejected": []}

    try:
        for i, rec in enumerate(records):
            if i in oversize:
                stats["skip:oversize"] += 1
                stats["records_with_no_pairs"] += 1
                continue
            pairs, skip = process_record(rec, i, H, args)
            if skip:
                stats[f"skip:{skip}"] += 1
            if not pairs:
                stats["records_with_no_pairs"] += 1
                continue
            stats["records_used"] += 1
            for p in pairs:
                per_doc[p["doc"]] += 1
                per_doc[f'{p["doc"]}:{p["kind"]}'] += 1
                stats["complete"] += int(p["complete"])
                lens["prefix"].append(p["prefix_chars"])
                lens["chosen"].append(p["chosen_chars"])
                lens["rejected"].append(p["rejected_chars"])
                files[split_of[i]].write(json.dumps(p) + "\n")
            stats["pairs"] += len(pairs)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(records)} records -> {stats['pairs']} pairs")
    finally:
        for f in files.values():
            f.close()

    def pct(v, d):
        return f"{100.0 * v / d:.1f}%" if d else "n/a"

    print("\n── summary ─────────────────────────────────────────────")
    print(f"records used            {stats['records_used']} / {len(records)}")
    for k in sorted(k for k in stats if k.startswith("skip:")):
        print(f"  {k:<40}{stats[k]}")
    print(f"records with no pairs   {stats['records_with_no_pairs']}")
    print(f"pairs                   {stats['pairs']}  "
          f"({stats['pairs'] / max(stats['records_used'], 1):.1f} per record)")
    for k in sorted(per_doc):
        print(f"  {k:<22}{per_doc[k]}")
    print(f"complete (EOS-eligible) {stats['complete']}  "
          f"({pct(stats['complete'], stats['pairs'])})")
    for name, v in lens.items():
        if v:
            v = sorted(v)
            print(f"{name+' chars':<24}median={v[len(v)//2]}  p90={v[int(.9*len(v))]}  max={v[-1]}")
    for s in splits:
        path = f"{out_prefix}_{s}.jsonl"
        n = sum(1 for _ in open(path))
        print(f"{s:<24}{n:>7} pairs  {os.path.getsize(path)/1e6:.1f} MB  {path}")


if __name__ == "__main__":
    main()
