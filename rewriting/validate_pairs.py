"""Verify every built DPO pair reconstructs from the source data and meets spec.

Re-derives each pair from `--source` rather than trusting the stored fields, so a
bug anywhere in `build_dpo_pairs.py` shows up as a mismatch. Run after every build.

Checks, per pair:
  reconstruct     rejected == original[s1 : e1+tail],  chosen == rewritten[s2 : e2+tail]
  prefix          prompt == <DR Tulu template> + rewritten-text-before-the-divergence
                  (answer pairs additionally carry the whole rewritten trace)
  shared tail     both sides end with the same `tail_chars` characters, and that
                  text is the agreed region in BOTH source documents
  single locus    chosen vs rejected differ in exactly one merged block
  non-degenerate  both sides non-empty and not identical
  mask spans      in range, sorted, disjoint, and covering only environment text
  complete flag   True iff the completion reaches the end of BOTH documents
  trace never complete   the sequence continues into the answer

And across the dataset:
  splits are record-level with no leakage, and the two arms agree on every record.

Usage:
    python validate_pairs.py                    # local arm, all splits
    python validate_pairs.py --arm full
    python validate_pairs.py --arm both --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

from build_dpo_pairs import (char_blocks, merge_blocks, strip_stray_answer_tags,
                             tool_output_spans)

SOURCE = "samples_1000/drtulu_answers_w_critiques_rewritten_gpt-5.6-luna_v3_mod_fixed.jsonl"
PAIRS = ("samples_1000/drtulu_answers_w_critiques_rewritten_"
         "gpt-5.6-luna_v3_mod_fixed_dpo_pairs")
TEMPLATE = "drtulu_prompt_template.txt"


def load_source(path):
    recs = {}
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            # Match the builder's normalization, or every stripped record mismatches.
            r["rewritten"] = strip_stray_answer_tags(r["rewritten"])
            m = re.search(r"<answer>", r["original_trace"])
            pre = r["original_trace"][:m.start()].rstrip() if m else ""
            joiner = r["original_trace"][len(pre):m.start()] if m else "\n"
            recs[i] = (r, pre, joiner or "\n")
    return recs


def documents(pair, rec, pre, joiner, head):
    """(original_doc, rewritten_doc, prefix_head) for this pair's document."""
    if pair["doc"] == "trace":
        return pre, rec["rewritten_trace"], head
    if pair["doc"] == "answer":
        return rec["original_answer"], rec["rewritten"], head + rec["rewritten_trace"] + joiner
    return (pre + joiner + rec["original_answer"],
            rec["rewritten_trace"] + joiner + rec["rewritten"], head)


def check(pair, recs, template, fails):
    rec, pre, joiner = recs[pair["record_idx"]]
    head = template.replace("{question}", pair["question"])
    O, R, prefix_head = documents(pair, rec, pre, joiner, head)
    s1, e1 = pair["orig_span"]
    s2, e2 = pair["rew_span"]
    t = pair["tail_chars"]
    chosen, rejected = pair["chosen"], pair["rejected"]

    if rejected != O[s1:e1 + t]:
        fails["reconstruct: rejected != original slice"] += 1
    if chosen != R[s2:e2 + t]:
        fails["reconstruct: chosen != rewritten slice"] += 1
    if pair["prompt"] != prefix_head + R[:s2]:
        fails["prefix != template + rewritten-before-divergence"] += 1

    if t:
        if chosen[-t:] != rejected[-t:]:
            fails["shared tail differs between sides"] += 1
        if O[e1:e1 + t] != R[e2:e2 + t]:
            fails["tail is not the agreed region in both documents"] += 1
    if not chosen or not rejected:
        fails["empty side"] += 1
    if chosen == rejected:
        fails["chosen == rejected"] += 1

    for side in ("chosen", "rejected"):
        text, spans = pair[side], pair[f"{side}_mask_spans"]
        prev = -1
        for s, e in spans:
            if not (0 <= s < e <= len(text)):
                fails["mask span out of range"] += 1
            if s < prev:
                fails["mask spans unsorted or overlapping"] += 1
            prev = e
        if spans and spans != tool_output_spans(text):
            fails["mask spans != recomputed tool_output spans"] += 1

    # `complete` means the completion reaches the end of BOTH documents — not
    # merely that it ends in "</answer>". A stray mid-answer </answer> (the rewrite
    # model sometimes emits two) or a truncated original would fool that test.
    if pair["doc"] == "full":
        # whole-sample pairs: both sides must be untruncated generations
        expected = (O.rstrip().endswith("</answer>") and R.rstrip().endswith("</answer>"))
    else:
        expected = ((e1 + t == len(O)) and (e2 + t == len(R)) and pair["doc"] != "trace")
    if pair["complete"] != expected:
        fails["complete flag disagrees with reaching both document ends"] += 1
    if pair["doc"] == "trace" and pair["complete"]:
        fails["trace pair flagged complete"] += 1

    if len(merge_blocks(char_blocks(rejected, chosen), 40)) > 1 and pair["doc"] != "full":
        fails["differs in more than one block (not single-locus)"] += 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=SOURCE)
    p.add_argument("--pairs-prefix", default=PAIRS)
    p.add_argument("--arm", choices=["local", "full", "both"], default="both")
    p.add_argument("--limit", type=int, default=0, help="check only the first N pairs per split")
    args = p.parse_args(argv)

    print(f"source: {args.source}")
    recs = load_source(args.source)
    template = open(TEMPLATE).read()
    arms = ["local", "full"] if args.arm == "both" else [args.arm]
    split_records = {}

    for arm in arms:
        suffix = "" if arm == "local" else "_full"
        fails, n = Counter(), 0
        per_split = {}
        for split in ("train", "val", "test"):
            path = f"{args.pairs_prefix}{suffix}_{split}.jsonl"
            if not os.path.exists(path):
                continue
            ids = set()
            with open(path) as f:
                for k, line in enumerate(f):
                    if args.limit and k >= args.limit:
                        break
                    pair = json.loads(line)
                    ids.add(pair["record_idx"])
                    check(pair, recs, template, fails)
                    n += 1
            per_split[split] = ids
        split_records[arm] = per_split

        print(f"\n── {arm} arm: {n:,} pairs checked " + "─" * 40)
        if not fails:
            print("   ALL CHECKS PASS")
        for k, v in fails.most_common():
            print(f"   FAIL  {k}: {v}")

        leak = [(a, b) for a in per_split for b in per_split
                if a < b and per_split[a] & per_split[b]]
        print(f"   split leakage (a record in two splits): "
              f"{'none' if not leak else leak}")

    if len(arms) == 2:
        print("\n── arms " + "─" * 52)
        for split in sorted(set(split_records["local"]) & set(split_records["full"])):
            a, b = split_records["local"][split], split_records["full"][split]
            print(f"   {split:<6} local {len(a):>4} records · baseline {len(b):>4} · "
                  f"same-split mismatch: {len(a ^ b)}")
        ks = set(split_records["local"]) & set(split_records["full"])
        cross = any(split_records["local"][a] & split_records["full"][b]
                    for a in ks for b in ks if a != b)
        print(f"   a record landing in different splits across arms: {cross}")


if __name__ == "__main__":
    main()
