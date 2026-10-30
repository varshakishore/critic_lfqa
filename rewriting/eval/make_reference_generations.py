"""Export the DPO training pair of each test question as two pseudo-models for the eval.

    original      the source DR Tulu trace + answer   (= rejected in the full arm)
    rewrite_luna  the gpt-5.6-luna rewrite             (= chosen   in the full arm)

Ceiling check for the metric: the rewrite was built to fix the critic's complaints,
so if a fresh gpt-5.4 critique does not find clearly fewer problems in `rewrite_luna`
than in `original`, critique count cannot detect the improvement DPO is meant to
teach, whatever the trained models do. One sample per question (sample_idx 0), since
each record has exactly one original and one rewrite.

Each sequence is asserted equal to the stored full-arm test pair, so the critic sees
byte-for-byte the text that was trained on.

    python eval/make_reference_generations.py
    python -u eval/score_eval.py --models original rewrite_luna   # critique them (~$5)
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from build_dpo_pairs import strip_stray_answer_tags  # noqa: E402

SOURCE = os.path.join(ROOT, "samples_1000", "drtulu_answers_w_critiques_rewritten_gpt-5.6-luna_v3_mod_fixed.jsonl")
FULL_TEST = os.path.join(ROOT, "samples_1000",
                         "drtulu_answers_w_critiques_rewritten_gpt-5.6-luna_v3_mod_fixed_dpo_pairs_full_test.jsonl")
OUT_DIR = os.path.join(HERE, "generations")


def answer_body(seq):
    m = re.search(r"<answer>(.*?)(</answer>|\Z)", seq, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def main():
    pairs = {}
    with open(FULL_TEST) as f:
        for line in f:
            p = json.loads(line)
            pairs[p["record_idx"]] = p
    with open(SOURCE) as f:
        source = f.readlines()

    out = {"original": [], "rewrite_luna": []}
    for idx in sorted(pairs):
        rec = json.loads(source[idx])
        orig_trace = rec["original_trace"]
        m = re.search(r"<answer>", orig_trace)
        orig_pre = orig_trace[:m.start()].rstrip()
        joiner = orig_trace[len(orig_pre):m.start()] or "\n"
        # Same construction as build_dpo_pairs.build_full_pair.
        seqs = {
            "original": orig_pre + joiner + rec["original_answer"],
            "rewrite_luna": rec["rewritten_trace"] + joiner + strip_stray_answer_tags(rec["rewritten"]),
        }
        side = {"original": "rejected", "rewrite_luna": "chosen"}
        for name, seq in seqs.items():
            stored = pairs[idx][side[name]]
            assert seq == stored, f"record {idx}: reconstructed {name} != stored {side[name]}"
            out[name].append({
                "model": name, "model_path": os.path.basename(SOURCE),
                "record_idx": idx, "sample_idx": 0, "question": rec["question"],
                "final_response": answer_body(seq), "full_traces": seq,
            })

    for name, rows in out.items():
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
