"""Test-retest reliability of the gpt-5.4 critic.

Re-critiques generations that already have a critique (same prompt, same model) and
asks how much of the critique-to-critique variation is the critic itself:

    retest      same generation, critiqued twice          -> critic noise only
    resample    other samples of the same question/model  -> generation variance + critic noise
    other q     different questions, same model           -> floor

Measures per pair: category-profile cosine, issue-text TF-IDF cosine, |difference in
total critiques|, and — for retest pairs, where both critiques quote the same text —
span agreement: the share of one critique's critique_span starts that the other also
flags (matched on the first 40 chars, the same anchoring the rewrite pipeline uses).
Plus, across generations, how well the retest reproduces each count (Pearson r).

    python -u eval/critic_retest.py                    # base_sft, sample 0: 31 critiques, ~$3
    python eval/critic_retest.py --analyze-only
"""
import argparse
import collections
import itertools
import json
import os
import re
import threading

import numpy as np

from critique_categories import NAMES, categorize
from score_eval import HERE, critique_model, load_critique_prompt, parse_critique, read_jsonl

STOP = set("the a an and or of to in on for is are be this that it its as with by from at not but which should "
           "answer trace plan more than into their would can was were has have these those such so also only".split())


def items_of(critique_text):
    d = parse_critique(critique_text)
    if not isinstance(d, dict) or not isinstance(d.get("local"), list):
        return None
    return [x for x in d["local"] if isinstance(x, dict)]


def profile(items):
    v = np.array([sum(categorize(x.get("tag", "")) == k for x in items) for k in NAMES], float)
    return v / (np.linalg.norm(v) + 1e-12)


def span_starts(items):
    out = set()
    for x in items:
        for pair in x.get("critique_span") or []:
            if isinstance(pair, (list, tuple)) and pair and isinstance(pair[0], str) and pair[0].strip():
                out.add(pair[0].strip()[:40])
    return out


def tfidf(docs):
    toks = [[w for w in re.findall(r"[a-z][a-z\-]+", d.lower()) if w not in STOP and len(w) > 2] for d in docs]
    df = collections.Counter(w for t in toks for w in set(t))
    vocab = {w: i for i, w in enumerate(w for w, c in df.items() if c >= 2)}
    idf = np.zeros(len(vocab))
    for w, i in vocab.items():
        idf[i] = np.log(len(docs) / df[w])
    X = np.zeros((len(docs), len(vocab)))
    for r, t in enumerate(toks):
        for w, c in collections.Counter(t).items():
            if w in vocab:
                X[r, vocab[w]] = (1 + np.log(c)) * idf[vocab[w]]
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def analyze(model, retest_path, crit_dir):
    first = {(c["record_idx"], c["sample_idx"]): c for c in read_jsonl(os.path.join(crit_dir, f"{model}.jsonl"))}
    second = {(c["record_idx"], c["sample_idx"]): c for c in read_jsonl(retest_path)}
    crits = []  # (key, which, items)
    for k, c in first.items():
        it = items_of(c["critique"])
        if it is not None:
            crits.append((k, "first", it))
    for k, c in second.items():
        it = items_of(c["critique"])
        if it is not None:
            crits.append((k, "retest", it))
    X = tfidf([" ".join(str(x.get("issue", "")) for x in it) for _, _, it in crits])
    P = np.array([profile(it) for _, _, it in crits])
    idx = {(k, w): i for i, (k, w, _) in enumerate(crits)}
    retested = sorted(k for k in second if (k, "first") in idx and (k, "retest") in idx)

    rows = collections.defaultdict(list)
    for k in retested:
        a, b = idx[(k, "first")], idx[(k, "retest")]
        sa, sb = span_starts(crits[a][2]), span_starts(crits[b][2])
        span = len(sa & sb) / max(1, min(len(sa), len(sb))) if sa and sb else np.nan
        rows["retest"].append((P[a] @ P[b], X[a] @ X[b], abs(len(crits[a][2]) - len(crits[b][2])), span))
        # the same generation's first critique vs the first critiques of the other samples of its question
        for (k2, w2, it2), j in ((crits[j], j) for j in range(len(crits))):
            if w2 != "first" or k2 == k:
                continue
            kind = "resample" if k2[0] == k[0] else "other q"
            rows[kind].append((P[a] @ P[j], X[a] @ X[j], abs(len(crits[a][2]) - len(it2)), np.nan))

    print(f"\n{model}: {len(retested)} generations critiqued twice")
    print(f"{'pair':12}{'pairs':>7}{'category profile':>18}{'issue text':>12}{'|Δ total|':>11}{'span agreement':>16}")
    for kind in ("retest", "resample", "other q"):
        v = np.array(rows[kind], dtype=float)
        if not len(v):
            continue
        span = np.nanmean(v[:, 3]) if not np.isnan(v[:, 3]).all() else float("nan")
        print(f"{kind:12}{len(v):>7}{v[:, 0].mean():>18.3f}{v[:, 1].mean():>12.3f}{v[:, 2].mean():>11.2f}"
              + (f"{100 * span:>15.0f}%" if not np.isnan(span) else f"{'—':>16}"))

    print("\nDoes the retest reproduce the counts? Pearson r across generations "
          "(1 = critic perfectly repeatable; compare with the resample ICC in the earlier analysis)")
    for k_name in ["total"] + NAMES:
        a, b = [], []
        for k in retested:
            ia, ib = crits[idx[(k, "first")]][2], crits[idx[(k, "retest")]][2]
            if k_name == "total":
                a.append(len(ia)); b.append(len(ib))
            else:
                a.append(sum(categorize(x.get("tag", "")) == k_name for x in ia))
                b.append(sum(categorize(x.get("tag", "")) == k_name for x in ib))
        a, b = np.array(a, float), np.array(b, float)
        r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else float("nan")
        print(f"  {k_name:14} r = {r:+.2f}   mean first {a.mean():.2f}  retest {b.mean():.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="base_sft")
    ap.add_argument("--samples", type=int, nargs="+", default=[0])
    ap.add_argument("--gen-dir", default=os.path.join(HERE, "generations"))
    ap.add_argument("--crit-dir", default=os.path.join(HERE, "critiques"))
    ap.add_argument("--critic", default="gpt-5.4")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--cost-limit", type=float, default=15.0)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(args.crit_dir, "retest")
    os.makedirs(out_dir, exist_ok=True)
    retest_path = os.path.join(out_dir, f"{args.model}.jsonl")
    if not args.analyze_only:
        # critique_model reads a generations file, so hand it just the rows to retest.
        subset = os.path.join(out_dir, f"{args.model}.inputs.jsonl")
        with open(subset, "w") as f:
            for g in read_jsonl(os.path.join(args.gen_dir, f"{args.model}.jsonl")):
                if g["sample_idx"] in args.samples:
                    f.write(json.dumps(g) + "\n")
        budget = {"spent": 0.0, "lock": threading.Lock(), "stop": threading.Event()}
        critique_model(subset, retest_path, load_critique_prompt(), args, budget)
        print(f"retest spend: ${budget['spent']:.2f}")
    analyze(args.model, retest_path, args.crit_dir)


if __name__ == "__main__":
    main()
