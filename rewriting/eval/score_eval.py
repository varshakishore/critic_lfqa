"""Behavioural eval, step 2: re-critique each generation with gpt-5.4 and compare arms.

Uses the exact critique prompt (`updated_prompt_v1`) and model that produced the
training critiques, so "fewer critiques" means "fewer of the problems the DPO data
was built to fix". The prompt is pulled out of pg_dr_tulu.py with `ast` rather than
imported, because importing that script runs it.

Two stages, both resumable:

1. critique  — eval/critiques/<model>.jsonl, one row per (record_idx, sample_idx).
               Already-critiqued rows are skipped; API failures are not written and
               are retried on the next run.
2. compare   — no API calls. Per model: critiques per generation (total and by
               location), plus cheap structural stats of the generations themselves.
               Each DPO model is compared with `base` *paired by question*: average
               over samples within a question, take the difference, bootstrap over
               the 31 questions. Questions — not generations — are the unit, since
               samples of one question are not independent.

    python eval/score_eval.py                 # critique whatever is missing, then compare
    python eval/score_eval.py --compare-only  # just the report
"""
import argparse
import ast
import glob
import json
import os
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from critique_categories import NAMES as CATEGORY_NAMES, categorize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAIRS_PREFIX = os.path.join(
    ROOT, "samples_1000", "drtulu_answers_w_critiques_rewritten_gpt-5.6-luna_v3_mod_fixed_dpo_pairs")

MODEL_COSTS = {"gpt-5.4": (2.50, 15.00)}  # $ per 1M in / out, as in pg_dr_tulu.py


def load_critique_prompt():
    src = open(os.path.join(ROOT, "pg_dr_tulu.py")).read()
    for node in ast.parse(src).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == "updated_prompt_v1"):
            return ast.literal_eval(node.value)
    raise RuntimeError("updated_prompt_v1 not found in pg_dr_tulu.py")


def parse_critique(s):
    """Same as rewrite_answer_from_critiques.parse_critique (that module also runs
    on import): plain json, then retry with quotes inside <...> tags escaped."""
    try:
        return json.loads(s)
    except Exception:
        pass
    repaired = re.sub(r"<[^<>]*>",
                      lambda m: re.sub(r'(?<!\\)"', lambda _: '\\"', m.group(0)),
                      s)
    try:
        return json.loads(repaired)
    except Exception:
        return None


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- critique

def critique_model(gen_path, out_path, prompt_tmpl, args, budget):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    done = {(r["record_idx"], r["sample_idx"]) for r in read_jsonl(out_path)}
    todo = [g for g in read_jsonl(gen_path) if (g["record_idx"], g["sample_idx"]) not in done]
    name = os.path.basename(gen_path)[:-len(".jsonl")]
    print(f"[{name}] {len(done)} critiqued, {len(todo)} to go")
    in_price, out_price = MODEL_COSTS.get(args.critic, (0.0, 0.0))

    def one(g):
        if budget["stop"].is_set():
            return
        prompt = prompt_tmpl.format(g["question"], g["full_traces"])
        resp = client.responses.create(model=args.critic, input=prompt)
        cost = (resp.usage.input_tokens * in_price + resp.usage.output_tokens * out_price) / 1e6
        row = {"model": g["model"], "record_idx": g["record_idx"], "sample_idx": g["sample_idx"],
               "critic": args.critic, "critique": resp.output_text, "cost": cost}
        with budget["lock"]:
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            budget["spent"] += cost
            if budget["spent"] >= args.cost_limit and not budget["stop"].is_set():
                budget["stop"].set()
                print(f"!! cost limit ${args.cost_limit:.2f} reached; not starting new critiques")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, g): g for g in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            g = futs[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"  [fail] {name} rec {g['record_idx']} s{g['sample_idx']}: {type(e).__name__}: {e}")
            if i % 10 == 0 or i == len(todo):
                print(f"  [{name}] {i}/{len(todo)}  spent ${budget['spent']:.2f}", flush=True)


# ---------------------------------------------------------------- compare

CITE_RE = re.compile(r"""<cite id=["']?([^"'>]*)""")
SNIPPET_RE = re.compile(r"""<snippet id=["']?([^"'>\s]+)""")


def gen_stats(g):
    trace, ans = g["full_traces"], g["final_response"] or ""
    snippets = set(SNIPPET_RE.findall(trace))
    cites = [c.strip() for v in CITE_RE.findall(ans) for c in v.split(",") if c.strip()]
    return {
        "answer_chars": len(ans),
        "trace_chars": len(trace),
        "search_calls": trace.count("<call_tool"),
        "cites": len(cites),
        "dangling_cites": sum(c not in snippets for c in cites),
        "has_answer_tag": "<answer>" in trace and "</answer>" in trace,
    }


def crit_stats(c):
    parsed = parse_critique(c["critique"])
    if not isinstance(parsed, dict) or not isinstance(parsed.get("local"), list):
        return None
    items = [x for x in parsed["local"] if isinstance(x, dict)]
    loc = Counter(x.get("location", "?") for x in items)
    tags = [str(x.get("tag", "")).strip().lower() for x in items]
    cats = Counter(categorize(t) for t in tags)
    return {
        "n": len(items),
        "answer": loc["answer"],
        "plan": loc["plan"],
        "both": loc["both"],
        "search_required": sum(bool(x.get("search_required")) for x in items),
        "organization": sum(bool(x.get("organization_related")) for x in items),
        "tags": tags,
        **{f"cat_{k}": cats[k] for k in CATEGORY_NAMES},
    }


def per_question(rows, key):
    """{record_idx: mean of key over that question's samples}."""
    acc = defaultdict(list)
    for r in rows:
        acc[r["record_idx"]].append(r[key])
    return {q: float(np.mean(v)) for q, v in acc.items()}


def paired(base_q, other_q, n_boot, seed=0):
    qs = sorted(set(base_q) & set(other_q))
    d = np.array([other_q[q] - base_q[q] for q in qs])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_q": len(qs), "diff": d.mean(), "lo": lo, "hi": hi,
            "better": int((d < 0).sum()), "tie": int((d == 0).sum()), "worse": int((d > 0).sum())}


def training_exposure():
    """Share of the DPO training signal per critique category, for each arm.
    local: share of local train pairs carrying the category (a pair with tags in two
    categories counts for both). full: share of the critique tags attached to the
    full-arm train pairs (one pair per record, carrying all of that record's tags).

    build_dpo_pairs.attribute() tags a pair only when the change falls inside a
    critique's edit_span, but Step 1b moves an inserted search round to after the
    round's </tool_output> — outside that span. So ~590 local trace pairs that are
    search insertions carry no tag; count an untagged pair whose changed segment adds
    a <call_tool> as `search`, or search exposure reads as half what it is."""
    out = {}
    for arm, path in (("local", f"{PAIRS_PREFIX}_train.jsonl"), ("full", f"{PAIRS_PREFIX}_full_train.jsonl")):
        cnt, n = Counter(), 0
        for r in read_jsonl(path):
            cats = [categorize(t) for t in r.get("critique_tags") or []]
            if arm == "local" and not cats and "<call_tool" in r["chosen"][:r["chosen_chars"] - r["tail_chars"]]:
                cats = ["search"]
            if arm == "local":
                cnt.update(set(cats))
                n += 1
            else:
                cnt.update(cats)
                n += len(cats)
        out[arm] = {k: cnt[k] / max(n, 1) for k in CATEGORY_NAMES}
    return out


def spearman(x, y):
    rx, ry = (np.argsort(np.argsort(v)).astype(float) for v in (x, y))
    return float(np.corrcoef(rx, ry)[0, 1])


def category_report(rows, models, args):
    """Did the critique categories training saw most shrink most?"""
    exposure = training_exposure()
    base = args.baseline
    others = [m for m in models if m != base]
    print("\nCritique categories (eval/critique_categories.py). Training share = fraction of local train pairs "
          "/ of full-arm tags in that category.\nRates are critiques per generation; Δ = paired per-question "
          f"difference vs {base} with 95% bootstrap CI; rel = Δ / {base} rate.")
    head = f"{'category':14}{'train local':>12}{'train full':>11}{base:>9}"
    for m in others:
        head += f"  | {m[:22]:>22} {'Δ [95% CI]':>22} {'rel':>6}"
    print(head)
    rel = {m: {} for m in others}
    for k in CATEGORY_NAMES:
        key = f"cat_{k}"
        b = np.mean([r[key] for r in rows[base]])
        line = (f"{k:14}{100 * exposure['local'][k]:>11.1f}%{100 * exposure['full'][k]:>10.1f}%{b:>9.2f}")
        base_q = per_question(rows[base], key)
        for m in others:
            p = paired(base_q, per_question(rows[m], key), args.n_boot)
            sig = "*" if p["lo"] > 0 or p["hi"] < 0 else " "
            r = p["diff"] / b if b > 0 else float("nan")
            rel[m][k] = r
            ci = f"{p['diff']:+.2f} [{p['lo']:+.2f},{p['hi']:+.2f}]{sig}"
            line += f"  | {np.mean([x[key] for x in rows[m]]):>22.2f} {ci:>22} {100 * r:>5.0f}%"
        print(line)
    print("  * = CI excludes 0. Eleven categories x models are tested at once, so expect ~1 false * per "
          "model by chance at 95%.")

    print("\nDose-response: Spearman rank correlation across categories between training share and relative "
          "change (negative = more-trained categories dropped more). 'other' excluded; descriptive only, "
          "n = 10 categories.")
    cats = [k for k in CATEGORY_NAMES if k != "other"]
    for m in others:
        arm = "local" if "local" in m else "full" if "full" in m else None
        for a in ([arm] if arm else ["local", "full"]):
            x = [exposure[a][k] for k in cats if not np.isnan(rel[m][k])]
            y = [rel[m][k] for k in cats if not np.isnan(rel[m][k])]
            if len(x) >= 3:
                print(f"  {m:24} vs {a}-arm training share: rho = {spearman(x, y):+.2f}  (n={len(x)})")


def compare(models, args):
    gens = {m: {(g["record_idx"], g["sample_idx"]): g
                for g in read_jsonl(os.path.join(args.gen_dir, f"{m}.jsonl"))} for m in models}
    crits = {m: read_jsonl(os.path.join(args.crit_dir, f"{m}.jsonl")) for m in models}

    # Restrict every model to the questions critiqued for *all* models, so a partial
    # run never compares models on different questions. By question, not by
    # (question, sample): the reference models (make_reference_generations.py) have
    # one sample per question, and keying on samples would silently drop samples 1-2
    # of every generated model. Paired stats average samples within a question anyway.
    common = set.intersection(*[{c["record_idx"] for c in crits[m]} for m in models])
    if not common:
        print("no question critiqued for every model yet")
        return

    rows, n_unparsed, spent = {}, {}, {}
    for m in models:
        rows[m], n_unparsed[m] = [], 0
        spent[m] = sum(c.get("cost", 0) for c in crits[m])
        for c in crits[m]:
            k = (c["record_idx"], c["sample_idx"])
            if k[0] not in common:
                continue
            cs = crit_stats(c)
            if cs is None:
                n_unparsed[m] += 1
                continue
            row = {"record_idx": k[0], "sample_idx": k[1], **cs, **gen_stats(gens[m][k])}
            row["n_per_1k_answer_chars"] = 1000 * row["n"] / max(row["answer_chars"], 1)
            rows[m].append(row)

    print(f"\n{len(common)} questions critiqued for all of: {', '.join(models)}  (generations: "
          + ", ".join(f"{m} {len(rows[m])}" for m in models) + ")")
    print("unparseable critiques (excluded): " + ", ".join(f"{m} {n_unparsed[m]}" for m in models))
    print("critique spend: " + ", ".join(f"{m} ${spent[m]:.2f}" for m in models))

    metrics = [  # (key, label)
        ("n", "critiques / generation   (lower = better)"),
        ("answer", "  location=answer"),
        ("plan", "  location=plan"),
        ("both", "  location=both"),
        ("search_required", "  search_required"),
        ("organization", "  organization_related"),
        ("n_per_1k_answer_chars", "critiques / 1k answer chars"),
        ("answer_chars", "answer chars"),
        ("trace_chars", "trace chars"),
        ("search_calls", "search calls"),
        ("cites", "cites in answer"),
        ("dangling_cites", "dangling cites"),
        ("has_answer_tag", "has <answer>...</answer>"),
    ]
    w = max(14, *(len(m) for m in models))
    print("\nMeans per generation")
    print(f"{'':42}" + "".join(f"{m:>{w + 2}}" for m in models))
    for key, label in metrics:
        print(f"{label:42}" + "".join(f"{np.mean([r[key] for r in rows[m]]):>{w + 2}.2f}" for m in models))

    if args.baseline in models:
        print(f"\nPaired vs {args.baseline}: per-question mean difference (model - {args.baseline}), "
              f"95% bootstrap CI over questions; better/tie/worse = questions with fewer/equal/more")
        for key in ("n", "answer", "plan", "n_per_1k_answer_chars", "answer_chars", "dangling_cites"):
            print(f"  {key}")
            base_q = per_question(rows[args.baseline], key)
            for m in models:
                if m == args.baseline:
                    continue
                p = paired(base_q, per_question(rows[m], key), args.n_boot)
                sig = "*" if p["lo"] > 0 or p["hi"] < 0 else " "
                print(f"    {m:{w}} {p['diff']:+9.2f}  [{p['lo']:+.2f}, {p['hi']:+.2f}]{sig}  "
                      f"{p['better']}/{p['tie']}/{p['worse']}  (n={p['n_q']})")
        print("  * = CI excludes 0")

    if args.baseline in models and len(models) > 1:
        category_report(rows, models, args)

    print(f"\nTop {args.top_tags} critique tags (count per generation)")
    for m in models:
        tags = Counter(t for r in rows[m] for t in r["tags"])
        top = ", ".join(f"{t} {c / len(rows[m]):.2f}" for t, c in tags.most_common(args.top_tags))
        print(f"  {m}: {top}")

    summary_path = os.path.join(args.crit_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(rows, f)
    print(f"\nper-generation rows -> {summary_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None,
                    help="model names (= eval/generations/<name>.jsonl); default: all files there")
    ap.add_argument("--gen-dir", default=os.path.join(HERE, "generations"))
    ap.add_argument("--crit-dir", default=os.path.join(HERE, "critiques"))
    ap.add_argument("--critic", default="gpt-5.4")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--cost-limit", type=float, default=50.0, help="$ cap across all models this run")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--top-tags", type=int, default=8)
    args = ap.parse_args()

    models = args.models or sorted(os.path.basename(p)[:-len(".jsonl")]
                                   for p in glob.glob(os.path.join(args.gen_dir, "*.jsonl")))
    if args.baseline in models:  # baseline first in every table
        models = [args.baseline] + [m for m in models if m != args.baseline]
    os.makedirs(args.crit_dir, exist_ok=True)

    if not args.compare_only:
        prompt_tmpl = load_critique_prompt()
        budget = {"spent": 0.0, "lock": threading.Lock(), "stop": threading.Event()}
        for m in models:
            critique_model(os.path.join(args.gen_dir, f"{m}.jsonl"),
                           os.path.join(args.crit_dir, f"{m}.jsonl"), prompt_tmpl, args, budget)
        print(f"\ncritique spend this run: ${budget['spent']:.2f}")

    compare(models, args)


if __name__ == "__main__":
    main()
