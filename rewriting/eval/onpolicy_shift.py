"""How far did DPO move each policy on the reference model's OWN samples?

`chosen` is luna's text, ~3x less likely per token than the policy's own, so a big
training margin can come from raising text the policy never samples. Generation
behaviour only changes if the distribution moved on text it *does* sample. This
scores the reference model's generations (eval/generations/<ref-gens>.jsonl) under
the reference and under each trained policy and reports, per region (trace
reasoning vs answer):

  KL       mean over tokens of log p_ref(y_t) - log p_pol(y_t), with y ~ ref:
           a Monte-Carlo estimate of the per-token KL(ref || pol). ~0 means the
           policy samples like the reference and the behavioural eval can show
           little.
  argmax   share of positions where the policy's greedy next token differs from
           the reference's: a direct read on "would it write something else here".

Same token accounting as training: DR Tulu prompt (drtulu_prompt_template.txt) +
generated trace, prompt and <tool_output> tokens excluded (tool_output_spans from
build_dpo_pairs, drop-dominant: a token touching a masked span is masked).

    python eval/onpolicy_shift.py                                   # SFT-init arms vs DR-Tulu-SFT-8B
    python eval/onpolicy_shift.py --ref rl-research/DR-Tulu-8B --ref-gens base \
        --policies full=dpo_out_full_b0.05_ep2 local=dpo_out_local_b0.05_ep1   # RL-init arms, for calibration
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from build_dpo_pairs import tool_output_spans  # noqa: E402

CHUNK = 4096  # positions per lm_head slice; the full fp32 log-softmax would be ~12 GB at 20k tokens


def build_items(gens_path, template, tok, limit):
    items = []
    with open(gens_path) as f:
        rows = [json.loads(l) for l in f if l.strip()][:limit]
    for r in rows:
        prompt = template.replace("{question}", r["question"].strip())
        trace = r["full_traces"]
        text = prompt + trace
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        masked = [(a + len(prompt), b + len(prompt)) for a, b in tool_output_spans(trace)]
        ans_at = trace.find("<answer>")
        ans_at = len(prompt) + (ans_at if ans_at >= 0 else len(trace))
        region = np.zeros(len(ids), dtype=np.int8)  # 0 = excluded, 1 = trace reasoning, 2 = answer
        for i, (s, e) in enumerate(offs):
            if s < len(prompt) or any(s < me and ms < e for ms, me in masked):
                continue
            region[i] = 2 if s >= ans_at else 1
        items.append({"key": (r["record_idx"], r["sample_idx"]), "ids": ids, "region": region})
    return items


@torch.no_grad()
def score(model_path, items, attn):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation=attn, device_map="cuda")
    model.eval()
    out = []
    for n, it in enumerate(items, 1):
        ids = torch.tensor([it["ids"]], device="cuda")
        hidden = model.model(input_ids=ids).last_hidden_state[0, :-1]  # predicts tokens 1..T-1
        targets = ids[0, 1:]
        lp, am = [], []
        for s in range(0, hidden.shape[0], CHUNK):
            logits = model.lm_head(hidden[s:s + CHUNK]).float()
            logp = torch.log_softmax(logits, dim=-1)
            lp.append(logp.gather(1, targets[s:s + CHUNK, None])[:, 0].cpu())
            am.append(logits.argmax(-1).cpu())
            del logits, logp
        # Position t of these arrays scores token t+1, so pad index 0 (the first
        # token has no prediction, and it is always prompt anyway).
        out.append((np.concatenate([[np.nan], torch.cat(lp).numpy()]),
                    np.concatenate([[-1], torch.cat(am).numpy()])))
        if n % 10 == 0 or n == len(items):
            print(f"    {n}/{len(items)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="rl-research/DR-Tulu-SFT-8B")
    ap.add_argument("--ref-gens", default="base_sft", help="eval/generations/<name>.jsonl sampled from --ref")
    ap.add_argument("--policies", nargs="+",
                    default=[f"full_sft=dpo_out_full_sft_b0.05_ep2", f"local_sft=dpo_out_local_sft_b0.05_ep1"])
    ap.add_argument("--template", default=os.path.join(ROOT, "drtulu_prompt_template.txt"))
    ap.add_argument("--limit", type=int, default=None, help="first N generations (smoke test)")
    ap.add_argument("--attn", default="flash_attention_2", choices=["flash_attention_2", "sdpa"])
    ap.add_argument("--out", default=None, help="default: eval/onpolicy_shift_<ref-gens>.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ref)
    assert tok.is_fast, "offset mapping needs a fast tokenizer"
    template = open(args.template).read()
    items = build_items(os.path.join(HERE, "generations", f"{args.ref_gens}.jsonl"), template, tok, args.limit)
    n_tok = [int((it["region"] == k).sum()) for it in items for k in (1, 2)]
    print(f"{len(items)} generations from {args.ref_gens}; scored tokens: "
          f"trace {sum(n_tok[0::2]):,}  answer {sum(n_tok[1::2]):,}  (max sequence {max(len(i['ids']) for i in items):,})")

    print(f"[ref] {args.ref}")
    ref = score(args.ref, items, args.attn)
    report = {}
    for spec in args.policies:
        name, path = spec.split("=", 1)
        path = path if os.path.isabs(path) or not os.path.exists(os.path.join(ROOT, path)) else os.path.join(ROOT, path)
        print(f"[policy] {name}: {path}")
        pol = score(path, items, args.attn)
        report[name] = {}
        for k, region in ((1, "trace"), (2, "answer"), (None, "all")):
            kl_seq, agree_tok, kl_tok = [], [], []
            for it, (rlp, ram), (plp, pam) in zip(items, ref, pol):
                m = (it["region"] > 0) if k is None else (it["region"] == k)
                if not m.any():
                    continue
                d = rlp[m] - plp[m]
                kl_seq.append(d.mean())
                kl_tok.append(d)
                agree_tok.append(ram[m] != pam[m])
            d_all, flip = np.concatenate(kl_tok), np.concatenate(agree_tok)
            report[name][region] = {
                "kl_per_token": float(d_all.mean()),
                "kl_per_token_seq_mean": float(np.mean(kl_seq)),
                "kl_per_token_seq_p10_p90": [float(np.percentile(kl_seq, 10)), float(np.percentile(kl_seq, 90))],
                "argmax_differs": float(flip.mean()),
                "tokens": int(d_all.size),
            }

    print(f"\nShift of each policy on {args.ref_gens} (sampled from {args.ref}); "
          "KL = mean log p_ref - log p_pol per token (nats), argmax = greedy token differs")
    print(f"{'policy':14}{'region':8}{'KL/token':>10}{'per-seq p10..p90':>22}{'argmax differs':>16}{'tokens':>12}")
    for name, regions in report.items():
        for region, r in regions.items():
            lo, hi = r["kl_per_token_seq_p10_p90"]
            print(f"{name:14}{region:8}{r['kl_per_token']:>10.4f}{f'{lo:.4f}..{hi:.4f}':>22}"
                  f"{100 * r['argmax_differs']:>15.2f}%{r['tokens']:>12,}")
    out = args.out or os.path.join(HERE, f"onpolicy_shift_{args.ref_gens}.json")
    with open(out, "w") as f:
        json.dump({"ref": args.ref, "ref_gens": args.ref_gens, "report": report}, f, indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
