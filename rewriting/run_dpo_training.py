"""DPO training on critique-guided rewrite pairs (DR-Tulu-8B).

Two arms, built by `build_dpo_pairs.py` from the same records and the same split:

  --arm local   one pair per critique-flagged change; chosen/rejected share a
                byte-identical prefix and diverge at one span. ~9.3k train pairs,
                ~250 gradient-bearing tokens each, ~90% of which are the fix.

  --arm full    the BASELINE global rewrite: one pair per record, prompt = the
                question alone, chosen/rejected = the entire rewritten / original
                sequence. ~750 train pairs, ~9.3k gradient tokens each, of which
                only ~15% is critique-flagged edit.

The arms are NOT compute- or token-matched, and shouldn't be forced to be: the
difference in signal density is the hypothesis. Tune each separately and compare
best held-out results, reporting the budget each used.

Things this script gets right that a stock DPO setup does not
------------------------------------------------------------
* `MaskedDPOTrainer` (dpo_trainer_utils.py) rather than `DPOTrainer`:
    - EOS is appended only to completions flagged `complete` (those reaching
      `</answer>`). TRL appends it to every completion, which on these pairs would
      train the model to stop mid-answer — only ~4% of local pairs actually end.
    - `<tool_output>` spans are dropped from the DPO log-prob sum. The policy never
      emits them (`</call_tool>` is a stop sequence; the harness splices results
      in), and both DR Tulu training stages exclude them. Unmasked they are ~90% of
      trace-pair completion tokens and ~80% of the baseline's.
* `remove_unused_columns=False` — required, or TRL drops `complete` and the mask
  spans before `tokenize_row` sees them, silently reverting both behaviours.
* Prompts are pre-rendered with the Qwen3 chat template by `build_dpo_pairs.py`
  (they end `<|im_start|>assistant\\n`), so no template is applied here. TRL calls
  `tokenize_row` with `add_special_tokens=False` and the dataset is
  non-conversational, so nothing gets prepended or appended. Checked at startup.

Lengths (trl 1.13)
------------------
1.13 dropped `max_prompt_length` / `max_completion_length`; only `max_length` caps
the whole sequence. `build_dpo_pairs.py --max-sequence-tokens 32768` already drops
any record that would not fit, so nothing truncates at the default.

It also dropped `use_logits_to_keep`, which on 0.24 confined the logits tensor to
the scored region. Logits are now materialised over the whole sequence, so the
local arm's ~20k-token prompts cost ~12 GB of logits per pair rather than ~0.1 GB.
If that is tight, `--use-liger-kernel` fuses the DPO loss and avoids materialising
them at all (`DPOTrainer._compute_loss_liger`).

Usage
-----
    python run_dpo_training.py --arm local --dry-run     # stats, no training
    python run_dpo_training.py --arm local
    python run_dpo_training.py --arm full --beta 0.05 --epochs 3

    # 8-GPU node. NOTE stage3_*_accelerate.conf are DEEPSPEED json configs despite
    # the name, so they go to --deepspeed, NOT to `accelerate launch --config_file`
    # (which expects an accelerate YAML).
    torchrun --nproc_per_node 8 run_dpo_training.py --arm local \\
        --deepspeed ../stage3_no_offloading_accelerate.conf
"""

from __future__ import annotations

import argparse
import os
import statistics as st

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig

from dpo_trainer_utils import MaskedDPOTrainer, load_pairs, spans_to_token_mask

PAIRS = ("samples_1000/drtulu_answers_w_critiques_rewritten_"
         "gpt-5.6-luna_v3_mod_fixed_dpo_pairs")

DEFAULT_MAX_LENGTH = 32768


def dataset_paths(arm: str, prefix: str, eval_split: str) -> tuple[str, str]:
    suffix = "" if arm == "local" else "_full"
    return f"{prefix}{suffix}_train.jsonl", f"{prefix}{suffix}_{eval_split}.jsonl"


def report(dataset, tokenizer, name: str, n: int = 100) -> None:
    """Token budget for the arm — the number that decides cost and feasibility."""
    idx = range(0, len(dataset), max(1, len(dataset) // n))
    pr, ch, rj = [], [], []
    for i in idx:
        row = dataset[i]
        pr.append(len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]))
        _, mc = spans_to_token_mask(row["chosen"], row.get("chosen_mask_spans") or [], tokenizer)
        _, mr = spans_to_token_mask(row["rejected"], row.get("rejected_mask_spans") or [], tokenizer)
        ch.append(sum(mc))
        rj.append(sum(mr))
    grad = len(dataset) * (st.mean(ch) + st.mean(rj))
    print(f"  {name}: {len(dataset):,} pairs")
    print(f"     prompt tokens     median {st.median(pr):>8,.0f}   p90 {sorted(pr)[int(.9 * len(pr))]:>8,}")
    print(f"     chosen  unmasked  median {st.median(ch):>8,.0f}")
    print(f"     rejected unmasked median {st.median(rj):>8,.0f}")
    print(f"     gradient-bearing tokens / epoch: {grad:,.0f}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=["local", "full"], default="local")
    p.add_argument("--pairs-prefix", default=PAIRS)
    p.add_argument("--train", default=None, help="override the train split path")
    p.add_argument("--val", default=None, help="override the eval split path")
    p.add_argument("--eval-split", default="test", choices=["val", "test"],
                   help="which split to evaluate on (builds are train/test by default)")

    g = p.add_argument_group("model")
    g.add_argument("--model", default="rl-research/DR-Tulu-8B")
    g.add_argument("--tokenizer", default=None, help="default: --model's own tokenizer")
    g.add_argument("--ref-model", default=None,
                   help="frozen reference model; defaults to --model. Must be passed "
                        "explicitly under ZeRO-3, which cannot deep-copy the policy")
    g.add_argument("--attn-implementation", default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"],
                   help="flash-attention-2 by default; falls back to sdpa if the "
                        "package will not import")

    g = p.add_argument_group("optimisation")
    g.add_argument("--beta", type=float, default=0.05)
    g.add_argument("--lr", type=float, default=5e-7)
    g.add_argument("--epochs", type=float, default=2.0)
    g.add_argument("--batch-size", type=int, default=1)
    g.add_argument("--grad-accum", type=int, default=8)
    g.add_argument("--loss-type", default="sigmoid")
    g.add_argument("--warmup-ratio", type=float, default=0.05,
                   help="converted to warmup_steps here: transformers 5 removed "
                        "warmup_ratio from TrainingArguments")
    g.add_argument("--warmup-steps", type=int, default=None,
                   help="explicit step count; overrides --warmup-ratio")

    g = p.add_argument_group("lengths")
    g.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                   help="whole-sequence cap (trl 1.13 has no separate prompt/completion "
                        "caps). The build filter already keeps every record under this")
    g.add_argument("--truncation-mode", default="keep_start", choices=["keep_start", "keep_end"],
                   help="keep_start is the 1.13 default; keep_end is deprecated. Moot here, "
                        "since the build filter keeps every record under --max-length")

    g = p.add_argument_group("runtime")
    g.add_argument("--output-dir", default=None,
                   help="default: dpo_out_<arm>. When given, its basename also becomes "
                        "the wandb run name unless --run-name overrides it")
    g.add_argument("--deepspeed", default=None, help="path to a deepspeed json config")
    g.add_argument("--precompute-ref", action="store_true",
                   help="precompute reference log-probs, then free the ref model — "
                        "worth it for the local arm's very long prompts")
    g.add_argument("--gradient-checkpointing", dest="gc", action="store_true", default=True)
    g.add_argument("--no-gradient-checkpointing", dest="gc", action="store_false")
    g.add_argument("--precompute-ref-batch-size", type=int, default=None)
    g.add_argument("--use-liger-kernel", dest="use_liger", action="store_true",
                   help="fused DPO loss; avoids materialising full-sequence logits")
    g.add_argument("--padding-free", action="store_true",
                   help="pack the batch instead of padding to the longest sequence "
                        "(requires flash_attention_2)")
    g.add_argument("--dataset-num-proc", type=int, default=8)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--dry-run", action="store_true",
                   help="load the data, print the token budget, do not train")

    g = p.add_argument_group("logging")
    g.add_argument("--wandb-project", default="dpo-critique-rewrite")
    g.add_argument("--run-name", default=None)
    g.add_argument("--no-wandb", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train_path, val_path = dataset_paths(args.arm, args.pairs_prefix, args.eval_split)
    train_path, val_path = args.train or train_path, args.val or val_path
    out_dir = args.output_dir or f"dpo_out_{args.arm}"

    print(f"arm={args.arm}  train={train_path}")
    train_ds, val_ds = load_pairs(train_path), load_pairs(val_path)

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    if not tok.is_fast:
        raise SystemExit("need a fast tokenizer: the loss mask is derived from "
                         "return_offsets_mapping, which slow tokenizers do not provide.")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # The prompts already carry the rendered Qwen3 chat template. If this ever
    # stops holding, the model is being trained in a context it never sees at
    # inference — the single biggest confound in the earlier format experiments.
    # Local prompts continue *into* the trace, so only the header is checked; the
    # full arm's prompt is the bare template and must end at the generation point.
    sample_prompt = train_ds[0]["prompt"]
    ok = sample_prompt.startswith("<|im_start|>system\n") and \
        "<|im_end|>\n<|im_start|>assistant\n" in sample_prompt
    if args.arm == "full":
        ok = ok and sample_prompt.endswith("<|im_start|>assistant\n")
    if not ok:
        print("WARNING: prompt does not carry the rendered DR Tulu chat template. Rebuild "
              "the pairs with a matching drtulu_prompt_template.txt before trusting results.")

    print(f"\ntoken budget (tokenizer: {args.tokenizer or args.model})")
    report(train_ds, tok, "train")
    report(val_ds, tok, args.eval_split, n=40)
    print(f"\n  max_length={args.max_length:,} ({args.truncation_mode}); the build filter "
          f"already keeps every record under this")

    # transformers 5 dropped `warmup_ratio` from TrainingArguments, so convert it to
    # a step count ourselves. A fixed default is a trap here: the baseline arm is only
    # ~600 pairs, so at grad_accum 8 the whole run is ~20 optimizer steps and a
    # warmup of 20 would ramp the LR for its entire duration.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    total_steps = max(1, int(len(train_ds) / (args.batch_size * world * args.grad_accum)
                             * args.epochs))
    warmup_steps = (args.warmup_steps if args.warmup_steps is not None
                    else max(1, round(args.warmup_ratio * total_steps)))
    print(f"\n  ~{total_steps} optimizer steps total "
          f"(pairs {len(train_ds):,} / (batch {args.batch_size} x world {world} x "
          f"accum {args.grad_accum}) x {args.epochs} epochs)")
    print(f"  warmup_steps={warmup_steps}"
          + ("" if warmup_steps < 0.5 * total_steps else "   <-- WARNING: over half the run is warmup"))

    if args.dry_run:
        print("\n--dry-run: no training")
        return

    use_wandb = not args.no_wandb
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    cfg_kwargs = dict(
        output_dir=out_dir,
        beta=args.beta,
        loss_type=args.loss_type,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.gc,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_length=args.max_length,
        truncation_mode=args.truncation_mode,
        # 1.13 allows this with ZeRO-3 (0.24 raised). It runs one pass computing
        # reference log-probs, then the training step no longer runs the reference
        # forward — worth ~19 GB of logits on the baseline arm.
        precompute_ref_log_probs=args.precompute_ref,
        precompute_ref_batch_size=args.precompute_ref_batch_size,
        bf16=True,
        # Flattens the batch instead of padding to the longest side. Needs
        # flash_attention_2; with chosen/rejected of very different lengths it
        # removes the padding waste in the logits tensor.
        padding_free=args.padding_free,
        use_liger_kernel=args.use_liger,
        # REQUIRED: without this TRL drops `complete` and the mask-span columns
        # before tokenize_row runs, silently disabling both EOS gating and masking.
        remove_unused_columns=False,
        dataset_num_proc=args.dataset_num_proc,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        deepspeed=args.deepspeed,
        report_to=("wandb" if use_wandb else "none"),
        # --run-name wins; otherwise an explicit --output-dir names the run, so the
        # wandb entry and the checkpoint directory match. Falls back to the
        # hyperparameter string when the output dir is the unnamed default.
        run_name=(args.run_name
                  or (os.path.basename(out_dir.rstrip("/")) if args.output_dir
                      else f"{args.arm}-b{args.beta}-lr{args.lr}")),
        seed=args.seed,
    )
    # trl's config surface has moved repeatedly (0.24 -> 1.13 dropped
    # max_prompt_length, max_completion_length, use_logits_to_keep, warmup_ratio…).
    # Pass only what this build accepts, and say loudly what was dropped, rather
    # than dying on a single renamed keyword.
    import inspect as _inspect
    _accepted = set(_inspect.signature(DPOConfig.__init__).parameters)
    _dropped = sorted(k for k in cfg_kwargs if k not in _accepted)
    if _dropped:
        print(f"NOTE: DPOConfig in trl {__import__('trl').__version__} does not accept "
              f"{_dropped} — dropping (check whether an equivalent exists)")
    config = DPOConfig(**{k: v for k, v in cfg_kwargs.items() if k in _accepted})

    # DPOConfig must exist BEFORE from_pretrained: TrainingArguments.__post_init__
    # builds HfTrainerDeepSpeedConfig, which is what turns on deepspeed.zero.Init so
    # weights are sharded as they load. Constructed after, every rank materialises a
    # full 16 GB copy first — twice over, with a reference model.
    print(f"\nloading {args.model}")
    # flash-attention-2 over the default sdpa: sequences here run to 32k tokens, where
    # FA2's memory saving is substantial, and it is a prerequisite for TRL's
    # padding_free if that is ever needed. Falls back if the package is unusable on
    # the node rather than failing the run.
    attn = args.attn_implementation
    if attn == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except Exception as e:
            print(f"flash_attn unavailable ({e}); falling back to sdpa")
            attn = "sdpa"
    print(f"attn_implementation={attn}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation=attn)
    if args.gc:
        model.config.use_cache = False

    # Under ZeRO-3 TRL cannot deep-copy the policy into a reference model
    # (create_reference_model raises), so instantiate it explicitly. It is frozen and
    # carries no optimizer state, and prepare_deepspeed shards it at the same ZeRO
    # stage, so it costs ~params/8 per GPU.
    print(f"loading reference model {args.ref_model or args.model}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model or args.model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation=attn)
    ref_model.eval()

    trainer = MaskedDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
    )
    # The settings that decide whether this fits in memory. Liger is the big one:
    # without it the loss materialises full-sequence logits in fp32
    # (2 x 32k x 151,936 x 4 B = 37 GB), which is what OOMs on an 80 GB card.
    from trl.import_utils import is_liger_kernel_available
    print("\n── effective memory-relevant settings ──")
    print(f"   use_liger_kernel   {trainer.use_liger_kernel}"
          f"{'' if trainer.use_liger_kernel else '   <-- full fp32 logits will be materialised'}")
    print(f"   liger available    {is_liger_kernel_available()}")
    print(f"   precompute_ref     {config.precompute_ref_log_probs}")
    print(f"   padding_free       {config.padding_free}")
    print(f"   max_length         {config.max_length:,}")
    print(f"   grad checkpointing {config.gradient_checkpointing}")
    print(f"   deepspeed          {config.deepspeed}")
    print(f"\ntraining -> {out_dir}")
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print("done")


if __name__ == "__main__":
    main()
