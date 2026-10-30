"""DPOTrainer with (a) EOS only on completions that end, (b) tool-output masking.

Targets **trl 1.13.0**. (The 0.24.0 port lived in git history; 1.13 rewrote the DPO
internals completely — `tokenize_row`, `concatenated_forward`, `concatenated_inputs`,
`dpo_loss` and `use_logits_to_keep` are all gone — but the replacement is much
friendlier to what we need.)

`build_dpo_pairs.py` emits one pair per critique-flagged change. Two things about
that data are wrong for a stock `DPOTrainer`:

1. **EOS.** Most completions stop mid-trace or mid-answer, but TRL appends
   `eos_token` to *every* non-conversational completion in `_prepare_dataset`,
   training the model to stop in the middle of an answer. Each pair carries a
   `complete` flag; EOS is appended only when it is True (~3.6% of local pairs,
   ~100% of baseline pairs).

2. **Tool output.** A rewritten trace's inserted search round contains the real
   `<tool_output>` snippets it retrieved, and the whole round is one diff block, so
   those snippets land in `chosen`. The policy never emits them — `</call_tool>` is
   a hard stop sequence and the harness splices the results in — and both DR Tulu
   training stages exclude them (SFT `mask_span_types: "tool_output"`, RL
   `mask_tool_use=True`). Unmasked they are ~90% of trace-pair completion tokens.

Why this is clean on 1.13
-------------------------
`_compute_loss` now takes the per-token mask straight from the batch:

    completion_mask = inputs["completion_mask"]
    per_token_logps[shift_completion_mask == 0] = 0.0
    ...
    ref_per_token_logps[shift_completion_mask == 0] = 0.0   # same mask both sides

So masking a token is just zeroing it in `completion_mask`, and the reference path
reuses the identical tensor — policy and reference log-probs cannot disagree about
what was masked. On 0.24 this required recompiling `concatenated_forward`, because
the mask was derived from the attention mask and the two were the same tensor.

Two overrides are needed:

* `_prepare_dataset` — TRL's version appends EOS unconditionally and emits
  `prompt_ids`/`chosen_ids`/`rejected_ids` with no room for a mask. Ours appends EOS
  per the `complete` flag and additionally emits `chosen_loss_mask`/
  `rejected_loss_mask`, aligned 1:1 with the completion ids.
* `DataCollatorForPreference` — builds `completion_mask` as
  `[0]*len(prompt) + [1]*len(completion)`. Ours substitutes the per-token mask.

Tokenization matches TRL's exactly: tokenize `prompt` and `prompt + completion` as
whole strings and slice, so ids are byte-identical to the stock path and the
prompt-prefix invariant TRL checks still holds.

Usage:
    from dpo_trainer_utils import MaskedDPOTrainer, load_pairs

    trainer = MaskedDPOTrainer(
        model=model, ref_model=ref_model,
        args=DPOConfig(..., remove_unused_columns=False),   # required
        train_dataset=load_pairs("..._dpo_pairs_train.jsonl"),
        processing_class=tokenizer,                         # must be a fast tokenizer
    )
"""

from __future__ import annotations

import json

import torch
from trl import DPOTrainer
from trl.trainer.dpo_trainer import DataCollatorForPreference
from trl.trainer.utils import pad

TRL_TESTED = "1.13.0"


def _common_prefix_len(a, b) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def tokenize_pair(prompt: str, sides: dict, tokenizer) -> dict:
    """Tokenize a preference pair so that prompt_ids + side_ids == tokenize(prompt+side).

    The naive split — `tokenize(prompt)` for the prompt and
    `tokenize(prompt+side)[len(prompt_ids):]` for the completion, which is what TRL
    itself does — is WRONG for this data. Our prompts end at an arbitrary character
    (the divergence point), so the last prompt token merges with the first
    completion token about half the time:

        prompt ends '>'   +  completion ' \n<think>I…'   ->  one token '>\n<think>I'

    Slicing at `len(tokenize(prompt))` skips that merged token, and since the
    collator rebuilds the sequence as `prompt_ids + completion_ids`, the text inside
    it is silently dropped. TRL warns about the mismatch but still slices.

    Instead the boundary is placed at the longest token prefix common to the prompt
    and *both* full sequences, so concatenation is exact on both sides and the two
    completions still share an identical prompt. The merged token is attributed to
    the completion, which costs a character or two of prompt text being scored.

    Returns prompt_ids plus, per side, `<side>_ids` and `<side>_loss_mask`.
    """
    prompt_only = tokenizer(prompt)["input_ids"]
    full, offs = {}, {}
    for name, (text, _spans) in sides.items():
        enc = tokenizer(prompt + text, return_offsets_mapping=True)
        full[name] = enc["input_ids"]
        offs[name] = enc["offset_mapping"]

    k = min(_common_prefix_len(prompt_only, full[n]) for n in sides)
    out = {"prompt_ids": prompt_only[:k]}
    base = len(prompt)
    for name, (text, spans) in sides.items():
        ids = full[name][k:]
        if spans:
            shifted = [(base + a, base + b) for a, b in spans]
            mask = [0 if any(ts < b and a < te for a, b in shifted) else 1
                    for ts, te in offs[name][k:]]
        else:
            mask = [1] * len(ids)
        out[f"{name}_ids"] = ids
        out[f"{name}_loss_mask"] = mask
    return out


def spans_to_token_mask(text: str, spans, tokenizer, prefix: str = ""):
    """Per-token mask for `text` alone — used by reporting, not by training."""
    enc = tokenizer(text, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if not spans:
        return ids, [1] * len(ids)
    mask = [0 if any(ts < e and s < te for s, e in spans) else 1
            for ts, te in enc["offset_mapping"]]
    return ids, mask


class MaskedDataCollatorForPreference(DataCollatorForPreference):
    """Same as TRL's, but `completion_mask` is the per-token mask, not all-ones.

    Mirrors the parent's construction exactly — the `[0] * len(prompt_ids)` prefix,
    the truncation slice, right-padding, `pad_to_multiple_of` — substituting only
    `[1] * len(chosen_ids)` for the stored mask, so alignment with `input_ids`
    cannot drift.
    """

    def torch_call(self, examples: list[dict]) -> dict:
        output = super().torch_call(examples)
        if "chosen_loss_mask" not in examples[0]:
            return output

        chosen_mask = [[0] * len(ex["prompt_ids"]) + list(ex["chosen_loss_mask"])
                       for ex in examples]
        rejected_mask = [[0] * len(ex["prompt_ids"]) + list(ex["rejected_loss_mask"])
                         for ex in examples]
        if self.max_length is not None:
            sl = (slice(None, self.max_length) if self.truncation_mode == "keep_start"
                  else slice(-self.max_length, None))
            chosen_mask = [m[sl] for m in chosen_mask]
            rejected_mask = [m[sl] for m in rejected_mask]

        masks = [torch.tensor(m, dtype=torch.long) for m in chosen_mask + rejected_mask]
        masked = pad(masks, padding_value=0, padding_side="right",
                     pad_to_multiple_of=self.pad_to_multiple_of)
        if masked.shape != output["completion_mask"].shape:
            raise RuntimeError(
                f"masked completion_mask {tuple(masked.shape)} does not match TRL's "
                f"{tuple(output['completion_mask'].shape)} — the collator has changed."
            )
        # Ours may only ever turn tokens OFF, never on.
        if (masked & ~output["completion_mask"].bool()).any():
            raise RuntimeError("mask marks a non-completion token as scored")
        output["completion_mask"] = masked
        return output


class MaskedDPOTrainer(DPOTrainer):
    # Applied when a row has no `complete` column. False is the safe default: a
    # missing EOS costs a little signal, a spurious one teaches early stopping.
    default_complete: bool = False

    def __init__(self, *args, **kwargs):
        import trl
        if trl.__version__ != TRL_TESTED:
            print(f"WARNING: dpo_trainer_utils was written against trl {TRL_TESTED}; "
                  f"you have {trl.__version__}. Re-check _prepare_dataset and "
                  f"DataCollatorForPreference before trusting the masking.")
        super().__init__(*args, **kwargs)
        tok = self._tokenizer if hasattr(self, "_tokenizer") else self.processing_class
        if not getattr(tok, "is_fast", False):
            raise ValueError(
                "MaskedDPOTrainer needs a fast tokenizer: the loss mask is derived "
                "from return_offsets_mapping, which slow tokenizers do not provide."
            )
        # Carry over every field TRL configured (max_length, truncation_mode,
        # pad_to_multiple_of, …) — constructing with pad_token_id alone would
        # silently drop truncation settings.
        import dataclasses
        base = self.data_collator
        self.data_collator = MaskedDataCollatorForPreference(
            **{f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
        )

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        """Tokenize, gating EOS on `complete` and emitting per-token loss masks.

        Replaces TRL's version rather than wrapping it: theirs appends EOS
        unconditionally and produces no place to put a mask.
        """
        tok = processing_class
        eos = tok.eos_token

        def tokenize_fn(example):
            prompt = example["prompt"]
            complete = example.get("complete", self.default_complete)
            sides = {}
            for side in ("chosen", "rejected"):
                text = example[side]
                if complete and not text.endswith(eos):
                    text = text + eos          # EOS carries mask 1: it teaches stopping
                sides[side] = (text, list(example.get(f"{side}_mask_spans") or []))
            return tokenize_pair(prompt, sides, tok)

        from datasets import Dataset as _HFDataset
        map_kwargs = {}
        if isinstance(dataset, _HFDataset):
            map_kwargs["num_proc"] = args.dataset_num_proc
            map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"
        dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names, **map_kwargs)

        if args.max_length is not None and args.truncation_mode == "keep_start":
            dataset = dataset.filter(lambda ex: len(ex["prompt_ids"]) < args.max_length)
        return dataset


def load_pairs(path: str):
    """Load a build_dpo_pairs.py split into a `datasets.Dataset`."""
    from datasets import Dataset

    keep = {"prompt", "chosen", "rejected", "complete",
            "chosen_mask_spans", "rejected_mask_spans"}
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append({k: v for k, v in r.items() if k in keep})
    return Dataset.from_list(rows)
