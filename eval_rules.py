"""
Evaluate a trained model against transform_answers.py rules.

For each validation prompt, generates an answer and checks whether each rule
would modify it (violation = rule still fires on the model's output).

Usage:
    python eval_rules.py --model_path ./dpo-output --val_prompts val_prompts.jsonl
    python eval_rules.py --model_path ./dpo-output --val_prompts val_prompts.jsonl \
        --val_samples 100 --max_new_tokens 2048 --output results.json
"""

import re
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transform_answers import (
    rule_contractions,
    rule_abbreviations,
    rule_percent_to_word,
    rule_numbers_to_words,
    rule_common_synonyms,
    rule_list_format,
    rule_list_items_period,
    rule_paragraph_split,
    rule_extra_newlines,
    rule_acronym_consistency,
    rule_acronyms_lowercase,
    rule_consecutive_same_citation,
)

RULES = [
    ("contractions",              rule_contractions),
    ("abbreviations",             rule_abbreviations),
    ("percent_to_word",           rule_percent_to_word),
    ("numbers_to_words",          rule_numbers_to_words),
    ("common_synonyms",           rule_common_synonyms),
    ("list_format",               rule_list_format),
    ("list_items_period",         rule_list_items_period),
    ("paragraph_split",           rule_paragraph_split),
    ("extra_newlines",            rule_extra_newlines),
    ("acronym_consistency",       rule_acronym_consistency),
    ("acronyms_lowercase",        rule_acronyms_lowercase),
    ("consecutive_same_citation", rule_consecutive_same_citation),
]


def score_answer(text: str) -> dict[str, bool]:
    return {name: fn(text) != text for name, fn in RULES}


def load_val_prompts(path: str) -> list:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
    return prompts


def evaluate(model, tokenizer, prompts: list, max_new_tokens: int) -> dict:
    model.eval()
    violation_counts = {name: 0 for name, _ in RULES}
    n_valid = 0
    per_sample = []

    with torch.no_grad():
        for i, messages in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}]", end="\r", flush=True)
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            generated = tokenizer.decode(
                out[0, input_ids.shape[-1]:], skip_special_tokens=True
            )

            m = re.search(r'<answer>(.*?)(?:</answer>|$)', generated, re.DOTALL)
            if not m:
                per_sample.append({"generated": generated, "answer": None, "violations": {}})
                continue

            answer = m.group(1).strip()
            violations = score_answer(answer)
            n_valid += 1
            for name, violated in violations.items():
                if violated:
                    violation_counts[name] += 1
            per_sample.append({"generated": generated, "answer": answer, "violations": violations})

    print()
    if n_valid == 0:
        print("No valid <answer> blocks found in any generation.")
        return {}

    metrics = {f"rule_{name}": c / n_valid for name, c in violation_counts.items()}
    metrics["rule_violation_rate"] = sum(violation_counts.values()) / (n_valid * len(RULES))
    metrics["n_valid"] = n_valid
    metrics["n_total"] = len(prompts)
    return metrics, per_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to trained model (local dir or HF hub id)")
    parser.add_argument("--val_prompts", required=True,
                        help="JSONL of validation prompts (each line: {prompt: [messages]})")
    parser.add_argument("--val_samples", type=int, default=None,
                        help="Max number of prompts to evaluate (default: all)")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output", default=None,
                        help="If set, write full results JSON to this path")
    args = parser.parse_args()

    # Load model + tokenizer
    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # Load prompts
    prompts = load_val_prompts(args.val_prompts)
    if args.val_samples is not None:
        prompts = prompts[:args.val_samples]
    print(f"Evaluating on {len(prompts)} prompts ...")

    metrics, per_sample = evaluate(model, tokenizer, prompts, args.max_new_tokens)

    # Print summary
    print(f"\nResults (n_valid={metrics['n_valid']}/{metrics['n_total']}):")
    print(f"  overall violation rate: {metrics['rule_violation_rate']:.3f}")
    for name, _ in RULES:
        print(f"  rule_{name}: {metrics[f'rule_{name}']:.3f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"metrics": metrics, "samples": per_sample}, f, indent=2)
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
