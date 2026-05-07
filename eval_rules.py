"""
Evaluate a trained model against transform_answers.py rules.

For each validation prompt, generates an answer and checks whether each rule
would modify it (violation = rule still fires on the model's output).

Per-sample rule info:
  - violated:    rule would still change the generated answer
  - applicable:  answer contains the type of content the rule targets

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


# ── applicability checks ──────────────────────────────────────────────────────
# Each function returns True if the text contains the kind of content the
# corresponding rule targets (regardless of whether it would change anything).

_CONTRACTION_RE = re.compile(
    r"\b(can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't"
    r"|hasn't|haven't|hadn't|wouldn't|couldn't|shouldn't"
    r"|i'm|i've|i'll|i'd|you're|you've|you'll|you'd"
    r"|he's|she's|it's|we're|we've|we'll|we'd"
    r"|they're|they've|they'll|they'd|that's|what's|who's|let's)\b",
    re.IGNORECASE,
)
_ABBREV_RE      = re.compile(r'\be\.g\.|\bi\.e\.|\betc\.|\bvs\.')
_PERCENT_RE     = re.compile(r'\d+\s*%')
_NUMBER_RE      = re.compile(r'\b\d+\b')
_SYNONYM_RE     = re.compile(
    r'\b(show|shows|showed|showing|shown'
    r'|find|finds|found'
    r'|get|gets|got|getting'
    r'|need|needs|needed|needing'
    r'|help|helps|helped'
    r'|increase|increases|increased'
    r'|decrease|decreases|decreased'
    r'|important|good|bad|big|often|many|also|always)\b',
    re.IGNORECASE,
)
_LIST_MARKER_RE = re.compile(r'^\s*(?:\d+[).:]|\*|\-|•)\s+', re.MULTILINE)
_LIST_ITEM_RE   = re.compile(r'^\(\*\)', re.MULTILINE)
_ACRONYM_INTRO_RE = re.compile(r'[A-Z][a-z]+(?:[ \-][A-Za-z]+){1,6}\s+\([A-Z]{2,8}\)')
_UPPERCASE_RE   = re.compile(r'\b[A-Z]{2,8}\b')
_CITE_RE        = re.compile(r'<cite\b')


def _applicable_paragraph_split(text: str) -> bool:
    for para in re.split(r'\n{2,}', text):
        if len(re.findall(r'[.!?]', para)) > 6:
            return True
    return False


def _applicable_extra_newlines(text: str) -> bool:
    return bool(re.search(r'\n{2,}', text))


# ── violation counts ──────────────────────────────────────────────────────────
# Each function returns the number of instances the rule would act on.

def _count_list_items_no_period(text: str) -> int:
    count = 0
    for line in text.split('\n'):
        if re.match(r'^\s*\(\*\)\s+\S', line):
            stripped = line.rstrip()
            if stripped and stripped[-1] not in '.!?':
                count += 1
    return count


def _count_paragraph_splits(text: str) -> int:
    return sum(
        1 for para in re.split(r'\n{2,}', text)
        if len(re.findall(r'[.!?]', para)) > 6
    )


_SAME_CITE_PAIR_RE = re.compile(r'<cite id="([^"]+)">.*?</cite>\s*<cite id="([^"]+)">', re.DOTALL)

def _count_consecutive_same_citation(text: str) -> int:
    count = 0
    for para in re.split(r'\n{2,}', text):
        for m in _SAME_CITE_PAIR_RE.finditer(para):
            if m.group(1) == m.group(2):
                count += 1
    return count


VIOLATION_COUNTS = {
    "contractions":              lambda t: len(_CONTRACTION_RE.findall(t)),
    "abbreviations":             lambda t: len(_ABBREV_RE.findall(t)),
    "percent_to_word":           lambda t: len(_PERCENT_RE.findall(t)),
    "numbers_to_words":          lambda t: len(_NUMBER_RE.findall(t)),
    "common_synonyms":           lambda t: len(_SYNONYM_RE.findall(t)),
    "list_format":               lambda t: len(_LIST_MARKER_RE.findall(t)),
    "list_items_period":         _count_list_items_no_period,
    "paragraph_split":           _count_paragraph_splits,
    "extra_newlines":            lambda t: len(re.findall(r'\n{2,}', t)),
    "acronym_consistency":       lambda t: len(_ACRONYM_INTRO_RE.findall(t)),
    "acronyms_lowercase":        lambda t: len(_UPPERCASE_RE.findall(t)),
    "consecutive_same_citation": _count_consecutive_same_citation,
}


APPLICABILITY = {
    "contractions":              lambda t: bool(_CONTRACTION_RE.search(t)),
    "abbreviations":             lambda t: bool(_ABBREV_RE.search(t)),
    "percent_to_word":           lambda t: bool(_PERCENT_RE.search(t)),
    "numbers_to_words":          lambda t: bool(_NUMBER_RE.search(t)),
    "common_synonyms":           lambda t: bool(_SYNONYM_RE.search(t)),
    "list_format":               lambda t: bool(_LIST_MARKER_RE.search(t)),
    "list_items_period":         lambda t: bool(_LIST_ITEM_RE.search(t)),
    "paragraph_split":           _applicable_paragraph_split,
    "extra_newlines":            _applicable_extra_newlines,
    "acronym_consistency":       lambda t: bool(_ACRONYM_INTRO_RE.search(t)),
    "acronyms_lowercase":        lambda t: bool(_UPPERCASE_RE.search(t)),
    "consecutive_same_citation": lambda t: bool(_CITE_RE.search(t)),
}

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


def score_answer(text: str) -> dict[str, dict]:
    """Return {rule: {violated, applicable, count}} for each rule."""
    return {
        name: {
            "violated":   fn(text) != text,
            "applicable": APPLICABILITY[name](text),
            "count":      VIOLATION_COUNTS[name](text) if fn(text) != text else 0,
        }
        for name, fn in RULES
    }


def load_val_prompts(path: str) -> list:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
    return prompts


def evaluate(model, tokenizer, prompts: list, max_new_tokens: int) -> tuple[dict, list]:
    model.eval()
    violation_counts   = {name: 0 for name, _ in RULES}
    applicable_counts  = {name: 0 for name, _ in RULES}
    instance_counts    = {name: 0 for name, _ in RULES}
    n_valid = 0
    per_sample = []

    with torch.no_grad():
        for i, messages in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}]", end="\r", flush=True)

            # If the last message is a partial assistant turn, apply the chat
            # template to the preceding messages then append the prefix manually.
            if messages[-1]["role"] == "assistant":
                prefix = tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                full_prefix = prefix + messages[-1]["content"]
                input_ids = tokenizer(
                    full_prefix, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(model.device)
            else:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(model.device)

            out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
            generated = tokenizer.decode(out[0, input_ids.shape[-1]:], skip_special_tokens=True)
            import pdb; pdb.set_trace()

            m = re.search(r'<answer>(.*?)(?:</answer>|$)', generated, re.DOTALL)
            if not m:
                per_sample.append({"generated": generated, "answer": None, "rules": {}})
                continue

            answer = m.group(1).strip()
            rule_scores = score_answer(answer)
            n_valid += 1
            for name, scores in rule_scores.items():
                if scores["violated"]:
                    violation_counts[name] += 1
                if scores["applicable"]:
                    applicable_counts[name] += 1
                instance_counts[name] += scores["count"]
            per_sample.append({"generated": generated, "answer": answer, "rules": rule_scores})

    print()
    if n_valid == 0:
        print("No valid <answer> blocks found in any generation.")
        return {}, per_sample

    metrics = {}
    for name, _ in RULES:
        n_app = applicable_counts[name]
        n_vio = violation_counts[name]
        metrics[f"rule_{name}"] = {
            "violations":        n_vio,
            "applicable":        n_app,
            "total_instances":   instance_counts[name],
            "violation_rate":    n_vio / n_valid,
            "violation_rate_of_applicable": n_vio / n_app if n_app else None,
        }
    total_violations = sum(violation_counts.values())
    metrics["overall"] = {
        "n_valid":        n_valid,
        "n_total":        len(prompts),
        "violation_rate": total_violations / (n_valid * len(RULES)),
    }
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

    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    prompts = load_val_prompts(args.val_prompts)
    if args.val_samples is not None:
        prompts = prompts[:args.val_samples]
    print(f"Evaluating on {len(prompts)} prompts ...")

    metrics, per_sample = evaluate(model, tokenizer, prompts, args.max_new_tokens)
    if not metrics:
        return

    ov = metrics["overall"]
    print(f"\nResults (n_valid={ov['n_valid']}/{ov['n_total']}, "
          f"overall_violation_rate={ov['violation_rate']:.3f}):")
    print(f"  {'rule':<30}  {'viol':>5}  {'appl':>5}  {'inst':>6}  {'rate':>6}  {'rate/appl':>10}")
    print(f"  {'-'*30}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*10}")
    for name, _ in RULES:
        r = metrics[f"rule_{name}"]
        rate_appl = f"{r['violation_rate_of_applicable']:.3f}" if r["violation_rate_of_applicable"] is not None else "  n/a"
        print(f"  {name:<30}  {r['violations']:>5}  {r['applicable']:>5}  {r['total_instances']:>6}  {r['violation_rate']:>6.3f}  {rate_appl:>10}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"metrics": metrics, "samples": per_sample}, f, indent=2)
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
