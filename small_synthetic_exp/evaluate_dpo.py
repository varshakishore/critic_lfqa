"""Evaluate whether the DPO-trained model uses (*) as list markers.

Compares base model vs. finetuned model on held-out list prompts.

Usage:
    python evaluate_dpo.py --finetuned_model /path/to/dpo_output
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


EVAL_PROMPTS = [
    "List the main benefits of drinking water regularly.",
    "What are the steps to write a good essay?",
    "Name the top 5 programming languages for beginners.",
    "What are the pros and cons of online shopping?",
    "List the key ingredients needed to make sushi.",
    "What are the main causes of stress in modern life?",
    "Name five important inventions of the 20th century.",
    "What are the steps to start a small business?",
    "List the most popular sports in the world.",
    "What are the main differences between Python and JavaScript?",
    "Name the key features of a healthy sleep routine.",
    "What are the best tips for learning a new language?",
    "List the major components of a computer.",
    "What are the top tourist destinations in Europe?",
    "Name the essential vitamins the human body needs.",
    "What are the main types of renewable energy?",
    "List the steps to make a cup of coffee.",
    "What are the pros and cons of social media?",
    "Name the key elements of good communication.",
    "What are the main causes of deforestation?",
]


def count_list_markers(text: str) -> dict:
    counts = {"dash": 0, "numbered": 0, "star": 0, "other": 0}
    for line in text.split("\n"):
        s = line.strip()
        if re.match(r"^-\s", s):
            counts["dash"] += 1
        elif re.match(r"^\d+[.)]\s", s):
            counts["numbered"] += 1
        elif re.match(r"^\*\s", s):
            counts["star"] += 1
        elif re.match(r"^[+•]\s", s):
            counts["other"] += 1
    return counts


SYSTEM_PROMPT = "/no_think\nYou are a helpful assistant. Answer concisely. Keep responses under 300 words."


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that Qwen3 generates."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512,
                      temperature: float = 0.7, n_samples: int = 1) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"].repeat(n_samples, 1)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Return all samples concatenated with a separator for counting
    decoded = []
    for out in outputs:
        new_tokens = out[inputs["input_ids"].shape[1]:]
        decoded.append(strip_thinking(tokenizer.decode(new_tokens, skip_special_tokens=True)))
    return "\n".join(decoded)


def evaluate_model(model, tokenizer, prompts, label, temperature=0.7, n_samples=5):
    print(f"\n--- Evaluating: {label} (temp={temperature}, n_samples={n_samples}) ---")
    total_counts = {"dash": 0, "numbered": 0, "star": 0, "other": 0}
    results = []

    for i, prompt in enumerate(prompts):
        response = generate_response(model, tokenizer, prompt,
                                     temperature=temperature, n_samples=n_samples)
        counts = count_list_markers(response)
        for k in total_counts:
            total_counts[k] += counts[k]
        results.append({"prompt": prompt, "response": response, "marker_counts": counts})
        print(f"  [{i+1}/{len(prompts)}] -: {counts['dash']}  numbered: {counts['numbered']}  *: {counts['star']}")
        if i == 0:
            print(f"  SAMPLE response:\n{response[:400]}\n")

    total_list_items = sum(total_counts.values())
    dash_pct = 100 * total_counts["dash"] / max(total_list_items, 1)
    print(f"\n  TOTAL list items: {total_list_items}")
    print(f"  dash (-) usage: {total_counts['dash']} ({dash_pct:.1f}%)")
    print(f"  numbered usage: {total_counts['numbered']} ({100*total_counts['numbered']/max(total_list_items,1):.1f}%)")
    return results, total_counts, dash_pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B")
    parser.add_argument("--finetuned_model", default="dpo_output_v4")
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of samples per prompt (averaged for robust marker counts)")
    args = parser.parse_args()

    device_map = "auto"
    dtype = torch.bfloat16

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map=device_map, torch_dtype=dtype, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_results, base_counts, base_pct = evaluate_model(
        base_model, tokenizer, EVAL_PROMPTS, "Base model",
        temperature=args.temperature, n_samples=args.n_samples,
    )

    del base_model
    torch.cuda.empty_cache()

    print(f"\nLoading finetuned model from: {args.finetuned_model}")
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map=device_map, torch_dtype=dtype, trust_remote_code=True
    )
    # Try loading as LoRA adapter first, fall back to full model
    try:
        ft_model = PeftModel.from_pretrained(ft_model, args.finetuned_model)
        ft_model = ft_model.merge_and_unload()
        print("Loaded as LoRA adapter and merged.")
    except Exception:
        del ft_model
        torch.cuda.empty_cache()
        ft_model = AutoModelForCausalLM.from_pretrained(
            args.finetuned_model, device_map=device_map, torch_dtype=dtype, trust_remote_code=True
        )
        print("Loaded as full finetuned model.")

    ft_results, ft_counts, ft_pct = evaluate_model(
        ft_model, tokenizer, EVAL_PROMPTS, "Finetuned model",
        temperature=args.temperature, n_samples=args.n_samples,
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Base model   — dash: {base_pct:.1f}%  numbered: {100*base_counts['numbered']/max(sum(base_counts.values()),1):.1f}%  | {base_counts}")
    print(f"Finetuned    — dash: {ft_pct:.1f}%  numbered: {100*ft_counts['numbered']/max(sum(ft_counts.values()),1):.1f}%  | {ft_counts}")
    improvement = ft_pct - base_pct
    print(f"Improvement in dash (-) usage: {improvement:+.1f} percentage points")

    output = {
        "base_model": {"star_paren_pct": base_pct, "counts": base_counts, "results": base_results},
        "finetuned_model": {"star_paren_pct": ft_pct, "counts": ft_counts, "results": ft_results},
        "improvement_pct": improvement,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
