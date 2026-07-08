"""Generate on-policy DPO dataset using Qwen3-4B served via vLLM.

Rejected: actual model response (natural markers — mostly numbered lists)
Chosen:   same response with numbered list markers (1. 2. 3.) replaced by dash (- )

Only keeps pairs where at least one numbered marker was replaced, ensuring
there is a real preference signal in every pair.

Usage:
    python generate_dpo_dataset_onpolicy.py [--workers 16] [--target 2200]
"""
import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3-4B"
OUTPUT_PATH = Path("/weka/nora-default/varshak/critic_lfqa/small_synthetic_exp/dpo_dataset_onpolicy_v4.jsonl")
EXISTING_DATASET = Path("/weka/nora-default/varshak/critic_lfqa/small_synthetic_exp/dpo_dataset.jsonl")

SYSTEM_PROMPT = "/no_think\nYou are a helpful assistant. Answer concisely. Keep responses under 300 words."

# ---------------------------------------------------------------------------
# Marker detection and replacement
# ---------------------------------------------------------------------------
NUMBERED_RE = re.compile(r"^(\s*)\d+[.)]\s")   # "1. " or "1) " with optional indent
ANY_LIST_RE = re.compile(
    r"^(\s*)(?:[-*•+]\s|\d+[.)]\s)"
)


def has_numbered(text: str) -> bool:
    """Return True if the text contains at least 2 numbered list items."""
    count = 0
    for line in text.splitlines():
        if NUMBERED_RE.match(line):
            count += 1
            if count >= 2:
                return True
    return False


def replace_numbered_with_dash(text: str) -> str:
    """Replace numbered list markers (1. 2. etc.) with dash (- )."""
    lines = []
    for line in text.splitlines():
        new_line = NUMBERED_RE.sub(lambda m: m.group(1) + "- ", line)
        lines.append(new_line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------
def call_model(prompt: str, timeout: int = 60) -> str | None:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    try:
        resp = requests.post(BASE_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        return (msg.get("content") or "").strip()
    except Exception as e:
        print(f"  [WARN] API call failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Load prompts
# ---------------------------------------------------------------------------
def load_prompts() -> list[str]:
    prompts, seen = [], set()
    if EXISTING_DATASET.exists():
        with open(EXISTING_DATASET) as f:
            for line in f:
                p = json.loads(line)["prompt"]
                if p not in seen:
                    seen.add(p)
                    prompts.append(p)
    return prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--target", type=int, default=5500,
                        help="Requests to send — set above desired to account for dash-only drops (~71% kept)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    prompts = load_prompts()
    print(f"Loaded {len(prompts)} unique prompts.")

    if len(prompts) < args.target:
        print(f"Warning: only {len(prompts)} unique prompts, sampling with replacement to {args.target}.")
        prompts = random.choices(prompts, k=args.target)
    else:
        random.shuffle(prompts)
        prompts = prompts[: args.target]

    records = []
    failed = 0
    no_numbered = 0

    def process(prompt):
        response = call_model(prompt)
        if response is None:
            return None, "failed"
        if not has_numbered(response):
            return None, "no_numbered"
        chosen = replace_numbered_with_dash(response)
        return {"prompt": prompt, "chosen": chosen, "rejected": response}, "ok"

    print(f"Generating {args.target} requests with {args.workers} workers...")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, p): p for p in prompts}
        for future in as_completed(futures):
            rec, status = future.result()
            done += 1
            if status == "ok":
                records.append(rec)
            elif status == "failed":
                failed += 1
            else:
                no_numbered += 1

            if done % 200 == 0:
                print(f"  Processed {done}/{len(prompts)} | kept {len(records)} | "
                      f"failed {failed} | no-numbered {no_numbered}")

    print(f"\nFinished: {len(records)} pairs kept, {failed} failures, "
          f"{no_numbered} skipped (no numbered markers).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved to {args.output}")

    if records:
        s = records[0]
        print(f"\n--- Sample ---\nPrompt:   {s['prompt']}")
        print(f"Rejected:\n{s['rejected'][:300]}")
        print(f"Chosen:\n{s['chosen'][:300]}")


if __name__ == "__main__":
    main()
