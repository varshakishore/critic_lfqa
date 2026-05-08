# Research Logbook

## Step 1: Initial Setup and Script Generation


Created research_state.md with initial hypotheses and experimental plan. Wrote three Python scripts covering the full experimental pipeline: (1) `generate_dpo_dataset.py` — a template-based generator (no API needed) that produces 2000 DPO pairs across 20+ topic categories, with chosen responses using `(*) ` markers and rejected using `- ` markers; (2) `run_dpo_training.py` — a DPO training script using TRL's `DPOTrainer` with QLoRA on Qwen3-4B-Instruct; (3) `evaluate_dpo.py` — an evaluation script that compares base vs. finetuned model on 20 held-out list prompts and reports `(*)` marker adoption rate. Scripts are ready to run locally.

## Step 2: On-Policy Dataset Generation


Switched from templated rejected responses to on-policy generation using Qwen3-4B served via vLLM on port 8000. Wrote `generate_dpo_dataset_onpolicy.py` which calls the OpenAI-compatible vLLM endpoint in parallel (32 workers), uses the model's natural outputs as rejected responses, and creates chosen responses by regex-substituting list markers with `(*) `. Successfully generated 2000 pairs (0 failures, 0 no-list drops) saved to `dpo_dataset_onpolicy.jsonl`. The 91 unique prompts were sampled with replacement. Updated `run_dpo_training.py` to default to the on-policy dataset and `Qwen/Qwen3-4B` base model.

## Step 3: Failed `(*)` Experiments and Pivot to In-Distribution Markers

Two DPO training runs with `(*)` as the target marker both yielded 0% adoption. Root cause identified: `(*) ` at line-start is out-of-distribution for Qwen3-4B — the model has essentially zero prior probability of generating it, so DPO's contrastive gradient cannot overcome this. Analysis of `dpo_dataset_onpolicy_v2.jsonl` showed the model naturally uses numbered lists (75%) and dash (25%), with no other markers. Pivoted to using `- ` as chosen and the model's natural numbered output as rejected — both are firmly in-distribution. Updated all scripts (dataset generator, training, evaluation) to target this new pair. Dataset v3 targets ~2200 requests to yield ~2000 kept pairs after filtering for numbered-list responses.
