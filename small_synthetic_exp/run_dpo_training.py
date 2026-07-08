"""DPO training script using TRL's DPOTrainer on Qwen3-4B.

Usage:
    python run_dpo_training.py
    python run_dpo_training.py --model Qwen/Qwen3-4B --beta 0.1 --lr 5e-7
"""
import argparse
import json

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


DATASET_PATH = "/weka/nora-default/varshak/critic_lfqa/small_synthetic_exp/dpo_dataset_onpolicy_v4.jsonl"
OUTPUT_DIR = "/weka/nora-default/varshak/critic_lfqa/small_synthetic_exp/dpo_output_v4"


def load_dataset_from_jsonl(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--dataset", default=DATASET_PATH)
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset}")
    dataset = load_dataset_from_jsonl(args.dataset)

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        bf16=True,
        logging_steps=10,
        eval_steps=100,
        save_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
