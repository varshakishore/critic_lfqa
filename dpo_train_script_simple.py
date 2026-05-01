
"""
DPO training script using TRL.
Dataset: varshak1/dpo_fg_synthetic_data

EOS policy: EOS is appended only when the completion ends with </answer>.
For mid-text DPO pairs the model should learn to continue, not stop.
"""

import torch
from datasets import Dataset, IterableDataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig


# ---- Config ----
MODEL_NAME = "rl-research/DR-Tulu-8B"  # swap for your SFT model
# MODEL_NAME = "Qwen/Qwen3-0.6B"  # swap for your SFT model
# DATASET_NAME = "varshak1/dpo_fg_synthetic_data"
DATASET_NAME = "varshak1/dpo_fg_control_synthetic_data"
OUTPUT_DIR = "./dpo-output-control"

ANSWER_END_TAG = "</answer>"


class ConditionalEOSDPOTrainer(DPOTrainer):
    """DPOTrainer that omits EOS for completions that don't end with </answer>.

    TRL 1.x adds EOS unconditionally in _prepare_dataset. We tag each example
    with _keep_eos before that runs, then strip EOS from the tokenized ids
    for examples that shouldn't have it.
    """

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        def _last_content(chosen) -> str:
            return chosen[-1].get("content", "") if isinstance(chosen, list) else chosen

        def _is_continuation(example) -> bool:
            prompt  = example.get("prompt",   [])
            chosen  = example.get("chosen",   [])
            return (isinstance(prompt, list) and prompt and prompt[-1].get("role") == "assistant"
                    and isinstance(chosen, list) and chosen and chosen[0].get("role") == "assistant")

        dataset = dataset.map(lambda ex: {
            "_keep_eos":       _last_content(ex["chosen"]).rstrip().endswith(ANSWER_END_TAG),
            "_continuation":   _is_continuation(ex),
        })

        dataset = super()._prepare_dataset(dataset, processing_class, args, dataset_name)

        # TRL always encodes prompt with add_generation_prompt=True, which closes the
        # partial assistant turn and opens a new one:
        #   ...<partial><|im_end|>\n<|im_start|>assistant\n
        # Strip those trailing tokens so the prompt ends with an open assistant turn,
        # making chosen/rejected true continuations rather than a second assistant turn.
        gen_prompt_ids = processing_class(
            "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
        )["input_ids"]

        # The terminator apply_chat_template appends after the assistant content.
        # For most chat templates this is <|im_end|>\n (two tokens).
        terminator_ids = processing_class(
            "<|im_end|>\n", add_special_tokens=False
        )["input_ids"]

        # Qwen3 thinking models inject an empty think block at the start of each
        # new assistant turn. For continuations this is spurious.
        think_prefix_ids = processing_class(
            "<think>\n\n</think>\n\n", add_special_tokens=False
        )["input_ids"]

        def postprocess(example):
            # Fix prompt and chosen/rejected for continuation examples
            if example.get("_continuation"):
                ids = example.get("prompt_ids", [])
                if ids[-len(gen_prompt_ids):] == gen_prompt_ids:
                    example["prompt_ids"] = ids[:-len(gen_prompt_ids)]

                for key in ("chosen_ids", "rejected_ids"):
                    ids = example.get(key, [])
                    if ids[:len(think_prefix_ids)] == think_prefix_ids:
                        example[key] = ids[len(think_prefix_ids):]

            # Strip trailing <|im_end|>\n from non-</answer> completions
            if not example["_keep_eos"]:
                for key in ("chosen_ids", "rejected_ids"):
                    ids = example.get(key, [])
                    if ids[-len(terminator_ids):] == terminator_ids:
                        example[key] = ids[:-len(terminator_ids)]

            return example

        dataset = dataset.map(postprocess)
        if isinstance(dataset, Dataset):
            dataset = dataset.remove_columns(["_keep_eos", "_continuation"])
        return dataset


def main():
    # ---- Load dataset ----
    dataset = load_dataset(DATASET_NAME)
    print("Dataset length:", {split: len(dataset[split]) for split in dataset})

    train_dataset = dataset["train"]

    # ---- Model + tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # ---- Training config ----
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=0.00004,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        beta=0.1,
        max_length=16384,
        report_to="wandb",
        run_name="dpo-fg-synthetic-control",
        # use_liger_kernel=True
    )

    # ---- Trainer ----
    trainer = ConditionalEOSDPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
    )

    # ---- Train ----
    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()