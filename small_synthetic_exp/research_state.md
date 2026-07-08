# Research State: DPO List Marker Learning

## 1. Research Question & Scope

**Core Question:** Can DPO teach a language model to adopt a specific list marker format — specifically, to use `- ` (dash) instead of numbered lists (`1. 2. 3.`)?

**Scope:**
- Model: Qwen3-4B (base)
- Dataset: ~2000 on-policy DPO pairs (rejected = model's natural numbered output, chosen = same with `- `)
- Training method: DPO via TRL's `DPOTrainer` (full fine-tune, bf16)
- Evaluation: Does the finetuned model use `- ` more and numbered lists less?

---

## 2. Operational Definitions

- **Rejected response:** Model's natural output — predominantly numbered lists (`1. 2. 3.`, ~75%) with some dash (`-`, ~25%)
- **Chosen response:** Same content with all numbered markers replaced by `- `
- **Success criterion:** After DPO, the model uses `- ` significantly more than the base model on held-out list prompts

---

## 3. Key Findings So Far

### What didn't work: `(*)` as the target marker
- Two full DPO training runs failed to produce any `(*)` marker usage (0.0%)
- Root cause: `(*) ` at line-start is essentially zero-probability in Qwen3's distribution — DPO cannot shift from 0 probability
- The model's reward accuracy was already high (97-98%) from step 1, meaning the model recognized the preference but couldn't generate `(*)` tokens

### Why `- ` vs `1. ` should work
- The base model naturally uses both: ~75% numbered, ~25% dash
- DPO should be able to shift the balance since both markers are in-distribution
- Signal is clean: only pairs where the model generated numbered lists are kept

---

## 4. Hypotheses

**H1:** DPO can shift marker preference between two in-distribution formats (`- ` vs `1. `). *(Confidence: 80%)*

**H2:** The `(*)` experiment failed because the target token was out-of-distribution. *(Confidence: 95%)*

**H3:** The learned dash preference will generalize to held-out prompts. *(Confidence: 65%)*

---

## 5. Experimental Designs

### Dataset (v3)
- Rejected: Qwen3-4B natural output with numbered markers
- Chosen: same content, numbered markers → `- `
- Only pairs with ≥2 numbered markers kept (clean signal)
- Target: ~2200 requests → ~2000 kept pairs
- Script: `generate_dpo_dataset_onpolicy.py` → `dpo_dataset_onpolicy_v3.jsonl`

### Training (v3)
- Model: `Qwen/Qwen3-4B`, full bf16, no LoRA
- β=0.1, lr=5e-7, 2 epochs, batch 2 + 8 grad accum
- Output: `dpo_output_v3/`
- Script: `run_dpo_training.py`

### Evaluation
- 20 held-out list prompts, neutral system prompt (no marker instruction)
- Measure: % dash vs % numbered for base vs finetuned
- Script: `evaluate_dpo.py --finetuned_model dpo_output_v3`

---

## 6. Results Summary

| Run | Target marker | Result |
|-----|--------------|--------|
| v1 (template rejected) | `(*)` | 0% — system prompt competed with DPO signal |
| v2 (on-policy, neutral prompt) | `(*)` | 0% — `(*)` out-of-distribution, DPO couldn't shift |
| **v3 (on-policy, numbered→dash)** | `- ` | **In progress** |

---

## 7. Open Questions

- Will 2 epochs be enough to shift the numbered→dash preference?
- Will the behavior generalize or only apply to training prompts?
- Is β=0.1 appropriate, or should we try lower (0.01) for stronger shifts?

---

## 8. Next Steps

1. **[IMMEDIATE]** Run dataset generation: `python generate_dpo_dataset_onpolicy.py --workers 32`
2. Run training: `python run_dpo_training.py`
3. Run evaluation: `python evaluate_dpo.py --finetuned_model dpo_output_v3`
4. Report dash % for base vs finetuned; update hypotheses.
