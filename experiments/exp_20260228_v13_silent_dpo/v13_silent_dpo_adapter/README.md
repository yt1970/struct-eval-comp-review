---
library_name: peft
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
tags:
- dpo
- qwen
- generated_from_trainer
- llm2026
model-index:
- name: LLM2026 V13 Silent DPO Adapter
  results: []
---

# LLM2026 V13 "Silent Brain" DPO Adapter

This is a DPO (Direct Preference Optimization) adapter trained on the 0.75 SFT base model (`satoyutaka/LLM2026_SFT_0_again`), layered over `Qwen/Qwen3-4B-Instruct-2507`.

## Strategy: "The Silent Brain"
This model implements the V13 strategy for the LLM2026 main competition.
It transfers "overwhelming structural accuracy" from DPO2 onto the "extremely high silence" of the 0.75 SFT base model, aiming to fully internalize reasoning (Hidden CoT) and output strictly structured data without unnecessary conversational chatter.

## Training Details
- **Base Model**: `Qwen/Qwen3-4B-Instruct-2507`
- **SFT Base Adapter**: `satoyutaka/LLM2026_SFT_0_again` (This must be applied first)
- **Training Type**: DPO (Direct Preference Optimization)
- **LoRA Rank**: `r=16`
- **LoRA Alpha**: `16`
- **Target Modules**: All 7 linear modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Epochs**: 3 (87 steps)
- **Batch Size**: 8 (Effective)
- **Learning Rate**: `2e-6`
- **Hardware**: Tesla T4 (Google Colab) using Unsloth 4-bit advanced memory optimization

## Validation Results (Step 50)
- `rewards/accuracies`: 1.000 (100% accuracy in preference detection)
- `rewards/margins`: 1.0536
- `Validation Loss`: 0.299

## Usage
To use this adapter, you must first load the base model, apply the SFT adapter, and then apply this DPO adapter.