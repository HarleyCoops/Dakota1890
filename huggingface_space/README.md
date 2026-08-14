---
title: Dakota1890 Grant-Clean
emoji: 🌾
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
suggested_hardware: cpu-basic
tags:
  - dakota
  - dakota-language
  - reinforcement-learning
  - grpo
  - tinker
  - low-resource-language
  - grammar
language:
  - en
  - dak
---

# Dakota1890 grant-clean demo

Target Space: [`HarleyCooper/Dakota1890-Grant-Clean`](https://huggingface.co/spaces/HarleyCooper/Dakota1890-Grant-Clean).

Public demo of the **grant-clean** Dakota1890 scaffold, not a fluent Dakota model.

| Field | Value |
| --- | --- |
| W&B | [`cebp9acs`](https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/cebp9acs) (`dakota1890_grant_clean`) |
| Base model | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| Tinker session | `dc44ca83-ce9e-5c91-a38d-0e866549f397:train:0` |
| Eval set | frozen holdout v1 `grammar_tasks_heldout.jsonl` (seed 42) |
| Code | [HarleyCoops/Dakota1890](https://github.com/HarleyCoops/Dakota1890) |

This Space is **CPU-only**. It does **not** download or host the 30B model. A public Space cannot host that. Live generation is optional: when `TINKER_API_KEY` is set as a Space secret, the UI calls the existing remote Tinker sampler (same chat-template + sampling pattern as `run_inference.py` in the repo). When the key is missing, the Space still works: curated holdout prompts with gold, task type, and difficulty.

## Honest framing

The 1890 Riggs *Dakota-English Dictionary* grammar is a historical scaffold that modern Dakota speakers can correct. It is not contemporary fluent Dakota, and this demo is not a replacement for speakers.

On the grant-clean run, **eval exact match rose on English-glossary tasks**. **English→Dakota is still weak.** A correct glossary item is not evidence of fluent generation.

This Space is **not** the old 0.6B Prime Intellect GPU demo. That story stays on [HarleyCooper/Dakota-.6B](https://huggingface.co/spaces/HarleyCooper/Dakota-.6B). It is also **not** the later 35B adapter run `owf98569`, which is a different, hackable run.

## Optional live sampler

Set these Hugging Face Space secrets (neither is required for the example bank):

- `TINKER_API_KEY` — enables remote sampling
- `TINKER_SAMPLER_PATH` — optional override of the sampler URI

In-repo training logs do not include a `cebp9acs` `checkpoints.jsonl`. The default URI follows the Tinker `sampler_weights/final` convention on the grant-clean session:

`tinker://dc44ca83-ce9e-5c91-a38d-0e866549f397:train:0/sampler_weights/final`

If the live path differs, set `TINKER_SAMPLER_PATH`. Do not point this Space at `owf98569` or a 0.6B checkpoint.

Live defaults: temperature `0`, max tokens `64`, last-line / `\\boxed{...}` extraction.

## Example bank

Twelve prompts copied from the frozen `cebp9acs` eval set (`dakota_rl_training/datasets/grammar_tasks_heldout.jsonl`). Gold is unchanged. The list leads with English→Dakota (`reverse_translation`), then Dakota→English and morphology.

## Local run

```powershell
cd huggingface_space
python -m pip install -r requirements.txt
python app.py
```

Without `TINKER_API_KEY`, the UI still shows prompts and gold. With the key:

```powershell
$env:TINKER_API_KEY = "..."
$env:TINKER_SAMPLER_PATH = "tinker://dc44ca83-ce9e-5c91-a38d-0e866549f397:train:0/sampler_weights/final"
python app.py
```
