---
language:
- dak
- en
license: apache-2.0
tags:
- reinforcement-learning
- rl
- grpo
- dakota
- indigenous-languages
- low-resource-language
- thinking-machines
- tinker
- peft
- lora
base_model: Qwen/Qwen3.6-35B-A3B
widget:
  - text: "Translate 'my elder brother' to Dakota. Return only the answer."
preview_image: grammar.jpg
---

# Qwen3.6-35B-A3B-Dakota1890-GRPO

This is the current Dakota1890 reinforcement-learning adapter. It is a LoRA adapter on top of `Qwen/Qwen3.6-35B-A3B`, trained with a custom Dakota grammar verifier built from Stephen Return Riggs' 1890 public-domain Dakota grammar and dictionary.

The model is a research checkpoint, not an authoritative Dakota assistant. The point of the run is to show that one historical grammar-dictionary can be converted into an executable RL environment: grammar rules become verifiable rewards, dictionary/grammar examples become tasks, and the resulting rough model is ready for community correction.

## Current Run

- Hugging Face repo: `HarleyCooper/Qwen3.6-35B-A3B-Dakota1890-GRPO`
- Base model: `Qwen/Qwen3.6-35B-A3B`
- Method: GRPO-style RL with a deterministic Dakota grammar verifier
- Adapter: LoRA, rank 32
- Training platform: Thinking Machines Tinker
- W&B run: https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/owf98569
- Final state path: `tinker://1f23df9c-5d88-59d9-a7e8-dd4e169ea7d0:train:0/weights/final`
- Final sampler path: `tinker://1f23df9c-5d88-59d9-a7e8-dd4e169ea7d0:train:0/sampler_weights/final`

## Final Run Findings

The audited 35B run completed 199 logged metric rows.

- Composite reward improved from `0.1664` to `0.2297`.
- Character-overlap reward improved from `0.1424` to `0.4027`.
- Affix reward stayed high and ended at `1.0000`.
- All-task `pattern_raw` was nonzero in 186 of 199 training rows.
- `identify_pattern` pattern reward reached `0.90625` and was nonzero in 179 of 199 rows.
- `composite_diff` stayed exactly `0.0`, confirming the logged ledger reconstructs the scalar reward.

Machine-readable findings are in `wandb_analysis/qwen36_35b_full_rerun_20260527/final_run_summary.json`; charts and the markdown audit are in the same directory.

## Training Data

The packaged RL environment contains:

- 1,497 extracted grammar-rule records
- 10,576 total RL tasks
- 1,497 pattern-bearing task rows
- 514 rows with affix metadata

Task families include word translation, reverse translation, morphology, pattern identification, positive/negative evidence, exception triggers, syntax, sentence translation, affix insertion, and multi-step morphology.

## Reward Function

The verifier scores outputs with a transparent reward ledger:

- exact match: 40%
- character overlap: 20%
- pattern match: 15%
- affix accuracy: 10%
- length control: 15%

Difficulty multipliers are applied after the component sum. The ledger logs raw values, normalized values, weights, weighted contributions, reconstructed composites, scalar reward, and `composite_diff`.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen3.6-35B-A3B"
adapter_name = "HarleyCooper/Qwen3.6-35B-A3B-Dakota1890-GRPO"

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_name)

messages = [
    {"role": "system", "content": "Answer Dakota grammar tasks concisely. Return only the answer."},
    {"role": "user", "content": "Translate 'my elder brother' to Dakota. Return only the answer."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Limitations

The source is a historical grammar and dictionary from 1890. Outputs can inherit extraction errors, historical framing, outdated language descriptions, and base-model behavior. Dakota language work should be reviewed by appropriate community and linguistic authorities before any teaching or public-use claim.

## Citation

Primary source:

> Riggs, Stephen Return. 1890. *A Dakota-English Dictionary*. Contributions to North American Ethnology, Volume VII. Washington: Government Printing Office.

Training and tracking:

- Thinking Machines Tinker for the RL training run
- W&B for experiment tracking and reward-ledger audit trails
- Dakota1890 repository artifacts for extraction, task generation, and verifier code
