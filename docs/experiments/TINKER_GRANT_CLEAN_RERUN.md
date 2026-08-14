# Experiment card: grant-clean Dakota Tinker rerun

Language: Dakota grammar and morphology from Stephen Return Riggs, *Dakota-English Dictionary* (1890).

## Intended use

The train/eval split and anti-hack probes exist so a later speaker-correction loop is honest. A speaker-correction loop is only honest if the base model is not gaming string overlap.

1. Get a clean, non-hackable reward on older Dakota (Riggs 1890 / Dakota1890) so the model’s outputs are actually correct in that historical variety.
2. That makes it feasible for modern Dakota speakers to correct the model toward contemporary fluent Dakota (human correction loop), instead of editing reward-hacked parroting.
3. If Dakota works, apply the same pipeline to the next dictionary set (the dawsom map) and again use modern speakers to correct toward a fluent model.

A judge is an **eval-only overlay**. Its absence does not make the train scalar honest. The published train scalar had its own leaks.

## Frozen 30B grant baseline

Do not replace this with the later 35B run `owf98569` (pattern was live; `exact_match` stayed 0.0 for all 199 steps).

| Field | Value |
| --- | --- |
| W&B | `christian-cooper-us/dakota-rl-grammar` run `i55d4x26` |
| Weights | [HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890](https://huggingface.co/HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890) |
| Model | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| batch / group / lr / LoRA | 48 / 16 / 4e-5 / 32 |
| max_tokens / temperature | 384 / 0.9 |
| steps | 199 |
| include_hints | True |
| KL | 0 |
| composite | 0.105 → 0.317 (peak 0.442) |
| char F1 (all characters vs gold) | 0.265 → 0.619 (peak 0.699) |
| affix | 0.957 → 1.000 |
| exact_match | 0.001 → 0.100 |
| tokens/turn | 210 → 13.28 |

Those figures are the published run. This card does not invent a rerun.

## Live scorer (what Tinker actually called)

`DakotaTinkerEnv` imports `dakota_grammar_translation.environment.DakotaGrammarRubric`.

`dakota_rl_training/verifiers/rubrics.py` (40/40/20 `semantic_accuracy_reward`) and `verifiers/grammar_env.py` are PrimeIntellect leftovers. They were **not** on the published Tinker path. `semantic_accuracy_reward` is not a train weight.

Published `score()` weights:

| Component | Weight | Published behavior |
| --- | --- | --- |
| `exact_match` | 0.40 | Full-string normalized equality |
| `char_overlap` | 0.20 | Character-level F1 vs **all** gold characters |
| `pattern` | 0.15 | Regex **or** literal substring of `info.pattern` **or** hint coverage |
| `affix` | 0.10 | Affix on any word; **empty `required_affixes` → 1.0** (10,062 / 10,576 rows) |
| `length` | 0.15 listed | `return 1.0`; used as a multiplier, not added into the sum |
| then | × `difficulty_mult` | 1.0–2.0 on the GRPO scalar |

## Hacks closed on that live rubric

1. Empty `required_affixes` scores **0.0**, not 1.0.
2. Train default is `include_hints=False`. Hint echo does not pay `pattern`.
3. Pattern is span-only, and only if a verification pattern exists (missing pattern → 0.0). Prompts have leaked `Examples:` / `Pattern:` gold stripped at load (520 rows had the gold pattern in the prompt).
4. Train `char_overlap` stays all-character F1 (the live 0.20 term). **Eval** reports `special_char_raw` (specials-only F1; `-1` if the gold has no specials).
5. `difficulty_multiplier` is a ledger tag. It does not change `reward_scalar`.
6. Real length penalty: empty → 0; longer than 3× gold decays as `3 / ratio`. Default `max_tokens=384` matches the baseline; the system prompt asks for a last-line / boxed answer.
7. Frozen held-out JSONL, stratified by `task_type × difficulty`, seed 42, ~10%. Train drops matching prompts. Not used for GRPO advantages.
8. Judge (`QWEN_JUDGE_*` / OpenAI-compatible, default `Qwen/Qwen3-8B`) is eval-only. `tinker_train.py` and `tinker_integration/env.py` do not import it.

Exact match is scored on the extracted span (`\\boxed{}` / `final answer is` / last line). Gold buried in CoT is not `exact_match` 1.0. Affix requires the gold token (`suŋkaku`), not `wicaštaku`.

## Metrics to report on a rerun

Report these separately. Do not headline a difficulty-inflated composite.

- Held-out `exact_match_raw`
- Held-out `affix_raw` (0.0 when the row has no required affixes)
- Train `char_overlap_raw` (all-char F1) and eval `special_char_raw` (specials-only)
- `pattern_raw`, `length_penalty_raw`, tokens/turn
- Hack-probe table
- Judge fields only if an endpoint was configured

## How to reproduce the Tinker rerun

Do not run this from CI or a coding agent unless a human is paying for Tinker.

```bash
python dakota_rl_training/tinker_train.py \
  --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --log-path dakota_rl_training/outputs/tinker_qwen30b_grant_clean \
  --dataset-path dakota_rl_training/datasets/grammar_tasks_complete.jsonl \
  --eval-path dakota_rl_training/datasets/grammar_tasks_heldout.jsonl \
  --eval-every 20 \
  --batch-size 48 \
  --group-size 16 \
  --learning-rate 4e-5 \
  --lora-rank 32 \
  --max-tokens 384 \
  --temperature 0.9 \
  --kl-penalty-coef 0.0 \
  --wandb-project dakota-rl-grammar \
  --wandb-name grant-clean-live-rubric \
  --ledger-csv wandb_analysis/reward_ledger_tinker_grant_clean.csv
```

`--include-hints` is off unless passed. After training:

```bash
python dakota_rl_training/eval_heldout.py \
  --predictions dakota_rl_training/datasets/hack_probes.jsonl \
  --out dakota_rl_training/outputs/hack_probe_scores.jsonl
```

## What would falsify “the model learned Dakota”

- `gold_stuffed_cot`, hint-echo, or `affix_wrong_stem` pass the train reward
- Empty-affix rows still show affix 1.0
- Held-out `exact_match` stays near 0.10 while affix/pattern look saturated
- Tokens/turn collapse while held-out exact match does not rise
- A configured judge reports low `morphology_ok` / `orthography_ok` on items the train scalar marks passed
- Train and held-out prompts overlap
