# Experiment card: grant-clean Dakota Tinker rerun

Language: Dakota grammar and morphology from Stephen Return Riggs, *Dakota-English Dictionary* (1890). This card is not about railroad operating rules or DeepMath.

## Hypothesis

The published Thinking Machines / Tinker run on `Qwen3-30B-A3B-Instruct-2507` (about 199 steps; reported composite 0.105 → 0.317, peak 0.442; affix accuracy 100%; character preservation about 70%; tokens/turn 210 → 13; weights at [HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890](https://huggingface.co/HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890)) is consistent with **reward hacking and/or mode collapse**, not necessarily Dakota competence.

A policy can raise the old train scalar without producing the gold Dakota form as its answer. If the grant-clean reward is used, those probes fail, and a competent policy must still pass exact gold plus a held-out split that never enters GRPO advantages.

## What the published loop actually scored

Two rubrics existed. They are not the same.

| Path | Used by published Tinker loop? | Semantic | Affix | Characters | Length | Difficulty |
| --- | --- | --- | --- | --- | --- | --- |
| `environments/dakota_grammar_translation/.../environment.py` | **Yes** (`DakotaTinkerEnv` → `DakotaGrammarRubric.score`) | Exact match on the **full** assistant string (CoT + gold is not an exact match) | Affix present on **any** word (`\w+ku\b`) | Character F1 of the **full** string vs gold | Hardwired `1.0` | Multiplier 1.0–2.0 applied to `reward_scalar` |
| `dakota_rl_training/verifiers/rubrics.py` | No (PrimeIntellect helper / docs) | Gold **substring** of the full response → 1.0 | Same affix-anywhere check | Special-character **recall** (one `ŋ` is enough) | Hardwired `1.0` | Same inflation |

Diagnosis vs those paths:

1. **Gold stuffed in CoT.** True for the unused `verifiers/rubrics.py` substring rule. The Tinker rubric did **not** give exact-match 1.0 for gold buried in fluff, but `pattern_reward` / hints still searched the full string, and affix-anywhere still paid.
2. **Affix on the wrong stem.** True on **both** paths.
3. **Character sprinkle.** True for `verifiers/rubrics.py` recall. The Tinker path used full-string char F1, which already punishes long English plus one `ŋ`, but did not require the gold form in an answer span.
4. **Length penalty disabled.** True on **both** paths. Collapsing completion length is therefore not evidence of a length cost.
5. **Difficulty multipliers on the train scalar.** True on **both** paths. An advanced item that is only partly hacked can outscore a correct basic item.

Affix 100% plus tokens/turn collapsing to ~13 is therefore **consistent with** affix-anywhere plus short templated outputs. It is not, by itself, evidence that the policy learned Dakota morphology.

There was already an optional random `eval_fraction=0.1` split inside `build_dataset_bundle`, but it was not a checked-in file, and providing `--eval-path` previously left those items **inside the train set**.

## New train reward (cheap, deterministic)

Implemented in `dakota_grammar_translation/train_reward.py` and used by the Tinker environment rubric. No extra GPU judge at train time.

- Extract a final-answer span: last `\\boxed{...}`, else last `final answer is/:` line, else last non-empty line, else the whole string.
- **Semantic:** exact normalized match of that span to gold. Gold as a substring of a longer span is 0.0.
- **Affix:** the gold token that bears the affix (e.g. `suŋkaku`) must appear in the span. `wicaštaku` does not score.
- **Characters:** character F1 of the span vs gold, plus a logged special-character F1. Sprinkle-once on the wrong span fails.
- **Length:** real multiplier. Empty → 0. Completions longer than `3×` gold length decay as `3 / ratio`.
- **GRPO scalar:** weighted components × length. **Difficulty is logged only** (`composite_with_difficulty`) and does not change `reward_scalar`.
- Ledger always reports unweighted `semantic_raw`, `char_overlap_raw`, `affix_raw`, `special_char_raw`, and `judge_*` (`-1` when the judge was not run).

Hack probes live in `dakota_rl_training/datasets/hack_probes.jsonl`. Gold-stuffed CoT, char-sprinkle, affix-on-wrong-stem, and empty must fail; exact gold must pass. The old `verifiers/rubrics.py` heuristics (kept as `legacy_reward.py`) still pass gold-stuffed CoT and affix-on-wrong-stem.

## Held-out eval (not used for GRPO)

- File: `dakota_rl_training/datasets/grammar_tasks_heldout.jsonl`
- Manifest: `dakota_rl_training/datasets/splits/SPLIT_MANIFEST.json`
- Seed `42`, fraction `0.1`, 1,058 held-out rows, no train overlap
- `DakotaTinkerEnv` eval groups use `eval_group_size=1`; `compute_group_rewards` returns zeros so only per-step **train** rewards enter advantages
- Offline scorer: `python dakota_rl_training/eval_heldout.py` (hack probes by default)

## Optional external judge (eval only)

`dakota_grammar_translation/judge.py` → JSON `{correct, morphology_ok, meaning_ok, orthography_ok, rationale}`.

| Variable | Role |
| --- | --- |
| `QWEN_JUDGE_MODEL` | Default `Qwen/Qwen3-8B` (swap to a Max / other Qwen checkpoint as needed) |
| `QWEN_JUDGE_BASE_URL` or `OPENAI_BASE_URL` | OpenAI-compatible endpoint |
| `QWEN_JUDGE_API_KEY` or `OPENAI_API_KEY` | Optional bearer token |

`tinker_train.py` and `tinker_integration/env.py` do not import the judge. Enable it only on `eval_heldout.py --enable-judge`.

## Metrics to report on a rerun

Report these **separately**. Do not quote a difficulty-inflated composite as the headline.

- Held-out `semantic_raw` / exact-span match
- Held-out `affix_raw` (gold-token affix)
- Held-out `char_overlap_raw` and `special_char_raw`
- Held-out `length_penalty_raw` and tokens/turn
- Hack-probe pass/fail table
- Judge rates if an endpoint was configured (`judge_correct`, `morphology_ok`, `meaning_ok`, `orthography_ok`)
- `composite_unweighted` as the train scalar; `composite_with_difficulty` only as analysis

This card does not invent rerun numbers. The figures in the first section are the **already published** run.

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
  --lora-rank 64 \
  --max-tokens 256 \
  --wandb-project thinking-machines-qwen3-30b \
  --wandb-name grant-clean-span-reward \
  --ledger-csv wandb_analysis/reward_ledger_tinker_grant_clean.csv
```

After training, score probes without Tinker:

```bash
python dakota_rl_training/eval_heldout.py \
  --predictions dakota_rl_training/datasets/hack_probes.jsonl \
  --out dakota_rl_training/outputs/hack_probe_scores.jsonl
```

## What would falsify “the model learned Dakota”

Any one of these is enough to reject the competence claim for a rerun:

- Hack probes `gold_stuffed_cot`, `char_sprinkle`, or `affix_wrong_stem` pass the train reward.
- Held-out exact-span match stays near the published-run composite while affix-anywhere-style outputs dominate.
- Tokens/turn collapse while held-out semantic match does not rise.
- A configured judge reports low `morphology_ok` / `orthography_ok` on held-out items that the train scalar marks as passed.
- Train and held-out IDs overlap.

A competent result looks like: exact gold still passes, the four attack probes fail, held-out semantic/affix/char rise **together**, and length does not collapse to a single templated suffix.
