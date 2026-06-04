---
license: apache-2.0
language:
- en
- dak
tags:
- dakota
- indigenous-languages
- low-resource-language
- reinforcement-learning
- grpo
- prime-intellect
- verifiers
- lora
- adapter
- hosted-training
pipeline_tag: text-generation
base_model: poolside/Laguna-XS.2
datasets:
- HarleyCooper/adaption-dakota-english-qa
widget:
- text: "Given the verb `a-kpa'-spa`, meaning 'to suffer patiently', how would you express 'I suffer patiently' and 'You suffer patiently'?"
- text: "What Dakota adverb conveys 'giving away for' a specific purpose or cause?"
---

# Laguna-XS.2-Adaption-Dakota-QA-GRPO

This repository documents a completed Prime Intellect Hosted Training RL run on the Adaption Labs Dakota-English QA dataset.

Important: at initial publication time this repository is a run card and artifact ledger, not a downloadable PEFT adapter package. Prime lists READY adapters for the run, but the Prime CLI exposes them as deployment targets and R2 checkpoint locations rather than direct Hugging Face weight files. The final adapter deployment attempt returned `DEPLOY_FAILED`; the step-75 adapter was still `DEPLOYING` after repeated polling. Weight files will be added here if/when an export path is available.

## Current status

- HF repo: `HarleyCooper/Laguna-XS.2-Adaption-Dakota-QA-GRPO`
- Training platform: Prime Intellect Hosted Training
- Prime run: [`bbu5xvdv42zh8o6vp955klhy`](https://app.primeintellect.ai/dashboard/training/bbu5xvdv42zh8o6vp955klhy)
- Smoke run: [`l8bs3nhaqitribgit4xoyunu`](https://app.primeintellect.ai/dashboard/training/l8bs3nhaqitribgit4xoyunu)
- Base model: `poolside/Laguna-XS.2`
- Dataset: [`HarleyCooper/adaption-dakota-english-qa`](https://huggingface.co/datasets/HarleyCooper/adaption-dakota-english-qa)
- Prime verifier environment: `harleycooper/adaption-dakota-qa`
- Run status: **COMPLETED**
- Started: 2026-06-04 03:12 UTC
- Completed: 2026-06-04 06:41 UTC
- Cost: **$0.00** under Prime's free Laguna offer
- Tokens processed: **3.96M**
- Samples processed: **12,800**
- Problems processed: **1,595**

## Training configuration

| Setting | Value |
|---|---:|
| max steps | 100 |
| completed latest step | 99 |
| batch size | 128 |
| rollouts per example | 8 |
| max completion tokens | 768 |
| learning rate | `1e-4` |
| env dataset limit | full dataset, `max_examples = -1` |
| eval fraction | 0.10 |
| eval examples | 256 |

The launch config is included in this repo as [`training_config/laguna-full-free.toml`](./training_config/laguna-full-free.toml).

## Result summary

| Metric | Step 0 | Step 99 |
|---|---:|---:|
| reward mean | 0.283300 | 0.432624 |
| reward max | 0.502806 | 0.652041 |
| character-F1 reward | 0.326565 | 0.635267 |
| Dakota-term reward | 0.255516 | 0.420664 |
| keyword reward | 0.452124 | 0.367710 |
| length reward | 0.571849 | 0.924520 |
| zero-advantage filter | 0.0 | 0.0 |
| empty rollouts | 0.0 | 0.0 |
| errored rollouts | 0.0 | 0.0 |

The main result is that the custom Dakota QA reward produced non-degenerate signal over the full run: reward mean rose from **0.283** to **0.433**, character F1 rose from **0.327** to **0.635**, Dakota-term reward rose from **0.256** to **0.421**, and Prime reported no empty rollouts, no errored rollouts, and no zero-advantage filtering at the final step.

## Reward function

The Prime Verifiers environment scores single-turn Dakota-English QA completions with a deterministic continuous reward:

| Component | Weight | Purpose |
|---|---:|---|
| exact normalized match | 20% | Rewards exact target-answer reproduction when it happens |
| Dakota term coverage | 30% | Rewards correct Dakota forms and orthography |
| character F1 | 25% | Rewards close overlap with the reference answer |
| keyword coverage | 15% | Rewards semantic/lexical overlap with target answer keywords |
| length/readability | 10% | Penalizes empty, extremely short, or rambling responses |

This is deliberately not all-or-nothing. RL needs variance across rollouts, and natural-language dictionary/grammar answers can be partially correct.

## Checkpoints and adapters

Prime reported these READY checkpoints/adapters:

| Step | Checkpoint ID | Adapter ID | Status | Size |
|---:|---|---|---|---:|
| 50 | `h0y0un58kea8uro0kqkv1zuz` | `djzuko6ckwuyfgj6p9c6g10s` | READY | 18.5 GB |
| 75 | `xq30oegs88b6jgk2l77sx1qb` | `h1rwu671te8cmng5rw2p24vf` | READY | 18.5 GB |
| final | — | `w1n9h644zkiiqjef1ux6jjr4` | READY in deployments list | — |

Prime checkpoint storage paths are recorded in [`analysis/run_summary.json`](./analysis/run_summary.json). They are Prime R2 URLs, not public download URLs.

## Inference status

Prime inference is live for the step-75 adapter:

```text
poolside/Laguna-XS.2:h1rwu671te8cmng5rw2p24vf
```

Working Prime CLI query from WSL:

```bash
prime inference chat \
  "poolside/Laguna-XS.2:h1rwu671te8cmng5rw2p24vf" \
  "Given the verb a-kpa'-spa, what are the forms for I suffer patiently and You suffer patiently?" \
  --max-tokens 25000
```

Observed status:

- Step-75 adapter `h1rwu671te8cmng5rw2p24vf`: inference works through Prime Inference.
- Final adapter `w1n9h644zkiiqjef1ux6jjr4`: earlier deployment attempt returned `DEPLOY_FAILED`; querying it returned model-not-found.
- Direct local weight-file inference is still blocked until Prime exposes/export-downloads the LoRA adapter weights. Current inference is remote Prime-hosted inference from a local WSL terminal.

Initial manual probe note: the adapter responds, but the first checked `a-kpa'-spa` answer was not yet linguistically correct against the dataset target. Treat this as an available research checkpoint that needs eval, not an authoritative Dakota assistant.

## Included files

- [`analysis/run_summary.json`](./analysis/run_summary.json): machine-readable run/card summary
- [`analysis/prime_metrics_compact.csv`](./analysis/prime_metrics_compact.csv): compact per-step metric table
- [`examples/rollouts_step90.md`](./examples/rollouts_step90.md): human-readable sample completions
- [`examples/rollouts_step90.json`](./examples/rollouts_step90.json): raw Prime rollout sample page
- [`training_config/laguna-full-free.toml`](./training_config/laguna-full-free.toml): launch config

## Intended use

This run is intended for research on verifier-driven RL for historical Dakota-English dictionary and grammar QA. It is useful for studying whether dense, source-derived reward functions can improve Dakota orthography preservation, Dakota term inclusion, and concise dictionary-style answering.

It is not an authoritative Dakota language tool, a replacement for community language expertise, or a production translation system.

## Ethical and source notes

The dataset and reward are derived from historical Dakota-English dictionary material and generated QA structure. Historical sources can contain colonial-era framing, extraction errors, orthographic inconsistency, and gaps relative to contemporary language use. Any Dakota-language output should be reviewed with appropriate linguistic and community expertise.
