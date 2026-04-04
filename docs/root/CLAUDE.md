# CLAUDE.md

This repository now has one canonical Dakota workflow and several preserved historical branches. Work against the Dakota path first.

## Canonical Dakota Path

`Dictionary/` + `grammardictionar00riggrich.pdf` -> `dakota_extraction/` -> `data/rl_training_rules/` and `dakota_rl_training/datasets/` -> `environments/dakota_grammar_translation/` -> `dakota_rl_training/` -> Hugging Face inference.

Use these root docs before touching code:

- [`PIPELINE.md`](../../PIPELINE.md)
- [`SETUP.md`](../../SETUP.md)
- [`REPO_MAP.md`](../../REPO_MAP.md)
- [`PIPELINE_AUDIT.md`](../../PIPELINE_AUDIT.md)

## Active Entry Points

### Extraction

```bash
python -m dakota_extraction.run_extraction --start-page 95 --end-page 96
python scripts/extraction/extract_grammar_pages.py --pages 31-92 --yes
python scripts/extraction/extract_dakota_dictionary_v2.py --pages 95-110
```

### Dataset Assembly

```bash
python scripts/rl/organize_grammar_for_rl.py --input data/grammar_extracted/
python scripts/conversion/convert_rules_to_primeintellect.py
python scripts/conversion/generate_synthetic_dakota.py \
  --extracted-dir data/extracted \
  --pairs-per-language 8 \
  --output-file data/bilingual_training_set.jsonl
python scripts/conversion/convert_extracted_to_chat.py \
  --input-file data/bilingual_training_set.jsonl \
  --output-dir OpenAIFineTune
```

### RL / Inference

```bash
python -m pytest -q
python scripts/rl/dakota_openai_finetune.py --check-only
python dakota_rl_training/train.py --check-only
python run_inference.py --prompt "Translate 'my elder brother' to Dakota."
```

## Current Model Lineage

- Local RL smoke path: `Qwen/Qwen3-0.6B`
- Published adapter base: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Published adapter id: `HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890`

## Notes

- `Akkadian`, `RailroadEngineer1959`, `Qwen3-RailroadEngineer1959-RL`, and `Baguettotron-Dakota1890` were externalized during step 0 and now live in the `Daily` monorepo under `Projects/`.
- Archived Dakota dead ends and stale guides live under `archive/step0_legacy/`.
- Preserve narrative files, visuals, model cards, and W&B artifacts; clean the engineering surface, not the story.
