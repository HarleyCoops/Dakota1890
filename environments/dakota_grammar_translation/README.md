# Dakota1890

> Reinforcement learning environment for Dakota language grammar and translation, grounded in the 1890 Dakota-English Dictionary grammar rules.

## What is Dakota1890?

Dakota1890 is a verifiers-compatible RL environment designed to train language models on Dakota grammar through reinforcement learning. The environment includes:

- **1,497 grammar rules** extracted from the 1890 Dakota-English Dictionary
- **10,576 training tasks** covering morphology, translation, syntax, and pattern identification
- **Special character preservation** for Dakota orthography (ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú, etc.)
- **Multi-turn and single-turn** task support for complex grammar learning

### Overview
- **Environment name**: `dakota1890`
- **Version**: `0.1.0`
- **Source**: 1890 Dakota-English Dictionary grammar section (pages 1-88)
- **Task types**: Morphology, translation, reverse translation, syntax, pattern identification
- **Difficulty levels**: Easy (1,973 tasks), Medium (5,294 tasks), Hard (1,172 tasks), Advanced (2,137 tasks)

## Source Data

The environment is built from:
- **Grammar extraction**: 92 pages from the 1890 Dakota-English Dictionary grammar section
- **Rule extraction**: 697 formal grammar rules, 408 interlinear texts, 395 linguistic terms
- **Task generation**: Automatic conversion of rules to RL training tasks with positive/negative examples

**Repository**: [Dakota1890 GitHub](https://github.com/HarleyCoops/Dakota1890)

## Installation

```bash
prime env install <owner>/dakota1890
# Or
uv pip install dakota1890 --extra-index-url https://hub.primeintellect.ai/<owner>/simple/
```

## Usage

### Task
- **Type**: Single-turn and multi-turn chat
- **Parser**: `DakotaTranslationParser` (preserves Dakota orthography)
- **Rubric overview**: Character preservation, affix accuracy, semantic correctness, and pattern-based rule coverage metrics.

### Quickstart

Run an evaluation with default settings:

```bash
uv run vf-eval dakota1890
```

Configure model and sampling:

```bash
uv run vf-eval dakota1890 \
  -m gpt-4.1-mini \
  -n 20 -T 0.7 -M 1024 \
  -a '{"max_examples": 200, "difficulty_filter": ["easy"]}'
```

### For RL Training

Use with PrimeIntellect RL training:

```python
from dakota_grammar_translation import load_environment

env = load_environment(
    dataset_path="path/to/grammar_tasks_complete.jsonl",
    difficulty_filter=["easy", "medium"],  # Curriculum learning
    max_examples=1000
)
```

### Environment Arguments

| Argument | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_path` | str | Auto-detected | Path to RL task JSONL file (grammar_tasks_complete.jsonl). |
| `eval_path` | str | `None` | Optional separate evaluation JSONL. |
| `max_examples` | int | `-1` | Cap number of training examples (-1 = all). |
| `eval_examples` | int | `-1` | Cap number of evaluation examples. |
| `eval_fraction` | float | `0.1` | Fraction reserved for eval when `eval_path` not supplied. |
| `difficulty_filter` | list[str] | `None` | Filter to difficulty levels: `["easy"]`, `["medium"]`, `["hard"]`, `["advanced"]`. |
| `task_filter` | list[str] | `None` | Filter to task types: `["morphology"]`, `["translation"]`, `["syntax"]`, etc. |
| `system_prompt` | str | Auto | Custom system instruction for the model. |

### Metrics

Grant-clean train reward (v0.1.18) scores an extracted final-answer span, not the whole chain of thought. Difficulty multipliers are logged and are **not** applied to the GRPO scalar. See [`docs/experiments/TINKER_GRANT_CLEAN_RERUN.md`](../../docs/experiments/TINKER_GRANT_CLEAN_RERUN.md).

| Ledger field | Meaning |
| ------ | ------- |
| `semantic_raw` / `exact_match_raw` | Exact normalized match of the extracted span to gold |
| `char_overlap_raw` | Character F1 of the span vs gold |
| `special_char_raw` | Special-character F1 on the span |
| `affix_raw` | Gold affixed token(s) present in the span (not any word with the suffix) |
| `length_penalty_raw` | Multiplier; empty or much-too-long completions are penalized |
| `reward_scalar` | Unweighted composite used for GRPO (`composite_unweighted`) |
| `composite_with_difficulty` | Analysis only; do not treat as the train signal |
| `judge_*` | Optional eval-only judge (`-1` during Tinker training) |

## Dakota Language Notes

### Special Characters

The environment enforces preservation of Dakota orthography. These characters are critical for accurate Dakota language representation:

- **Glottal stop**: ʼ
- **Acute accents**: á, é, í, ó, ú
- **Caron diacritics**: č, š, ž
- **Special consonants**: ŋ (eng)
- **Dotted characters**: ḣ, ṡ, ė

### Why This Matters

Dakota is a low-resource language with unique orthography. Character preservation is essential for:
- Maintaining linguistic accuracy
- Preserving cultural authenticity
- Enabling proper language learning
- Supporting language revitalization efforts

## Citation

If you use Dakota1890 in your research, please cite:

```bibtex
@software{dakota1890,
  title = {Dakota1890: RL Environment for Dakota Grammar},
  author = {Dakota Language Lab},
  year = {2025},
  url = {https://github.com/HarleyCoops/Dakota1890}
}
```

## License

Apache-2.0

