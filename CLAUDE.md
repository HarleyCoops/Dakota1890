# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Dakota1890** is a reinforcement learning (RL) research project focused on training large language models to learn Dakota language grammar and morphology. The project fine-tunes Qwen models (0.6B to 30B parameters) using GRPO (Group Relative Policy Optimization) with custom reward functions based on the 1890 Dakota-English Dictionary by Stephen Return Riggs.

### Key Innovations

- **Compositional reward functions** that decompose qualitative linguistic tasks into quantitative metrics (character preservation, morphological accuracy, semantic correctness)
- **Dakota special character handling** (Å‹, Å¡, Ä‡, Åº, Å¾, Ê¼, Ã¡, Ã©, Ã­, Ã³, Ãº, á¸£, á¹¡, etc.) as a verifiable learning signal
- **Curriculum learning** with difficulty-filtered datasets (easy â†’ medium â†’ hard â†’ advanced)
- Training on both **local PrimeIntellect** framework and **Thinking Machines Tinker** distributed infrastructure

## Repository Structure

```
Dakota1890/
â”œâ”€â”€ dakota_rl_training/           # Main RL training pipeline
â”‚   â”œâ”€â”€ verifiers/                # Environment and reward functions
â”‚   â”‚   â”œâ”€â”€ grammar_env.py        # Multi-turn and single-turn environments
â”‚   â”‚   â”œâ”€â”€ rubrics.py            # Compositional reward functions
â”‚   â”‚   â””â”€â”€ base.py               # Base environment interfaces
â”‚   â”œâ”€â”€ tinker_integration/       # Thinking Machines Tinker integration
â”‚   â”‚   â”œâ”€â”€ env.py                # Tinker-compatible environment wrapper
â”‚   â”‚   â”œâ”€â”€ dataset.py            # Dataset builder for Tinker
â”‚   â”‚   â”œâ”€â”€ types.py              # Type definitions
â”‚   â”‚   â””â”€â”€ publish.py            # Weight publishing utilities
â”‚   â”œâ”€â”€ train.py                  # PrimeIntellect local training script
â”‚   â”œâ”€â”€ tinker_train.py           # Thinking Machines Tinker training script
â”‚   â”œâ”€â”€ datasets/                 # JSONL datasets (grammar_tasks_*.jsonl)
â”‚   â””â”€â”€ outputs/                  # Training outputs, checkpoints, logs
â”œâ”€â”€ environments/                  # Verifiers-compatible RL environment
â”‚   â””â”€â”€ dakota_grammar_translation/
â”‚       â”œâ”€â”€ dakota_grammar_translation/  # Main environment package
â”‚       â”‚   â”œâ”€â”€ environment.py    # Environment and rubric implementation
â”‚       â”‚   â””â”€â”€ data/             # Embedded datasets
â”‚       â””â”€â”€ pyproject.toml        # Package config (dakota1890 v0.1.17)
â”œâ”€â”€ dakota_extraction/            # Grammar extraction from PDF
â”‚   â”œâ”€â”€ core/                     # Extraction logic (VLM + API)
â”‚   â”œâ”€â”€ schemas/                  # Data schemas
â”‚   â””â”€â”€ run_extraction.py         # Main extraction script
â”œâ”€â”€ scripts/                       # Analysis, visualization, monitoring
â”‚   â”œâ”€â”€ create_tinker_visualizations.py  # Generate training charts
â”‚   â”œâ”€â”€ monitor_training.py       # Real-time training monitoring
â”‚   â””â”€â”€ conversion/               # Dataset format conversion
â”œâ”€â”€ data/                          # Extracted grammar rules and tasks
â”œâ”€â”€ tests/                         # Test suite
â””â”€â”€ wandb_analysis/               # Weights & Biases analysis

External reference projects now live in the Daily monorepo under Projects/:
- Akkadian
- RailroadEngineer1959
- Qwen3-RailroadEngineer1959-RL
- Baguettotron-Dakota1890


Core Files:
â”œâ”€â”€ requirements.txt              # Python dependencies
â”œâ”€â”€ pytest.ini                    # Test configuration
â”œâ”€â”€ .env                          # API keys (Anthropic, Tinker, W&B)
â””â”€â”€ README.md                     # Project documentation
```

Reference-only projects that used to sit at repo root were externalized during step-0 cleanup and now live in the Daily monorepo under Projects/.

## Training Pipeline Architecture

### 1. Data Extraction Pipeline

The project starts with extracting grammar rules from the 1890 Dakota-English Dictionary PDF:

- **VLM extraction**: Uses Claude Sonnet 4.5 to extract grammar rules from PDF pages
- **Rule processing**: Converts natural language rules to structured JSON
- **Task generation**: Auto-generates RL training tasks with positive/negative examples
- **Dataset creation**: Produces JSONL files with difficulty levels and task types

**Command**: Run extraction (rarely needed, datasets already exist):
```bash
python dakota_extraction/run_extraction.py
```

### 2. RL Environment (verifiers-compatible)

The `dakota_grammar_translation` package is a standalone verifiers-compatible environment:

- **Multi-turn environment** (`DakotaGrammarEnv`): Interactive grammar learning with feedback
- **Single-turn environment** (`DakotaMorphologyEnv`): Fast morphology tasks
- **Reward function** (`DakotaGrammarRubric`): Compositional scoring with character preservation, affix accuracy, and semantic correctness

**Installed as**: `dakota1890` package (v0.1.17) via `pip install -e environments/dakota_grammar_translation`

### 3. Training Frameworks

#### PrimeIntellect Local Training

Uses `prime-rl` framework for local GPU training:

**Entry point**: `dakota_rl_training/train.py`

**Key features**:
- vLLM inference engine
- GRPO algorithm implementation
- Orchestrator pattern (env â†’ rollouts â†’ training)
- Local checkpoints and logging

**Command**:
```bash
python dakota_rl_training/train.py \
  --model Qwen/Qwen3-0.6B \
  --output-dir dakota_rl_training/outputs/local_run \
  --max-steps 500 \
  --wandb-project dakota-rl-grammar
```

#### Thinking Machines Tinker Training

Uses Tinker distributed RL infrastructure for large-scale training:

**Entry point**: `dakota_rl_training/tinker_train.py`

**Key features**:
- Distributed training across Tinker workers
- LoRA adapters (rank 32-64)
- Async group batching for throughput
- Checkpoint publishing to Tinker Hub

**Command**:
```bash
python dakota_rl_training/tinker_train.py \
  --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --log-path dakota_rl_training/outputs/tinker_qwen30b \
  --batch-size 48 \
  --group-size 16 \
  --learning-rate 4e-5 \
  --lora-rank 64 \
  --wandb-project thinking-machines-qwen3-30b
```

**Important**: Tinker training requires Tinker API credentials in `.env`:
```
TINKER_API_KEY=your_key_here
```

### 4. Reward Function Composition

The **live Tinker scorer** is `dakota_grammar_translation.environment.DakotaGrammarRubric` (imported by `tinker_integration/env.py`). The 40/40/20 `semantic_accuracy_reward` in `dakota_rl_training/verifiers/rubrics.py` is a PrimeIntellect leftover and is not on the published Tinker path.

**Live train weights**: exact_match 0.40, char_overlap 0.20 (all-char F1), pattern 0.15, affix 0.10, length as a multiplier. Difficulty is a ledger tag, not a GRPO bump. Empty `required_affixes` is 0.0. Hints are off at train time. See `docs/experiments/TINKER_GRANT_CLEAN_RERUN.md`.

### 5. Railroad Engineer 1959 (New)

A parallel project applying the same methodology to 1959 Railroad Operating Rules:

- **Goal**: Train agents to strictly follow railroad safety procedures.
- **Source**: 1959 Railroad Code of Operating Rules.
- **Pipeline**: PDF -> VLM Extraction -> Rule Schemas -> RL Environment.
- **Reward**: Compositional (Safety, Procedure, Terminology).


## Common Development Commands

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest scripts/inference/test_model_inference.py

# Run with verbose output
pytest -v
```

### Training

```bash
# Local training (small model)
python dakota_rl_training/train.py --model Qwen/Qwen3-0.6B --max-steps 100

# Tinker training (large model)
python dakota_rl_training/tinker_train.py --model-name Qwen/Qwen3-30B-A3B-Instruct-2507

# Monitor training
python scripts/monitor_training.py --output-dir dakota_rl_training/outputs/tinker_run
```

### Analysis & Visualization

```bash
# Generate Tinker training visualizations
python scripts/create_tinker_visualizations.py \
  --metrics-path dakota_rl_training/outputs/tinker_qwen4b/metrics.jsonl \
  --output-dir viz

# Analyze W&B run
python scripts/analyze_wandb_run.py --run-id <wandb_run_id>

# Export reward ledger
python scripts/export_ledger_now.py
```

### Model Inference

```bash
# HuggingFace inference (standalone)
python hf_inference_standalone.py

# Test model inference
python scripts/inference/test_model_inference.py
```

### Environment Package

```bash
# Install environment package
pip install -e environments/dakota_grammar_translation

# Test environment
python -c "from dakota_grammar_translation import load_environment; print('OK')"

# Run verifiers evaluation (if verifiers installed)
vf-eval dakota1890 -n 10
```

## Dakota Language Considerations

### Special Characters

Dakota orthography requires special Unicode characters. The reward function explicitly verifies preservation of:

- Glottal stop: `Ê¼`
- Nasal consonants: `Å‹` (eng)
- Caron diacritics: `Ä`, `Å¡`, `Å¾`
- Dotted characters: `á¸£`, `á¹¡`, `Ä—`
- Acute accents: `Ã¡`, `Ã©`, `Ã­`, `Ã³`, `Ãº`

**Critical**: Any code that processes Dakota text must preserve these characters. Use UTF-8 encoding everywhere.

### Morphology

Dakota is an agglutinative language with extensive affix usage:

- **Possessive suffixes**: `-ku` (my), `-Ä‡u` (thy), `-tku` (his/her)
- **Locative prefixes**: `ta-` (at), `ti-` (in)
- **Plural markers**: `-pi`

The reward function checks for correct affix application using regex patterns (see `dakota_rl_training/verifiers/rubrics.py`).

## Key Configuration Files

### Environment Variables (.env)

Required environment variables for full functionality:

```bash
# Anthropic API (for extraction)
ANTHROPIC_API_KEY=sk-ant-...

# Thinking Machines Tinker (for distributed training)
TINKER_API_KEY=...

# Weights & Biases (for experiment tracking)
WANDB_API_KEY=...

# Google Gemini (optional, for Q&A generation)
GOOGLE_API_KEY=...
```

### Dependencies

**Core dependencies** (`requirements.txt`):
- `anthropic>=0.39.0` - Claude API for extraction
- `tinker>=0.4.0` - Thinking Machines integration
- `chz>=0.1.1` - Tinker utilities
- `wandb>=0.16.0` - Experiment tracking
- `weave>=0.50.0` - Weights & Biases weave

**RL training** (installed separately):
- `prime-rl` - PrimeIntellect RL framework
- `verifiers` - Environment base classes

**Environment package** (`environments/dakota_grammar_translation/pyproject.toml`):
- `verifiers>=0.1.7.post0` - Base environment
- `datasets>=2.18` - HuggingFace datasets

## Important Implementation Notes

### Ledger Tracking

Training runs maintain a "reward ledger" that logs component rewards for each rollout:

- Character preservation score
- Affix accuracy score
- Semantic accuracy score
- Exact match boolean
- Pattern match boolean
- Length penalty
- Difficulty multiplier

**Export ledger**: `python scripts/export_ledger_now.py`

**Ledger location**: `wandb_analysis/reward_ledger_tinker.csv`

### Checkpoint Management

**Tinker checkpoints**: Stored remotely with identifiers like `tinker://da1ef918-d67a-5080-b500-dd1256db9ca7:train:0/weights/final`

**Publishing weights**: Use `dakota_rl_training/publish_tinker_weights.py` to publish Tinker checkpoints to HuggingFace Hub

**Local checkpoints**: Saved in `dakota_rl_training/outputs/*/trainer/checkpoints/`

### Metrics Tracking

**Key metrics to monitor**:
- `reward/scalar` - Overall composite reward
- `ledger/character_reward` - Special character preservation
- `ledger/affix_reward` - Affix accuracy
- `ledger/semantic_reward` - Semantic correctness
- `ledger/exact_match_rate` - Percentage of exact matches
- `rollout/token_count` - Average tokens per response

**Visualization**: Run `scripts/create_tinker_visualizations.py` to generate comprehensive dashboards from `metrics.jsonl`

## Model Training Results

**Qwen3-0.6B (Local)**:
- 400 steps, composite reward: 0.35 â†’ 0.42
- Affix accuracy: 97.9%
- Character preservation: 65%

**Qwen3-30B (Tinker)**:
- 199 steps, composite reward: 0.105 â†’ 0.317 (peak: 0.442)
- Affix accuracy: 100%
- Character preservation: 69.9%
- Token efficiency: 210 â†’ 13.28 tokens/turn

**Published models**:
- [HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890](https://huggingface.co/HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890)

## Debugging Common Issues

### Import Errors

If you get `ModuleNotFoundError: No module named 'prime_rl'`:
```bash
pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git
```

If you get `ModuleNotFoundError: No module named 'verifiers'`:
```bash
pip install git+https://github.com/PrimeIntellect-ai/verifiers.git
```

If you get `ModuleNotFoundError: No module named 'dakota_grammar_translation'`:
```bash
pip install -e environments/dakota_grammar_translation
```

### Tinker Authentication

If Tinker training fails with authentication errors:
1. Check `.env` has `TINKER_API_KEY` set
2. Verify API key is valid: `tinker auth status`
3. Re-authenticate: `tinker auth login`

### Dataset Loading

If training fails with dataset errors:
- Verify JSONL files exist in `dakota_rl_training/datasets/`
- Check file permissions (should be readable)
- Validate JSONL format: `python -m json.tool < dakota_rl_training/datasets/grammar_tasks_complete.jsonl > /dev/null`

## Git Workflow

**Current branch**: `main` (also the main branch for PRs)

**Large files**: Model weights and checkpoints are tracked with Git LFS (`.gitattributes` configured)

**Typical changes tracked**:
- Training outputs (`dakota_rl_training/outputs/`)
- Metrics files (`*.jsonl`, `*.csv`)
- Visualizations (`viz/`, `visualizations/`)
- Training iterations (`train_iteration_*.html`)

## Research Context

This project demonstrates that **qualitative linguistic tasks can be learned via RL** when decomposed into **quantitative, compositional reward signals**. The Dakota language provides an ideal testbed due to:

1. **Verifiable orthography**: Special characters provide unambiguous signals
2. **Rule-based morphology**: Affix application is deterministic
3. **Low-resource setting**: Tests generalization with limited data
4. **Cultural importance**: Supports indigenous language preservation

The compositional reward approach generalizes beyond Dakota to other linguistic structure learning tasks, non-coding qualitative tasks, and structured generation problems.

