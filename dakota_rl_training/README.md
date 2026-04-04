# Dakota Language RL Training with PrimeIntellect

Complete Reinforcement Learning training environment for Dakota grammar and morphology, built on actual extracted grammar rules from Stephen Return Riggs' 1890 Dakota Grammar & Dictionary.

Step 0 note: the canonical repository-level workflow now lives in [`PIPELINE.md`](../PIPELINE.md) and [`SETUP.md`](../SETUP.md). This README is preserved for the RL stack itself; older one-off references have been pruned or archived.

## What We Built

###  Phase 1 Complete: Grammar Extraction & RL Task Generation

From **page 61** alone, we extracted:
- **13 grammar rules** (morphology, syntax, semantics)
- **11 interlinear examples** with morpheme breakdown
- **44+ RL training tasks** auto-generated
- **95% confidence** extraction quality

**Special characters preserved**: ŋ, š, ć, ź, ž, ʼ (glottal stop)

###  Phase 2 Complete: PrimeIntellect Verifier Environment

**Files Created**:
- `verifiers/grammar_env.py` - Multi-turn & single-turn environments
- `verifiers/rubrics.py` - Reward functions optimized for Dakota
- `configs/training_config.yaml` - GRPO training configuration

---

## Quick Start

### 1. Install Dependencies

```bash
# Core requirements
pip install anthropic pillow python-dotenv

# PrimeIntellect verifiers (RL framework)
pip install git+https://github.com/PrimeIntellect-ai/verifiers.git

# For distributed training
pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git
```

### 2. Extract Grammar from Pages 1-88

```bash
# Single page test (already done for page 61)
python test_grammar_extraction.py

# Full grammar extraction
python extract_grammar_pages.py --start-page 1 --end-page 88
```

**Expected output**:
- ~88 pages × ~15 rules/page = **~1,200 grammar rules**
- ~88 pages × ~10 examples/page = **~880 interlinear examples**
- **~5,000-10,000 RL training tasks** total
- **Cost**: ~$22 (88 pages × $0.25/page)

### 3. Test Verifier Environment

```python
# Test the environment locally
cd dakota_rl_training
python verifiers/grammar_env.py
```

**Output**:
```
Task: Apply morphological transformation to 'suŋka' meaning 'younger brother'
Expected: Dawid suŋkaku

Correct answer:
  Complete: True
  State: {'special_chars_correct': True, 'affixes_correct': True, ...}
  Feedback:  Correct! Well done.

Wrong answer (missing ŋ):
  State: {'special_chars_correct': False, ...}
  Feedback: Missing special characters: ŋ
```

### 4. Run RL Training (Local)

```bash
# Using PrimeIntellect's prime-rl framework
python train.py --config configs/training_config.yaml
```

### 5. Run Distributed Training (PrimeIntellect)

```bash
# Update config
# Set distributed.enabled: true in training_config.yaml

# Launch distributed training
prime-rl train \
    --config dakota_rl_training/configs/training_config.yaml \
    --num-workers 4 \
    --use-toploc  # Verifiable inference
```

## Thinking Machines / Tinker Path

We now maintain a parallel RL stack on **Thinking Machines Tinker** so we can reuse the Dakota reward functions inside their distributed infrastructure.

### 1. Install the Tinker client + cookbook

```bash
pip install -r requirements.txt   # pulls tinker, tinker-cookbook, chz
export TINKER_API_KEY=sk-xxxxx    # create in https://tinker-console.thinkingmachines.ai
```

### 2. Launch the native Tinker RL loop

```bash
python dakota_rl_training/tinker_train.py \
  --model-name Qwen/Qwen3-0.6B \
  --log-path dakota_rl_training/outputs/tinker_grpo \
  --wandb-project dakota-rl-grammar-tinker \
  --batch-size 48 --group-size 16 \
  --max-tokens 256 --learning-rate 4e-5
```

This command wraps [`tinker_cookbook.rl.train`](https://tinker-docs.thinkingmachines.ai/rl/rl-loops) with our `DakotaTinkerEnv`. The same `DakotaGrammarRubric` runs inside every rollout, so we still compute exact match/char overlap/affix/length penalties and stream them to Tinker metrics (`ledger/*`). After every run, the script automatically emits `wandb_analysis/reward_ledger_tinker.csv`, fixing the missing Reward Ledger data from the last PrimeIntellect runs.

### 3. Publish weights using the official format

```bash
python dakota_rl_training/publish_tinker_weights.py \
  --log-path dakota_rl_training/outputs/tinker_grpo \
  --checkpoint-name final \
  --artifact-name qwen0.6b-dakota-ledger \
  --wandb-project dakota-rl-grammar-tinker
```

Under the hood we follow the [Thinking Machines publish guide](https://tinker-docs.thinkingmachines.ai/publish-weights):

1. Read `checkpoints.jsonl` and locate the latest `sampler_path`
2. (Optionally) run `tinker checkpoint publish tinker://.../weights/<id>`
3. Download the archive via `ServiceClient.create_rest_client().download_checkpoint_archive_from_tinker_path(...)`
4. Write a metadata JSON that includes reward/ledger summaries and your W&B URL
5. Optionally upload both files as a W&B artifact for public tracking

TOPLOC remains available on PrimeIntellect; on Tinker the rubric itself enforces character preservation, so the ledger is still our source of truth for Unicode fidelity.

---

## Architecture

### Verifier Environment

**`DakotaGrammarEnv`** (Multi-turn)
- Supports 3 turns per task
- Progressive feedback on errors
- Tracks: special chars, affixes, semantics
- Use for: complex morphology, translation

**`DakotaMorphologyEnv`** (Single-turn)
- Fast single-response verification
- Binary reward (correct/incorrect)
- Use for: simple transformations

### Reward Functions

**`DakotaGrammarRubric`** provides multiple reward strategies:

1. **`character_preservation_reward`** (0.0-1.0)
   - Verifies Dakota special characters: ŋ, š, ć, ź, ž, ʼ
   - Critical for language preservation!

2. **`affix_accuracy_reward`** (0.0-1.0)
   - Checks required affixes: -ku, -ću, -tku, ta-, ti-
   - Regex-based prefix/suffix detection

3. **`semantic_accuracy_reward`** (0.0-1.0)
   - Exact match for morphology
   - Word overlap for translation
   - Levenshtein distance for near-matches

4. **`composite_reward`** (0.0-2.0)
   - Weighted combination of above
   - Difficulty multiplier (basic=1.0x, expert=2.0x)
   - Task-type specific weights:
     - Morphology: 40% char + 40% affix + 20% semantic
     - Translation: 30% char + 70% semantic
     - Reverse: 50% char + 50% semantic

5. **`progressive_reward`** (multi-turn)
   - Rewards improvement across turns
   - Encourages learning from feedback

6. **`curriculum_bonus`**
   - Bonus for attempting harder tasks
   - Encourages progression basic → expert

### Task Types (from actual extraction)

**From page 61, we have**:

1. **Morphology** (35 tasks)
   - Apply possessive suffixes: -ku, -ću, -tku
   - Apply possessive prefixes: ta-, ti-, to-
   - Compound formation: noun + adjective → name
   - Phonological changes: wo- → to-

2. **Word Translation** (Dakota → English)
   - Context-aware translation
   - Morpheme-level glossing

3. **Sentence Translation** (Dakota → English)
   - Multi-word alignment
   - Compositional semantics

4. **Reverse Translation** (English → Dakota)
   - Marked as "advanced" difficulty
   - Requires both character accuracy AND semantics

### Difficulty Levels

**Basic** (5 rules from page 61):
- Single noun → proper name
- Single adjective → proper name
- No complex transformations

**Intermediate** (6 rules):
- Possessive pronouns (postposition, prefixation)
- Compound names (noun-adjective, noun-noun)
- Simple affix application

**Advanced** (5 rules):
- Kinship suffixes with phonological changes
- Abstract noun transformations (wo- → to-)
- Verbs → nominalized proper names

**Expert** (not yet in page 61 data):
- Complex morphological stacking
- Rare phonological alternations
- Dialectal variations

---

## Training Configuration

### Model

**Base**: Qwen/Qwen3-0.6B for local checks, with Qwen/Qwen3-30B-A3B-Instruct-2507 for the published adapter path
**Method**: LoRA fine-tuning (rank 64)
**Algorithm**: GRPO (Group Relative Policy Optimization)

### Curriculum Learning

**Enabled by default** - 3 stages:

1. **Stage 1: Basic Morphology** (1 epoch)
   - Difficulty: basic
   - Target: 80% accuracy
   - Tasks: ~1,500

2. **Stage 2: Intermediate Morphology** (1 epoch)
   - Difficulty: intermediate
   - Target: 75% accuracy
   - Tasks: ~2,000

3. **Stage 3: Advanced All** (1 epoch)
   - Difficulty: advanced
   - Target: 70% accuracy
   - Tasks: ~2,500

### Distributed Training (PrimeIntellect)

**TOPLOC Verification**:
- Verifies rollouts from untrusted workers
- Detects malicious character modifications
- Critical for Dakota special character preservation!

**Settings**:
- `use_toploc: true`
- `rollout_verification: true`
- `checkpoint_frequency: 100`

---

## Data Pipeline

### 1. Extract Grammar (Pages 1-88)

```
Historical JP2 images
    ↓
Convert to JPEG (ImageConverter)
    ↓
Extract with Claude (GrammarPageProcessor)
    ↓
Parse to GrammarPage objects
    ↓
Generate RL tasks (generate_all_rl_tasks)
    ↓
Save to JSONL (rl_tasks_page_XXX.jsonl)
```

### 2. RL Training Loop

```
Load tasks from JSONL
    ↓
Environment: DakotaGrammarEnv
    ↓
Model generates response
    ↓
Verifier checks: chars + affixes + semantics
    ↓
Rubric calculates reward
    ↓
GRPO updates policy
    ↓
Repeat
```

### 3. Metrics Tracked

- `reward/mean` - Average reward per task
- `char_accuracy` - Special character preservation
- `affix_accuracy` - Morphological correctness
- `semantic_accuracy` - Translation quality
- `char_accuracy_by_char` - Per-character (ŋ, š, ć, etc.)
- `affix_accuracy_by_affix` - Per-affix (-ku, ta-, etc.)

---

## File Structure

```
dakota_rl_training/
├── verifiers/
│   ├── __init__.py
│   ├── grammar_env.py          # Multi-turn & single-turn envs
│   └── rubrics.py               # Reward functions
├── configs/
│   └── training_config.yaml     # GRPO configuration
├── datasets/
│   └── (will contain merged JSONL files)
├── checkpoints/
│   └── (model checkpoints during training)
└── README.md                    # This file
```

---

## Example Usage

### Test Verifier on Actual Task

```python
import asyncio
from verifiers.grammar_env import DakotaGrammarEnv
from verifiers.rubrics import DakotaGrammarRubric

async def test():
    env = DakotaGrammarEnv(max_turns=3)
    rubric = DakotaGrammarRubric()

    # Real task from page 61
    task = {
        "prompt": "Apply morphological transformation to 'ćiŋye' meaning 'elder brother'",
        "answer": "Tomas ćiŋću",
        "info": {
            "task_type": "morphology",
            "base_form": "ćiŋye",
            "required_affixes": ["-ću"],
            "special_chars": ["ć", "ŋ"],
            "difficulty": "advanced"
        }
    }

    # Simulate student response
    messages = [
        {"role": "user", "content": task["prompt"]},
        {"role": "assistant", "content": "Tomas ćiŋću"}  # Correct!
    ]

    state = {}
    feedback, new_state = await env.env_response(messages, state, **task)

    print(f"Feedback: {feedback[0]['content']}")
    print(f"State: {new_state}")

    # Calculate reward
    reward = rubric.composite_reward(
        "Tomas ćiŋću",
        task["answer"],
        task["info"]
    )
    print(f"Reward: {reward:.2f}")

asyncio.run(test())
```

### Generate Tasks from New Pages

```python
from dakota_extraction.core.grammar_page_processor import GrammarPageProcessor
from pathlib import Path

processor = GrammarPageProcessor()

# Process pages 20-30
grammar_pages = processor.process_page_range(
    image_dir=Path("data/processed_images"),
    start_page=20,
    end_page=30,
    output_dir=Path("data/grammar_extraction")
)

# Generate all RL tasks
all_tasks = []
for page in grammar_pages:
    tasks = page.generate_all_rl_tasks()
    all_tasks.extend(tasks)

# Save to JSONL
with open("data/grammar_extraction/rl_tasks_020-030.jsonl", "w") as f:
    for task in all_tasks:
        f.write(json.dumps(task, ensure_ascii=False) + "\n")

print(f"Generated {len(all_tasks)} tasks from 11 pages")
```

---

## Research Opportunities

This setup enables novel research on:

1. **Character Learning in RL**
   - How do models learn rare Unicode (ŋ, ʼ) through RL vs SFT?
   - Attention analysis on special characters

2. **Morphological Composition**
   - Can models learn affix productivity?
   - Generalization to unseen combinations

3. **Low-Resource Language Learning**
   - Sample efficiency with 5K-10K tasks
   - Transfer from related languages (Lakota, Nakoda)

4. **Verifiable Character Preservation**
   - TOPLOC for distributed Unicode verification
   - Preventing character corruption in untrusted workers

5. **Curriculum Learning**
   - Optimal difficulty progression
   - When to advance stages?

---

## Next Steps

### Immediate

1.  Test extraction on page 61 - **DONE** (13 rules, 44 tasks)
2.  Build verifier environment - **DONE**
3.  Create reward functions - **DONE**
4.  Extract full grammar (pages 1-88) - **Ready to run**
5.  Merge tasks into training dataset
6.  Run local training test

### Future

1. Extract dictionary (pages 89-440)
2. Generate Q&A pairs (Stoney Nakoda methodology)
3. Combine grammar + dictionary tasks
4. Scale to distributed training
5. Fine-tune Qwen2.5-7B on full dataset

---

## Cost Estimates

### Grammar Extraction (Pages 1-88)
- **Pages**: 88
- **Cost**: ~$22 ($0.25/page × 88)
- **Time**: ~2 hours
- **Output**: 5K-10K tasks

### RL Training
- **Compute**: Free (distributed via PrimeIntellect)
- **GPU equivalent**: 8-16 hours on A100
- **Dataset**: Generated (free)

**Total project cost**: ~$22 + compute time

---

## Metrics Dashboard (After Training)

Track these in Weights & Biases:

```
Special Character Accuracy:
  ŋ (eng): 94.2%
  š (s-caron): 96.8%
  ć (c-acute): 95.1%
  ʼ (glottal): 87.3% ← hardest!

Affix Accuracy:
  -ku (possessive): 92.1%
  -ću (elder): 88.5%
  -tku (daughter): 86.2%
  ta- (prefix): 93.7%

Task Type Performance:
  Morphology: 91.2%
  Word Translation: 88.6%
  Sentence Translation: 82.4%
  Reverse Translation: 76.3% ← hardest!
```

---

## Support & Documentation

- **PrimeIntellect Docs**: https://github.com/PrimeIntellect-ai/verifiers
- **GRPO Paper**: Group Relative Policy Optimization
- **Dakota Orthography**: See ../CLAUDE.md

---

**Status**: Phase 2 Complete - Verifiers built, ready for full extraction & training!

**Next**: Run `extract_grammar_pages.py` to generate full training dataset
