# Dakota Grammar RL Training Setup

## What We Built

A complete pipeline for extracting Dakota grammar rules from pages 1-88 and converting them into Reinforcement Learning training tasks for PrimeIntellect.

### New Files Created

#### 1. **Grammar Schema** (`blackfeet_extraction/schemas/grammar_schema.py`)
- `GrammarRule` dataclass: Stores testable grammar rules
- `MorphologicalTransformation`: Stores affix transformations (root → inflected)
- `InterlinearExample`: Word-by-word aligned translations
- `GrammarPage`: Complete page extraction
- **Key Feature**: Automatic RL task generation with `generate_rl_tasks()`

#### 2. **Grammar Extraction Prompt** (`blackfeet_extraction/core/grammar_extraction_prompt.py`)
- Specialized prompt for extracting grammar rules (vs dictionary definitions)
- Focuses on:
  - Morphological patterns (prefix/suffix rules)
  - Testable transformations
  - Verification criteria for RL rewards
  - Multi-turn dialogue examples
- **Functions**:
  - `build_grammar_extraction_prompt()`: General grammar extraction
  - `build_focused_rule_extraction_prompt()`: Target specific rule types

#### 3. **Grammar Page Processor** (`blackfeet_extraction/core/grammar_page_processor.py`)
- Claude API wrapper optimized for grammar pages
- Processes pages 1-88 with grammar-specific schema
- **Key Method**: `process_page_range()` for batch processing
- Automatic validation and RL task generation

#### 4. **Test Script** (`test_grammar_extraction.py`)
- Tests extraction on page 61 (interlinear translations)
- Validates grammar rules
- Generates RL tasks in JSONL format
- Provides statistics on task types and special characters

---

## How It Works

### Step 1: Extract Grammar Rules from Historical Text

```python
from blackfeet_extraction.core.grammar_page_processor import GrammarPageProcessor

processor = GrammarPageProcessor()
grammar_page = processor.process_page(
    image_path=Path("page_061.jpg"),
    page_number=61,
    page_context="Chapter IX: Possessive Forms"
)

# grammar_page contains:
# - grammar_rules: List[GrammarRule]
# - interlinear_examples: List[InterlinearExample]
# - linguistic_notes: str
```

**Example Extracted Rule**:
```json
{
  "rule_id": "dakota_possessive_01",
  "rule_name": "Third-person possessive suffix -ku",
  "pattern": "{noun} + -ku → his/her {noun}",
  "transformations": [
    {
      "base_form": "iŋhiŋ",
      "transformed_form": "éiŋhiŋtku",
      "affixes": ["é-", "-ku"],
      "gloss_base": "son",
      "gloss_transformed": "his son"
    }
  ],
  "verification_criteria": [
    "Suffix -ku present",
    "Special characters preserved",
    "Meaning includes possessive"
  ]
}
```

### Step 2: Generate RL Training Tasks

```python
# Automatically generate tasks from extracted rules
tasks = grammar_page.generate_all_rl_tasks()

# Returns tasks in PrimeIntellect verifiers format:
[
  {
    "prompt": "Apply possessive -ku to 'iŋhiŋ' (son)",
    "answer": "éiŋhiŋtku",
    "info": {
      "rule_id": "dakota_possessive_01",
      "required_affixes": ["é-", "-ku"],
      "special_chars": ["ŋ"],
      "verification_criteria": ["Suffix -ku present", ...]
    }
  }
]
```

### Step 3: Use with PrimeIntellect Verifiers

The generated tasks are ready for PrimeIntellect's GRPO training:

```python
# dakota_verifiers/grammar_env.py (Next Phase)
import verifiers as vf

class DakotaGrammarEnv(vf.MultiTurnEnv):
    async def is_completed(self, messages, state, **kwargs):
        return state.get("rule_applied") and state.get("chars_correct")

    async def env_response(self, messages, state, **kwargs):
        # Verify special characters
        expected_chars = kwargs["info"]["special_chars"]
        chars_correct = all(c in messages[-1]["content"] for c in expected_chars)

        # Verify affixes
        required_affixes = kwargs["info"]["required_affixes"]
        affixes_correct = all(a in messages[-1]["content"] for a in required_affixes)

        new_state = {
            "chars_correct": chars_correct,
            "rule_applied": affixes_correct
        }

        return [{"role": "system", "content": feedback}], new_state
```

---

## Task Types Generated

### 1. **Morphological Application**
- **Prompt**: "Apply possessive -ku to 'iŋhiŋ'"
- **Answer**: "éiŋhiŋtku"
- **Reward**: Character accuracy + affix presence

### 2. **Word Translation**
- **Prompt**: "Translate Dakota word to English: éiŋhiŋtku"
- **Answer**: "his son"
- **Reward**: Semantic accuracy

### 3. **Sentence Translation**
- **Prompt**: "Translate: Wićašta wań éiŋhiŋtku nonpa"
- **Answer**: "A man had two sons"
- **Reward**: Multi-word alignment

### 4. **Reverse Translation**
- **Prompt**: "Translate to Dakota: his son"
- **Answer**: "éiŋhiŋtku"
- **Reward**: Character preservation + morphology

### 5. **Character Corruption Detection**
- **Prompt**: "Fix corrupted Dakota: 'einhintku'"
- **Answer**: "éiŋhiŋtku"
- **Reward**: Special character accuracy

---

## Quick Start Guide

### Setup

1. **Create `.env` file**:
```bash
cp .env.template .env
```

2. **Add your API key to `.env`**:
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

3. **Install dependencies** (if not already done):
```bash
pip install anthropic pillow python-dotenv
```

### Test Grammar Extraction

```bash
# Extract grammar from page 61 (interlinear translations)
python test_grammar_extraction.py
```

**Expected Output**:
```
 Grammar extraction successful!
 Generated 30+ RL training tasks
 Results saved to data/grammar_test/
```

**Output Files**:
- `data/grammar_test/grammar_page_061.json` - Full extraction
- `data/grammar_test/rl_tasks_page_061.jsonl` - RL tasks

### Extract Full Grammar Section (Pages 1-88)

```python
from pathlib import Path
from blackfeet_extraction.core.grammar_page_processor import GrammarPageProcessor

processor = GrammarPageProcessor()

grammar_pages = processor.process_page_range(
    image_dir=Path("data/processed_images"),
    start_page=1,
    end_page=88,
    output_dir=Path("data/grammar_extraction")
)

# Generate all RL tasks
all_tasks = []
for page in grammar_pages:
    tasks = page.generate_all_rl_tasks()
    all_tasks.extend(tasks)

# Save to JSONL for PrimeIntellect
with open("data/training_datasets/dakota_grammar_tasks.jsonl", "w") as f:
    for task in all_tasks:
        f.write(json.dumps(task, ensure_ascii=False) + "\n")
```

**Estimated Results**:
- ~88 pages
- ~200-300 grammar rules
- ~1000-2000 interlinear examples
- **~5000-10000 RL training tasks**

**Cost**: ~88 pages × $0.25/page = **~$22 total**

---

## Next Steps

### Phase 2: Build PrimeIntellect Verifiers (Ready to implement)

Create `dakota_rl_training/verifiers/grammar_env.py`:

```python
import verifiers as vf

class DakotaGrammarEnv(vf.MultiTurnEnv):
    """Multi-turn environment for Dakota grammar learning"""

    async def is_completed(self, messages, state, **kwargs):
        # Task complete when rule applied correctly
        return (
            state.get("rule_applied") and
            state.get("special_chars_correct")
        ) or state.get("attempts", 0) >= 3

    async def env_response(self, messages, state, **kwargs):
        last_response = messages[-1]["content"]

        # Verify against expected answer
        expected = kwargs["answer"]
        correct = last_response.strip() == expected.strip()

        # Check special characters
        expected_chars = kwargs["info"]["special_chars"]
        chars_correct = all(c in last_response for c in expected_chars)

        new_state = {
            "rule_applied": correct,
            "special_chars_correct": chars_correct,
            "attempts": state.get("attempts", 0) + 1
        }

        # Generate feedback
        if correct:
            feedback = " Correct! Rule applied properly."
        elif not chars_correct:
            missing = set(expected_chars) - set(last_response)
            feedback = f"Missing special characters: {missing}"
        else:
            feedback = "Incorrect transformation. Check affixes."

        return [{"role": "system", "content": feedback}], new_state
```

### Phase 3: Training Configuration

```yaml
# configs/dakota_grammar_training.yaml
model:
  base: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 64

training:
  framework: "prime-rl"
  algorithm: "GRPO"
  num_epochs: 3
  batch_size: 16

environments:
  - name: "dakota_grammar"
    type: "MultiTurnEnv"
    dataset: "data/training_datasets/dakota_grammar_tasks.jsonl"
    rubric: "DakotaGrammarRubric"
    max_turns: 3

rewards:
  character_preservation: 0.4
  morphology_accuracy: 0.4
  semantic_correctness: 0.2
```

### Phase 4: Run Distributed Training

```bash
# Using PrimeIntellect's prime-rl framework
python dakota_rl_training/train.py \
    --config configs/dakota_grammar_training.yaml \
    --output-dir models/dakota-qwen-7b-grammar \
    --distributed
```

---

## Research Opportunities

This setup enables novel research:

1. **Character Learning in RL**
   - How do models learn rare Unicode characters through RL?
   - Does RL help preserve special characters better than SFT?

2. **Morphological Composition**
   - Can models learn to decompose Dakota words into morphemes?
   - Do attention patterns reflect morpheme boundaries?

3. **Low-Resource Language Learning**
   - Compare RL vs supervised learning for Dakota
   - Measure sample efficiency with synthetic tasks

4. **Curriculum Learning**
   - Does difficulty progression (basic → expert) improve learning?
   - Optimal task sequencing for morphology acquisition

5. **Verifiable Character Preservation**
   - Using TOPLOC for distributed verification of Unicode accuracy
   - Preventing character corruption in untrusted workers

---

## Files Overview

```
Dakota1890/
├── blackfeet_extraction/
│   ├── core/
│   │   ├── grammar_extraction_prompt.py  ← Prompt engineering for grammar
│   │   └── grammar_page_processor.py     ← Claude API wrapper
│   └── schemas/
│       └── grammar_schema.py             ← Data structures + RL task gen
├── test_grammar_extraction.py            ← Test script
├── GRAMMAR_RL_SETUP.md                   ← This file
└── data/
    ├── grammar_test/                     ← Test outputs
    │   ├── grammar_page_061.json
    │   └── rl_tasks_page_061.jsonl
    └── grammar_extraction/               ← Full extraction (when run)
        ├── grammar_page_001.json
        ├── grammar_page_002.json
        └── ...
```

---

## What Makes This Novel

1. **First RL Environment for Indigenous Language Morphology**
   - No existing work on RL for Dakota grammar
   - Verifiable character preservation critical

2. **Synthetic Task Generation from Historical Texts**
   - Extracting testable rules from 1890s grammar books
   - Automatic conversion to RL tasks

3. **Multi-Turn Grammatical Correction**
   - Progressive feedback on morphological errors
   - Curriculum from basic → expert

4. **Distributed Verifiable Training**
   - PrimeIntellect's TOPLOC ensures character accuracy
   - Critical for special character preservation

5. **Morpheme-Level Rewards**
   - Reward correct affix application
   - Encourage compositional understanding

---

## Cost Estimates

### Grammar Extraction (Pages 1-88)
- **Pages**: 88
- **Cost per page**: ~$0.25 (Claude Sonnet 4.5)
- **Total cost**: ~$22
- **Time**: ~1-2 hours
- **Output**: 5000-10000 RL tasks

### RL Training (with PrimeIntellect)
- **Base model**: Qwen2.5-7B-Instruct (open source)
- **Training**: Free (distributed compute)
- **GPU equivalent**: ~8-16 hours on single A100
- **Dataset**: Generated tasks (free)

**Total project cost**: **~$22 for extraction + compute time**

---

## Contact & Support

- **PrimeIntellect Docs**: https://github.com/PrimeIntellect-ai/verifiers
- **GRPO Paper**: Group Relative Policy Optimization
- **Dakota Language**: See CLAUDE.md for orthography details

---

**Status**: Phase 1 complete - Ready to extract grammar and build verifiers!
