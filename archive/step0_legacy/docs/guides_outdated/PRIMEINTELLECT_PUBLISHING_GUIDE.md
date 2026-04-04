# Publishing Dakota Environment to PrimeIntellect - Guide

## Overview

This document explains how to publish Dakota grammar rules and RL environment to PrimeIntellect's Environment Hub, following the same process used for Stoney Nakoda. **RL environments are distinct from SFT datasets** - they can be published independently and used for reinforcement learning training.

---

## Key Understanding: RL vs SFT

### SFT (Supervised Fine-Tuning)
- **Data Format**: JSONL files with prompt-answer pairs
- **Training**: Direct fine-tuning on examples
- **Currently**: `dakota_rl_training/datasets/*.jsonl` (5,657 tasks)

### RL (Reinforcement Learning)
- **Environment Format**: Python package with verifiers specification
- **Training**: Agent learns through reward feedback from environment
- **Current Status**: Environment code exists but not yet published
- **Components**:
  - Environment class (`DakotaGrammarEnv`)
  - Rubric/reward function (`DakotaGrammarRubric`)
  - Rules embedded in environment or loaded from JSON

**Key Point**: The rules and environment can be published separately and used together for RL training, independent of SFT datasets.

---

## PrimeIntellect Environment Publishing Process

Based on PrimeIntellect's documentation and the Stoney Nakoda approach:

### Step 1: Package Environment as Python Module

Create a proper Python package structure with `pyproject.toml`:

```
dakota-grammar-env/
├── pyproject.toml           # Package metadata and dependencies
├── README.md                 # Environment description
├── src/
│   └── dakota_grammar_env/
│       ├── __init__.py       # Exports environment and rubric
│       ├── environment.py    # DakotaGrammarEnv implementation
│       └── rubric.py         # DakotaGrammarRubric implementation
└── tests/
    └── test_environment.py  # Unit tests
```

### Step 2: Create `pyproject.toml`

The package definition file:

```toml
[project]
name = "dakota-grammar-env"
version = "0.1.0"
description = "Dakota language grammar RL environment for PrimeIntellect"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "verifiers>=0.1.4",  # PrimeIntellect verifiers framework
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "black>=23.0.0",
]

[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

### Step 3: Adapt Environment to Verifiers Specification

Ensure your environment classes match the `verifiers` interface:

**Current Implementation** (`dakota_rl_training/verifiers/grammar_env.py`):
-  Already implements `MultiTurnEnv` interface
-  Has `is_completed()` and `env_response()` methods
-  Uses async/await pattern (required)

**What Needs to Change**:
1. Import from `verifiers` package instead of local base classes
2. Ensure compatibility with verifiers API
3. Add proper error handling for distributed training

### Step 4: Package Rules Data

Rules can be:
- **Embedded**: Included in the Python package as JSON files
- **Loaded from Hub**: Published separately and referenced by environment
- **Hybrid**: Default rules included, additional rules can be loaded

**Recommended Structure**:
```
dakota-grammar-env/
├── src/
│   └── dakota_grammar_env/
│       ├── __init__.py
│       ├── environment.py
│       ├── rubric.py
│       └── data/
│           └── rules/
│               ├── all_rl_rules.json      # Default rules
│               └── rules_by_category.json # Organized rules
```

### Step 5: Upload to PrimeIntellect Environment Hub

Once packaged, upload using PrimeIntellect CLI:

```bash
# Install PrimeIntellect CLI
pip install prime-intellect-cli

# Login to PrimeIntellect
prime login

# Publish environment
prime env publish dakota-grammar-env/ \
  --owner your-username \
  --name dakota-grammar-env \
  --version 0.1.0
```

**Alternative**: Upload via Environment Hub UI:
1. Go to https://app.primeintellect.ai/dashboard/environments
2. Click "Publish Environment"
3. Upload package or connect GitHub repository

---

## Adopting Stoney Nakoda's Process for Dakota

### Current Dakota Status

** Already Built**:
- Environment classes: `DakotaGrammarEnv`, `DakotaMorphologyEnv`
- Rubric classes: `DakotaGrammarRubric`
- Rules data: `data/rl_training_rules/all_rl_rules.json` (1,085 rules)
- Task datasets: `dakota_rl_training/datasets/*.jsonl` (5,657 tasks)

** Not Yet Published**:
- Package structure with `pyproject.toml`
- Integration with verifiers package
- Upload to PrimeIntellect Hub

### Step-by-Step Adoption Plan

#### Phase 1: Package Structure Setup

1. **Create package directory**:
```bash
mkdir -p dakota-grammar-env/src/dakota_grammar_env/data/rules
```

2. **Create `pyproject.toml`** (see template above)

3. **Move environment code**:
```bash
cp dakota_rl_training/verifiers/grammar_env.py dakota-grammar-env/src/dakota_grammar_env/environment.py
cp dakota_rl_training/verifiers/rubrics.py dakota-grammar-env/src/dakota_grammar_env/rubric.py
cp data/rl_training_rules/all_rl_rules.json dakota-grammar-env/src/dakota_grammar_env/data/rules/
```

4. **Update imports** to use `verifiers` package:
```python
# In environment.py
from verifiers import MultiTurnEnv, SingleTurnEnv

# In rubric.py
from verifiers import Rubric
```

#### Phase 2: Environment Integration

1. **Update `__init__.py`** to export environment:
```python
"""Dakota Grammar Environment for PrimeIntellect RL Training"""

from .environment import DakotaGrammarEnv, DakotaMorphologyEnv
from .rubric import DakotaGrammarRubric

__all__ = ["DakotaGrammarEnv", "DakotaMorphologyEnv", "DakotaGrammarRubric"]
__version__ = "0.1.0"
```

2. **Add rules loading**:
```python
# In environment.py
import json
from pathlib import Path

class DakotaGrammarEnv(MultiTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Load rules from package data
        rules_path = Path(__file__).parent / "data" / "rules" / "all_rl_rules.json"
        if rules_path.exists():
            with open(rules_path, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
                self.rules = rules_data.get('rules', [])
        else:
            self.rules = []
```

#### Phase 3: Testing

1. **Install locally**:
```bash
cd dakota-grammar-env
pip install -e .
```

2. **Test import**:
```python
from dakota_grammar_env import DakotaGrammarEnv, DakotaGrammarRubric

env = DakotaGrammarEnv(max_turns=3)
rubric = DakotaGrammarRubric()
```

3. **Test with PrimeIntellect**:
```bash
python -c "import dakota_grammar_env; print(' Environment ready')"
```

#### Phase 4: Publish to PrimeIntellect

1. **Build package**:
```bash
cd dakota-grammar-env
pip install build
python -m build
```

2. **Upload to Environment Hub**:
```bash
prime env publish dist/ \
  --owner HarleyCoops \
  --name dakota-grammar-env \
  --version 0.1.0
```

3. **Verify publication**:
```bash
prime env info HarleyCoops/dakota-grammar-env
```

---

## Using Published Environment for RL Training

Once published, the environment can be used in RL training:

### Option 1: Install from Hub

```bash
prime env install HarleyCoops/dakota-grammar-env
# Or
uv pip install dakota-grammar-env --extra-index-url https://hub.primeintellect.ai/HarleyCoops/simple/
```

### Option 2: Use in Training Config

```toml
# train.toml
[model]
name_or_path = "Qwen/Qwen2.5-7B-Instruct"

[env]
# Use published environment
id = "HarleyCoops/dakota-grammar-env"
max_turns = 3

[trainer]
algorithm = "grpo"
num_epochs = 3
learning_rate = 5.0e-6

[dataset]
# Rules are embedded in environment
# Tasks can be loaded separately or generated by environment
train_file = "dakota_rl_training/datasets/grammar_tasks_easy.jsonl"
```

### Option 3: Launch Training

```bash
uv run rl \
  --trainer @ configs/train.toml \
  --orchestrator @ configs/orch.toml \
  --inference @ configs/infer.toml \
  --wandb-project dakota-rl-grammar
```

---

## Key Differences from SFT Workflow

| Aspect | SFT | RL (Environment Publishing) |
|--------|-----|---------------------------|
| **Data Format** | JSONL files | Python package with environment code |
| **Publishing** | HuggingFace Datasets Hub | PrimeIntellect Environment Hub |
| **Usage** | Direct fine-tuning | RL training with reward feedback |
| **Dependencies** | Dataset loading | Verifiers framework |
| **Rules** | Embedded in data | Embedded in environment or loaded separately |
| **Training** | Standard SFT loop | RL loop with environment interaction |

---

## Benefits of Publishing Environment

1. **Reusability**: Others can use your environment for RL training
2. **Versioning**: Track environment versions and improvements
3. **Sharing**: Community can contribute improvements
4. **Separation**: Rules and environment separate from training data
5. **Flexibility**: Same environment can be used with different models/data

---

## Next Steps

1. **Create package structure** following the template above
2. **Adapt code** to use verifiers package imports
3. **Test locally** before publishing
4. **Publish to Environment Hub** using PrimeIntellect CLI
5. **Document** environment usage and examples
6. **Share** with community for feedback

---

## Example: Complete Package Structure

```
dakota-grammar-env/
├── pyproject.toml              # Package definition
├── README.md                    # Environment documentation
├── LICENSE                      # License file
├── src/
│   └── dakota_grammar_env/
│       ├── __init__.py          # Package exports
│       ├── environment.py       # DakotaGrammarEnv
│       ├── rubric.py            # DakotaGrammarRubric
│       └── data/
│           └── rules/
│               ├── all_rl_rules.json      # 1,085 rules
│               ├── rules_morphology.json  # 346 rules
│               ├── rules_syntax.json      # 182 rules
│               └── ...
├── tests/
│   ├── test_environment.py
│   └── test_rubric.py
└── examples/
    └── usage.py                 # Example usage
```

---

## Resources

- **PrimeIntellect Environment Hub**: https://app.primeintellect.ai/dashboard/environments
- **Verifiers Documentation**: https://github.com/PrimeIntellect-ai/verifiers
- **Prime-RL Documentation**: https://github.com/PrimeIntellect-ai/prime-rl
- **Environment Publishing Guide**: https://docs.primeintellect.ai/tutorials-environments/environments

---

**Status**: Ready to package and publish
**Next Action**: Create package structure and adapt code to verifiers specification

