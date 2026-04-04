# How to Train a Different Model with dakota1890 Environment

## Quick Answer

**YES!** You can train ANY open source model with the dakota1890 environment. The environment is model-agnostic.

## Steps

### 1. Install the Environment

```bash
prime env install <your-username>/dakota1890
# Or if already installed, skip this step
```

### 2. Update Training Config

Edit `dakota_rl_training/configs/training_config.yaml`:

**Change this line:**
```yaml
model:
  base: "Qwen/Qwen2.5-7B-Instruct"  # CHANGE THIS
```

**To any model you want:**

```yaml
# Examples:
model:
  base: "meta-llama/Llama-3.1-8B-Instruct"

model:
  base: "mistralai/Mistral-7B-Instruct-v0.2"

model:
  base: "microsoft/Phi-3-medium-4k-instruct"

model:
  base: "Qwen/Qwen2.5-14B-Instruct"  # Even bigger Qwen!

model:
  base: "path/to/your/existing/checkpoint"  # Your fine-tuned model!
```

### 3. The Environment Reference Stays the Same

In your config, reference the environment:

```yaml
environments:
  - name: "dakota1890"  # This is what we uploaded!
    dataset: "datasets/grammar_tasks_complete.jsonl"
    # ... rest of config
```

Or in code:

```python
from dakota_grammar_translation import load_environment

env = load_environment()
# Use env with ANY model via prime-rl
```

### 4. Run Training

```bash
python dakota_rl_training/train.py
```

The environment will evaluate whatever model you specify!

## Example: Training Llama Instead of Qwen

**Before:**
```yaml
model:
  base: "Qwen/Qwen2.5-7B-Instruct"
```

**After:**
```yaml
model:
  base: "meta-llama/Llama-3.1-8B-Instruct"
```

That's it! Everything else stays the same. The environment will evaluate Llama's responses to Dakota grammar tasks.

## Key Point

**The environment doesn't care about the model!**

- Environment = The test (dakota1890)
- Model = The student (Qwen, Llama, Mistral, etc.)
- You can swap students (models) without changing the test (environment)

## What Gets Uploaded vs What Gets Trained

**Uploaded to PrimeIntellect Hub:**
-  Environment code (dakota1890)
-  Reward functions
-  Task evaluation logic
-  NOT the model (models are separate)

**Trained Locally/On PrimeIntellect:**
-  Your chosen model (Qwen, Llama, etc.)
-  Model learns from environment rewards
-  Environment stays the same (it's the test!)

