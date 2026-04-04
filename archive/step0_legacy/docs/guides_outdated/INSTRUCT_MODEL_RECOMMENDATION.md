# Should You Use an "Instruct" Model?

## Quick Answer

**YES - Always use instruction-tuned models for RL training!**

However, **Qwen3-30B-A3B may already be instruction-tuned** (Qwen3 series often are).

## What Are Instruct Models?

### Base Models (Not Instruction-Tuned)
- Raw pre-trained models
- Poor at following instructions
- Don't understand chat format
-  **Not suitable for RL training**

### Instruct Models (Instruction-Tuned)
- Fine-tuned on instruction-following data
- Understand prompts and chat format
- Better baseline performance
-  **Essential for RL training**

## For Qwen3-30B-A3B

### Available Model:
- **qwen/qwen3-30b-a3b** (no "-Instruct" variant shown)

### Qwen3 Series Behavior:
- **Qwen3 models are often instruction-tuned by default**
- A3B suffix = Architecture variant (may include instruction tuning)
- Check HuggingFace/Qwen docs to confirm

### Evidence:
- Prime RL framework uses `Qwen/Qwen3-30B-A3B` (without "-Instruct")
- This suggests it's already instruction-tuned
- Framework wouldn't use base model for RL

## For Qwen2.5 Series (Comparison)

### Explicit Variants:
- **Qwen2.5-7B**: Base model (not instruction-tuned)
- **Qwen2.5-7B-Instruct**: Instruction-tuned 

Your config uses `Qwen/Qwen2.5-7B-Instruct` - correct!

## Recommendation

### Use: `qwen/qwen3-30b-a3b`

**Why:**
-  Likely already instruction-tuned (Qwen3 series default)
-  Matches Prime RL framework config (proven to work)
-  No "-Instruct" variant available = probably already instruction-tuned

**If you want to verify:**
1. Check Qwen documentation
2. Run small eval - if it follows prompts well, it's instruction-tuned
3. Compare to explicit "-Instruct" variant if one exists

## Why Instruct Models Matter for RL

### For Your Dakota Grammar Tasks:

 **System prompts**: "You are a Dakota language expert..."
- Instruct models understand this format
- Base models may ignore it

 **Chat format**: system + user messages
- Instruct models trained for this
- Base models don't understand chat

 **Better baseline**: 
- Higher starting rewards
- Faster RL convergence
- More stable training

 **Prompt following**:
- Tasks need model to follow instructions
- Instruct models are better at this

## Bottom Line

**Use `qwen/qwen3-30b-a3b`** - it's likely already instruction-tuned!

If you're unsure:
1. Test with small eval (5 examples)
2. If model follows prompts well → it's instruction-tuned 
3. If it ignores prompts → look for "-Instruct" variant

**For RL training, always use instruction-tuned models!**

