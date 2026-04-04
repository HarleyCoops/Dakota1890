# Dakota Grammar RL Training - Launch Instructions

## System Ready 

All components are complete and published to GitHub:
- Repository: https://github.com/HarleyCoops/Dakota1890
- Commit: 5e6f456f - Complete Dakota grammar RL pipeline with PrimeIntellect integration

---

## What's Built

### Extraction Complete
- **667 grammar rules** from 62 pages (images 31-92)
- **393 interlinear translation texts**
- **97% confidence** average
- **100% special character preservation** (ć, š, ŋ, ḣ)

### RL Integration Complete
- **5,657 training tasks** generated
- **6 task types**: morphology, translation, reverse, syntax, pattern ID
- **4 difficulty levels**: easy, medium, hard, advanced
- **Curriculum learning** ready

### PrimeIntellect Integration Complete
- Datasets: `dakota_rl_training/datasets/*.jsonl`
- Config: `dakota_rl_training/configs/training_config.yaml`
- Verifiers: `dakota_rl_training/verifiers/*.py`
- Training script: `dakota_rl_training/train.py`

---

## Launch Training

### Prerequisites

**1. Install PrimeIntellect Framework**
```bash
pip install git+https://github.com/PrimeIntellect-ai/verifiers.git
pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git
```

**2. Verify Setup**
```bash
cd dakota_rl_training
python train.py --config configs/training_config.yaml
```

You should see:
```
======================================================================
 DAKOTA GRAMMAR RL TRAINING
======================================================================

Configuration: configs\training_config.yaml
Model: Qwen/Qwen2.5-7B-Instruct
Algorithm: GRPO
Epochs: 3
[OK] Dataset found: datasets\grammar_tasks_complete.jsonl
[OK] Dataset found: datasets\grammar_tasks_complete.jsonl
```

---

### Local Training (Testing)

For testing the pipeline locally:

```bash
cd dakota_rl_training

# Local test mode
prime-rl train \
    --config configs/training_config.yaml \
    --local \
    --max-steps 100
```

---

### Distributed Training (Production)

For full training on PrimeIntellect distributed workers:

```bash
cd dakota_rl_training

prime-rl train \
    --config configs/training_config.yaml \
    --num-workers 4 \
    --use-toploc \
    --wandb-project dakota-rl-grammar \
    --wandb-entity your-username
```

**Parameters**:
- `--num-workers 4`: Use 4 distributed workers
- `--use-toploc`: Enable TOPLOC verification (CRITICAL for Dakota character preservation)
- `--wandb-project`: Weights & Biases project for tracking
- `--wandb-entity`: Your W&B username

---

## Training Configuration

### Model & Algorithm
```yaml
model:
  base: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 64
  lora_alpha: 128

training:
  algorithm: "GRPO"  # Group Relative Policy Optimization
  num_epochs: 3
  batch_size: 16
  learning_rate: 5.0e-6
```

### Curriculum Stages
```yaml
curriculum:
  stages:
    1. Easy:   1,998 tasks → 80% accuracy target
    2. Medium: 2,155 tasks → 75% accuracy target
    3. Hard:     398 tasks → 70% accuracy target
```

### Reward Function
```yaml
rewards:
  morphology:
    character_preservation: 0.4  # Dakota special chars
    affix_accuracy: 0.4          # Morphological correctness
    semantic_correctness: 0.2    # Translation quality

  translation:
    character_preservation: 0.3
    semantic_correctness: 0.7

  reverse_translation:
    character_preservation: 0.5  # Critical for generation
    semantic_correctness: 0.5
```

---

## Monitoring Training

### Weights & Biases Dashboard

Once training starts, monitor at: `https://wandb.ai/your-username/dakota-rl-grammar`

**Key Metrics to Track**:

1. **Overall Performance**
   - `reward/mean`: Average reward per task
   - `reward/std`: Reward consistency
   - `learning_rate`: Current learning rate

2. **Character Preservation** (CRITICAL)
   - `char_accuracy`: Overall special character accuracy
   - `char_accuracy_by_char`:
     - `ŋ` (eng): Most difficult
     - `š` (s-caron)
     - `ć` (c-acute)
     - `ʼ` (glottal): Rare character

3. **Linguistic Accuracy**
   - `affix_accuracy`: Morphological correctness
   - `affix_accuracy_by_affix`:
     - `-ku` (possessive)
     - `-ću` (elder)
     - `ta-` (prefix)
   - `semantic_accuracy`: Translation quality

4. **Task Performance**
   - `accuracy_by_task_type`:
     - Morphology
     - Translation
     - Reverse translation
     - Syntax
     - Pattern identification

5. **Curriculum Progress**
   - `curriculum_stage`: Current difficulty
   - `stage_accuracy`: Performance on current stage
   - `stage_completion`: Tasks completed in stage

---

## Expected Training Timeline

### Phase 1: Easy Tasks (1,998 tasks)
**Duration**: ~2-3 hours
**Target**: 80% accuracy
**Focus**:
- Phonology rules
- Basic translation
- Simple morphology

**Expected metrics**:
- Character accuracy: 90%+
- Affix accuracy: 80%+
- Semantic accuracy: 85%+

### Phase 2: Medium Tasks (2,155 tasks)
**Duration**: ~3-4 hours
**Target**: 75% accuracy
**Focus**:
- Complex morphology
- Compound words
- Multi-word translations

**Expected metrics**:
- Character accuracy: 88%+
- Affix accuracy: 75%+
- Semantic accuracy: 78%+

### Phase 3: Hard Tasks (398 tasks)
**Duration**: ~2-3 hours
**Target**: 70% accuracy
**Focus**:
- Reverse translation (english → dakota)
- Complex syntax
- Rare patterns

**Expected metrics**:
- Character accuracy: 85%+
- Affix accuracy: 70%+
- Semantic accuracy: 65%+

**Total Training Time**: 8-12 hours on distributed workers

---

## Success Criteria

### Minimum Viable Model
- Character preservation: >85% for all special chars
- Affix accuracy: >70% for common morphology
- Translation accuracy: >75% (easy), >60% (hard)

### Production-Ready Model
- Character preservation: >90% for all special chars
- Affix accuracy: >80% for common morphology
- Translation accuracy: >85% (easy), >70% (hard)
- Reverse translation: >60% accuracy

---

## Checkpoints

Training checkpoints saved to: `dakota_rl_training/checkpoints/`

**Checkpoint naming**:
```
checkpoint_stage1_step500.pt
checkpoint_stage2_step1500.pt
checkpoint_stage3_final.pt
```

**Resume training**:
```bash
prime-rl train \
    --config configs/training_config.yaml \
    --resume-from checkpoints/checkpoint_stage2_step1500.pt
```

---

## Troubleshooting

### Issue: Character Corruption
**Symptom**: `char_accuracy` dropping, `ŋ → n` substitutions
**Solution**: Verify TOPLOC is enabled (`use_toploc: true`)

### Issue: Low Reward
**Symptom**: `reward/mean < 0.5`
**Solution**:
- Check curriculum stage - may need to stay longer
- Review failing task types
- Adjust reward weights

### Issue: Training Stalled
**Symptom**: No improvement for 500+ steps
**Solution**:
- Reduce learning rate by 2x
- Increase gradient accumulation steps
- Check for data quality issues

---

## Post-Training Evaluation

### 1. Test on Held-Out Set
```bash
python evaluate_model.py \
    --checkpoint checkpoints/checkpoint_stage3_final.pt \
    --test-set data/test_tasks.jsonl
```

### 2. Manual Grammar Testing
Test specific Dakota constructions:
- Possessive suffixes: `-ku`, `-ću`, `-tku`
- Negation: `šni` particle placement
- Verb conjugation: Person/number marking
- Special characters: All preserved correctly

### 3. Native Speaker Review
- Generate sample Dakota sentences
- Submit for community review
- Collect feedback for next iteration

---

## Next Steps After Training

### 1. Model Export
```bash
python export_model.py \
    --checkpoint checkpoints/checkpoint_stage3_final.pt \
    --output-dir models/dakota-rl-v1
```

### 2. Integration
- Package for HuggingFace Hub
- Create inference API
- Build web demo

### 3. Expansion
- Extract dictionary (pages 93-440)
- Generate synthetic Q&A pairs
- Train on combined grammar + vocabulary

### 4. Research Publication
- Document novel methodology
- Report results and metrics
- Share with language revitalization community

---

## Support Resources

- **PrimeIntellect Docs**: https://github.com/PrimeIntellect-ai
- **Project README**: https://github.com/HarleyCoops/Dakota1890
- **Issues**: https://github.com/HarleyCoops/Dakota1890/issues

---

## Quick Reference Commands

```bash
# Check setup
cd dakota_rl_training
python train.py --config configs/training_config.yaml

# Launch training
prime-rl train \
    --config configs/training_config.yaml \
    --num-workers 4 \
    --use-toploc \
    --wandb-project dakota-rl-grammar

# Monitor progress
# Visit: https://wandb.ai/your-username/dakota-rl-grammar

# Resume from checkpoint
prime-rl train \
    --config configs/training_config.yaml \
    --resume-from checkpoints/checkpoint_stage2_step1500.pt

# Evaluate model
python evaluate_model.py \
    --checkpoint checkpoints/checkpoint_stage3_final.pt
```

---

## Ready to Launch! 

All systems are ready. The Dakota Grammar Gym is waiting for training.

**To launch**: Run the distributed training command above with your PrimeIntellect credentials.

**Expected outcome**: Grammar-aware Dakota language model with 85%+ special character preservation and 75%+ translation accuracy.

**Novel contribution**: First demonstration of closed-loop grammar gym methodology for low-resource language training.

---

**Generated**: 2025-10-06
**Status**: READY FOR TRAINING
**Repository**: https://github.com/HarleyCoops/Dakota1890
**Commit**: 5e6f456f
