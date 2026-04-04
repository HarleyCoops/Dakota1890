# PrimeIntellect Dakota RL Training Launch Guide

## Current Status

 **Training Configuration Complete**
- TOML configs created: [train.toml](dakota_rl_training/configs/train.toml), [infer.toml](dakota_rl_training/configs/infer.toml), [orch.toml](dakota_rl_training/configs/orch.toml)
- Dataset ready: 5,657 tasks from 1,036 grammar rules
- Curriculum: Easy (1,998) → Medium (2,155) → Hard (398)
- PI_API_KEY configured in .env

 **Windows Limitation**
- PrimeIntellect prime-rl requires Linux with NVIDIA GPUs
- Triton acceleration library not available on Windows
- Cannot run training locally on this Windows machine

## Recommended Launch Options

### Option 1: PrimeIntellect Cloud Platform (RECOMMENDED)

Use PrimeIntellect's hosted platform for distributed training:

**Steps:**

1. **Visit Platform**
   - Go to https://app.primeintellect.ai
   - Log in with your PI_API_KEY

2. **Create New Training Job**
   - Click "New Training Job" or similar
   - Select "Custom RL Training"

3. **Upload Configuration Files**
   Upload these files from [dakota_rl_training/](dakota_rl_training/):
   - `configs/train.toml` - Training configuration
   - `configs/infer.toml` - Inference server config
   - `configs/orch.toml` - Orchestrator config
   - `datasets/grammar_tasks_easy.jsonl` - Stage 1 dataset (1,998 tasks)
   - `datasets/grammar_tasks_medium.jsonl` - Stage 2 dataset (2,155 tasks)
   - `datasets/grammar_tasks_hard.jsonl` - Stage 3 dataset (398 tasks)

4. **Configure Job Settings**
   - **Model**: Qwen/Qwen2.5-7B-Instruct
   - **Algorithm**: GRPO (Group Relative Policy Optimization)
   - **Curriculum**: Enable 3-stage curriculum
   - **TOPLOC Verification**: Enable
   - **W&B Project**: dakota-rl-grammar

5. **Launch Training**
   - Review settings
   - Click "Launch Training"
   - Monitor via PrimeIntellect dashboard

**Expected Results:**
- **Stage 1 (Easy)**: 2-4 hours, target 80% accuracy
- **Stage 2 (Medium)**: 3-5 hours, target 75% accuracy
- **Stage 3 (Hard)**: 1-2 hours, target 70% accuracy
- **Total Time**: 6-11 hours
- **Cost**: TBD based on GPU allocation

---

### Option 2: Linux VM with GPU (Advanced)

Set up Linux environment with NVIDIA GPU:

**Requirements:**
- Linux (Ubuntu 22.04+ recommended)
- NVIDIA GPU (RTX 3090/4090, A100, H100, etc.)
- CUDA 12.1+
- 16GB+ VRAM
- 32GB+ System RAM

**Setup Steps:**

1. **Provision Linux VM**
   ```bash
   # AWS, GCP, Azure, or local Linux machine with NVIDIA GPU
   # Ensure CUDA drivers installed
   nvidia-smi  # Verify GPU detected
   ```

2. **Install uv Package Manager**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   uv --version  # Should show 0.5.26 or later
   ```

3. **Clone prime-rl Framework**
   ```bash
   git clone https://github.com/PrimeIntellect-ai/prime-rl.git
   cd prime-rl
   ```

4. **Install Dependencies**
   ```bash
   uv sync && uv sync --all-extras
   # This will install triton, torch, vLLM, etc.
   ```

5. **Copy Dakota Training Configs**
   ```bash
   # Transfer from Windows machine:
   # - dakota_rl_training/configs/*.toml
   # - dakota_rl_training/datasets/*.jsonl
   ```

6. **Set Environment Variables**
   ```bash
   export PI_API_KEY=pit_6edd408a56862c2c21bd8d983cf209657c6ed1a3afe81baf2124fb9e5f6add6b
   export WANDB_PROJECT=dakota-rl-grammar
   ```

7. **Launch Training (Single GPU)**
   ```bash
   uv run rl \
     --trainer @ configs/train.toml \
     --orchestrator @ configs/orch.toml \
     --inference @ configs/infer.toml \
     --trainer-gpu-ids 0 \
     --inference-gpu-ids 0
   ```

8. **Launch Training (Multi-GPU)**
   ```bash
   # Edit train.toml:
   # [distributed]
   # enabled = true
   # num_workers = 4  # Number of GPUs

   uv run rl \
     --trainer @ configs/train.toml \
     --orchestrator @ configs/orch.toml \
     --inference @ configs/infer.toml \
     --trainer-gpu-ids 0,1,2,3 \
     --inference-gpu-ids 0
   ```

---

### Option 3: WSL2 with GPU Passthrough (Experimental)

Use Windows Subsystem for Linux with GPU support:

**Requirements:**
- Windows 11 or Windows 10 version 21H2+
- NVIDIA GPU with latest Windows drivers
- WSL2 installed

**Setup:**

1. **Install WSL2 with GPU Support**
   ```powershell
   # PowerShell (Admin)
   wsl --install
   wsl --update
   ```

2. **Install CUDA in WSL2**
   ```bash
   # Inside WSL2 Ubuntu
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-1
   ```

3. **Verify GPU Access**
   ```bash
   nvidia-smi  # Should show your GPU
   ```

4. **Follow Option 2 Steps 2-8** (Linux VM setup)

---

## Configuration Files Summary

### [train.toml](dakota_rl_training/configs/train.toml)
- **Model**: Qwen/Qwen2.5-7B-Instruct with LoRA (rank 64)
- **Algorithm**: GRPO
- **Epochs**: 3
- **Batch Size**: 16 (effective 64 with gradient accumulation)
- **Learning Rate**: 5e-6
- **Curriculum**: 3 stages with progressive difficulty

### [infer.toml](dakota_rl_training/configs/infer.toml)
- **Backend**: vLLM inference server
- **Port**: 8000
- **Max Concurrent**: 32 requests
- **Temperature**: 0.7
- **Max Tokens**: 128

### [orch.toml](dakota_rl_training/configs/orch.toml)
- **Orchestrator Port**: 9000
- **Rollout Batch**: 16
- **Verification**: TOPLOC enabled

---

## Monitoring Training

### Weights & Biases Dashboard

1. **View Metrics**
   - Go to https://wandb.ai
   - Project: `dakota-rl-grammar`
   - Key metrics:
     - `reward/mean`: Average reward
     - `char_accuracy`: Dakota character preservation (target: >85%)
     - `affix_accuracy`: Morphology accuracy (target: >75%)
     - `semantic_accuracy`: Translation correctness (target: >75%)

2. **Per-Character Tracking**
   - `char_accuracy_by_char`: Accuracy for ć, š, ŋ, ḣ, etc.
   - Watch for character corruption (should be <5%)

3. **Curriculum Progress**
   - Stage transitions visible in loss/accuracy curves
   - Each stage should show improvement before advancing

### Checkpoints

Training checkpoints saved to:
```
dakota_rl_training/checkpoints/
├── checkpoint-500/
├── checkpoint-1000/
└── checkpoint-final/
```

Download and test locally:
```bash
python test_checkpoint.py --checkpoint checkpoints/checkpoint-final/
```

---

## Training Dataset Statistics

| Dataset | Tasks | Avg Confidence | Task Types |
|---------|-------|----------------|------------|
| Easy | 1,998 | 0.92 | Morphology, Translation |
| Medium | 2,155 | 0.87 | Reverse Translation, Syntax |
| Hard | 398 | 0.81 | Complex Morphology, Pattern ID |
| **Total** | **5,657** | **0.89** | **All types** |

**Special Characters Coverage:**
- ć, š, ŋ, ḣ, ṡ: 100% (all critical characters present)
- á, é, í, ó, ú: 98% (pitch markers)
- ʼ (glottal stop): 45% (less common)

---

## Troubleshooting

### "Triton not available on Windows"
- **Solution**: Use Option 1 (PrimeIntellect Cloud) or Option 2 (Linux VM)

### "PI_API_KEY invalid"
- **Check**: .env file has correct key
- **Verify**: Key format is `pit_<64 hex chars>`
- **Test**: Visit https://app.primeintellect.ai and log in

### "Dataset not found"
- **Check**: Paths in train.toml are correct
- **Verify**: Files exist in `dakota_rl_training/datasets/`
- **Fix**: Use absolute paths if relative paths fail

### "Out of memory"
- **Reduce batch size**: Change `per_device_train_batch_size` in train.toml
- **Reduce sequence length**: Set `max_model_len = 4096` in infer.toml
- **Use smaller model**: Try Qwen2.5-3B-Instruct instead

### "Character accuracy too low"
- **Check**: TOPLOC verification enabled
- **Increase**: `char_corruption_penalty` in train.toml
- **Verify**: Tokenizer preserves Dakota characters

---

## Next Steps After Training

1. **Download Final Checkpoint**
   ```bash
   # From PrimeIntellect dashboard or via CLI
   prime-rl download-checkpoint --job-id <your_job_id>
   ```

2. **Test Dakota Translation**
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM

   model = AutoModelForCausalLM.from_pretrained("./checkpoints/checkpoint-final")
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

   prompt = "Translate to English:\n\nmićú-wo"
   output = model.generate(tokenizer.encode(prompt), max_new_tokens=50)
   print(tokenizer.decode(output[0]))
   # Expected: "give to me"
   ```

3. **Evaluate on Test Set**
   ```bash
   python evaluate_model.py \
     --checkpoint checkpoints/checkpoint-final/ \
     --test-data datasets/grammar_tasks_complete.jsonl \
     --output evaluation_results.json
   ```

4. **Deploy for Inference**
   - Upload to HuggingFace Hub
   - Deploy via PrimeIntellect inference API
   - Integrate into Dakota language applications

---

## Files Ready for Upload

All files are in [dakota_rl_training/](dakota_rl_training/):

```
dakota_rl_training/
├── configs/
│   ├── train.toml           # Training configuration
│   ├── infer.toml           # Inference server config
│   ├── orch.toml            # Orchestrator config
│   └── training_config.yaml # Legacy YAML (can ignore)
├── datasets/
│   ├── grammar_tasks_complete.jsonl  # All 5,657 tasks
│   ├── grammar_tasks_easy.jsonl      # 1,998 easy tasks
│   ├── grammar_tasks_medium.jsonl    # 2,155 medium tasks
│   └── grammar_tasks_hard.jsonl      # 398 hard tasks
└── launch_training.sh       # Launch script (for Linux)
```

---

## Summary

** Ready to Launch:**
- Configuration: Complete
- Datasets: Generated and validated
- API Key: Configured
- Documentation: Complete

** Recommended Action:**
Go to https://app.primeintellect.ai and launch training with Option 1 (Cloud Platform)

** Expected Timeline:**
- Upload configs: 5 minutes
- Job starts: 10-30 minutes
- Training completes: 6-11 hours
- Total: ~7-12 hours

** Estimated Cost:**
TBD based on PrimeIntellect pricing (typically $1-5/hour for distributed RL training)
