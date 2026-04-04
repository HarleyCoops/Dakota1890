# W&B Data Analysis Guide

## Overview

This guide explains how to pull and analyze data from your Dakota RL training runs in Weights & Biases.

The canonical refresh path is now:

```powershell
python scripts/analysis/refresh_tracking_assets.py
```

This script refreshes the maintained public W&B runs, inventories linked Hugging Face model repos, rebuilds the core PNG set in `wandb_visualizations/`, and repairs the generic reward-ledger artifacts expected by the README.

## Key Runs

### Orchestrator Run (Reward Data)
- **Run ID**: `29hn8w98`
- **URL**: https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/29hn8w98
- **Purpose**: Contains reward component data, environment interactions, and task-level metrics
- **Key Metrics**: Reward components, task success rates, environment statistics

### Trainer Run (Training Metrics)
- **Run ID**: `7nikv4vp`
- **URL**: https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/7nikv4vp
- **Purpose**: Contains model training metrics, loss, entropy, performance stats
- **Key Metrics**: Loss, entropy, inference probabilities, throughput, MFU

## Available Scripts

### 1. Canonical Refresh
```powershell
# Refresh all maintained public Dakota runs
python scripts/analysis/refresh_tracking_assets.py

# Refresh and sync the published 30B visual assets from Hugging Face
python scripts/analysis/refresh_tracking_assets.py --sync-hf-visuals
```

### 2. Export Single Run Data
```powershell
# Export orchestrator run (default)
python scripts/analysis/export_wandb_data.py

# Export specific run
python scripts/analysis/export_wandb_data.py --run-path "christian-cooper-us/dakota-rl-grammar/29hn8w98"

# Export trainer run
python scripts/analysis/export_wandb_data.py --run-path "christian-cooper-us/dakota-rl-grammar/7nikv4vp" --output-dir wandb_analysis/trainer
```

### 3. Comprehensive Reward Analysis
```powershell
# Analyze both runs (uses local data if available, fetches if needed)
python scripts/analysis/analyze_wandb_rewards.py

# Force fetch from W&B API
python scripts/analysis/analyze_wandb_rewards.py --force-fetch

# Use only local data (no API calls)
python scripts/analysis/analyze_wandb_rewards.py --local-only

# Analyze specific runs
python scripts/analysis/analyze_wandb_rewards.py --orchestrator-run 29hn8w98 --trainer-run 7nikv4vp
```

## Setup

1. **Install dependencies**:
   ```powershell
   pip install wandb pandas python-dotenv
   ```

2. **Optional W&B API key** in `.env`:
   ```
   WANDB_API_KEY=your_key_here
   ```
   Get your key from: https://wandb.ai/authorize

Public reads for the maintained Dakota runs currently work without a local W&B API key. A key is still useful if you want authenticated access beyond the public surfaces.

## Output Files

### Export Script Outputs
- `wandb_analysis/dakota_rl_wandb_export.json` - Full run data in JSON
- `wandb_analysis/dakota_rl_wandb_history.csv` - Time series data as CSV

### Analysis Script Outputs
- `wandb_analysis/reward_analysis_summary.json` - Comprehensive analysis summary
- `wandb_analysis/orchestrator_rewards.csv` - Reward metrics only
- `wandb_analysis/{run_id}/{run_id}_summary.json` - Run summary
- `wandb_analysis/{run_id}/{run_id}_history.csv` - Run history

### Canonical Refresh Outputs
- `wandb_analysis/tracking_inventory.json` - Combined W&B and Hugging Face inventory
- `wandb_analysis/hf_repo_inventory.json` - Hugging Face repo file inventory
- `wandb_analysis/reward_ledger.csv` - Canonical ledger CSV used by the README
- `wandb_analysis/reward_ledger.png` - Canonical ledger figure used by the README
- `wandb_visualizations/*.png` - refreshed hero figures for the 1000-step run
- `wandb_visualizations/qwen4b/*.png` - refreshed figures for the 400-step ledger run

## What Data is Available

### From Orchestrator Run (29hn8w98)
- **Reward Components**: Individual reward function components
- **Task Metrics**: Success rates, task completion statistics
- **Environment Stats**: Episode lengths, environment interactions
- **Rollout Data**: Per-rollout metrics and statistics

### From Trainer Run (7nikv4vp)
- **Training Loss**: Mean, std, min, max loss values
- **Entropy**: Model entropy statistics
- **Probabilities**: Inference and trainer probability distributions
- **Performance**: Throughput (tokens/sec), MFU, memory usage
- **Optimization**: Learning rate, gradient norms

## Quick Start

1. **Export orchestrator run** (contains rewards):
   ```powershell
   python scripts/analysis/export_wandb_data.py --run-path "christian-cooper-us/dakota-rl-grammar/29hn8w98"
   ```

2. **Run comprehensive analysis**:
   ```powershell
   python scripts/analysis/analyze_wandb_rewards.py
   ```

3. **Check results**:
   - Open `wandb_analysis/reward_analysis_summary.json` for summary
   - Open `wandb_analysis/orchestrator_rewards.csv` for reward data
   - View individual run data in `wandb_analysis/{run_id}/`

## Analyzing Reward Components

The analysis script automatically:
- Identifies all reward-related metrics
- Calculates statistics (mean, median, std, min, max)
- Tracks trends (first half vs second half of training)
- Exports reward data to CSV for further analysis

## Using Local Data

If you've already exported data, you can analyze it without API calls:

```powershell
python scripts/analysis/analyze_wandb_rewards.py --local-only
```

This is faster and doesn't require W&B API access.

## Troubleshooting

**"WANDB_API_KEY not found"**
- Add `WANDB_API_KEY=your_key` to `.env` file
- Or use `--local-only` if you have local data

**"No data available"**
- Check that run IDs are correct
- Ensure local data exists in `wandb_analysis/` if using `--local-only`
- Verify W&B API key is valid if fetching from API

**"Empty history data"**
- Some runs may have limited metrics logged
- Check the run in W&B web UI to see what's available
- Try increasing `--max-samples` in export script

