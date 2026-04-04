# Wandb Run Data Analysis - Learning Summary

## Overview

This document summarizes what data we can extract from wandb runs after training completes. This is a learning exercise to understand what insights are available from wandb tracking.

## Run Analyzed

- **Run ID**: `7nikv4vp`
- **Run Name**: `dakota-0.6b-rl-trainer`
- **Project**: `dakota-rl-grammar`
- **Entity**: `christian-cooper-us`
- **State**: Finished 
- **URL**: https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/7nikv4vp

## Types of Data Available

### 1. Summary Metrics (Final Values)
These are aggregated metrics that represent the final state or overall statistics:

- **Training Metrics**: Loss (mean, std, min, max), entropy, probabilities
- **Performance Metrics**: Throughput (tokens/sec), MFU (Model FLOPS Utilization), peak memory usage
- **Optimization Metrics**: Learning rate, gradient norms
- **KL Divergence**: Mismatch KL, masked mismatch KL (measures distribution differences)
- **Masking Statistics**: Information about which tokens were masked during training

**Total**: 73 summary metrics tracked

### 2. Configuration (Hyperparameters)
All the training configuration settings:

- Model configuration (name, parallelism settings, attention mechanism)
- Training settings (max steps, batch size, learning rate)
- Loss function configuration
- Checkpoint settings
- Wandb logging configuration

**Total**: 17 config parameters

### 3. History (Time Series Data)
Step-by-step metrics logged during training:

- **1,000 logged steps** are available from the public trainer run when refreshed via `scan_history()`
- Metrics tracked include:
  - Loss metrics (mean, std, min, max at each step)
  - Entropy metrics
  - Performance metrics (throughput, MFU, memory)
  - Probability distributions
  - KL divergence measures
  - Optimization metrics (gradient norms, learning rate)

Earlier local exports in this repo were truncated. The maintained refresh path is now `python scripts/analysis/refresh_tracking_assets.py`.

### 4. Files & Artifacts
Files saved with the run:

- `config.yaml` - Training configuration
- `output.log` - Training logs
- `requirements.txt` - Python dependencies
- `wandb-summary.json` - Summary metrics
- `wandb-metadata.json` - Run metadata
- Artifacts (if any were logged)

## Key Insights from This Run

### Training Success
-  Training completed successfully (finished state)
-  Training duration: **1.54 hours** (5,537 seconds)
-  Steps completed: **998 steps** (out of 1000 max)

### Training Progress
-  **Loss decreased** from 0.000009 to -0.000068 (negative loss indicates policy improvement in RL)
-  **Entropy decreased** from 0.93 to 0.21 - model became more confident
-  **Inference probabilities increased** from 0.63 to 0.86 - model is more certain

### Performance
-  **Average throughput**: 8,178 tokens/sec
-  **Average MFU**: 2.68% (Model FLOPS Utilization - indicates GPU efficiency)
-  **Peak memory**: 11.5 GiB

### Model Behavior
- **Low final entropy** (0.21) indicates the model is confident in its predictions
- **Increasing KL divergence** suggests the model is learning new behaviors (policy is changing)
- **Stable learning rate** (1e-6) throughout training

## What We Can Do With This Data

### 1. **Post-Training Analysis**
- Compare different runs to see which hyperparameters work best
- Identify training issues (crashes, failures, convergence problems)
- Track model performance over time

### 2. **Visualization**
- Plot training curves (loss, entropy, rewards)
- Compare multiple runs side-by-side
- Identify trends and anomalies

### 3. **Debugging**
- Check if training converged properly
- Identify if there were memory issues
- See if gradients were stable

### 4. **Reporting**
- Generate reports for model cards
- Document training statistics
- Share results with team

## Scripts Created

### 1. `scripts/analyze_wandb_run.py`
Basic analysis script that:
- Fetches recent runs from wandb
- Extracts summary, config, and history
- Exports data to JSON/CSV files
- Shows what data is available

**Usage**:
```bash
python scripts/analyze_wandb_run.py
```

### 2. `scripts/visualize_wandb_run.py`
Advanced analysis with visualizations:
- Analyzes training metrics trends
- Generates insights automatically
- Creates plots (loss curves, performance metrics)
- Exports detailed analysis

**Usage**:
```bash
python scripts/visualize_wandb_run.py --run-id 7nikv4vp
```

## Data Export Locations

All exported data is saved to:
- `wandb_analysis/{run_id}/` - Main analysis directory
  - `{run_id}_summary.json` - Final metrics
  - `{run_id}_config.json` - Hyperparameters
  - `{run_id}_history.csv` - Time series data
  - `full_analysis.json` - Complete analysis
  - `detailed_analysis.json` - Detailed insights
  - `plots/` - Visualization images

## Key Learnings

1. **Wandb tracks everything**: From hyperparameters to step-by-step metrics
2. **Rich metadata**: Can see training duration, state, URLs, tags
3. **Time series data**: Full history of training allows for detailed analysis
4. **Easy comparison**: Can compare multiple runs programmatically
5. **Exportable**: All data can be exported for offline analysis

## Next Steps

To analyze other runs:
1. Find the run ID from wandb dashboard or use `analyze_wandb_run.py` to list recent runs
2. Run the visualization script with the run ID
3. Check the exported data files for detailed analysis
4. Compare multiple runs by analyzing their exported data

## Example: Comparing Runs

You can modify the scripts to:
- Compare loss curves between runs
- Find best performing hyperparameters
- Identify common failure patterns
- Track improvements over time



