"""
Launch script for Dakota RL training with reward ledger logging.

This script sets up training with:
- Clean Qwen baseline model (no checkpoint resume)
- Wandb logging enabled
- Reward ledger logging integrated
"""

#!/usr/bin/env python3

"""
Example usage:

# Basic training with ledger logging
python dakota_rl_training/launch_with_ledger.py \
    --model Qwen/Qwen3-0.6B \
    --max-steps 1000 \
    --wandb-project dakota-rl-grammar \
    --wandb-name dakota-0.6b-ledger-run

# Or use config files
python dakota_rl_training/launch_with_ledger.py \
    --trainer-config configs/train_30b.toml \
    --orchestrator-config configs/orch_30b.toml \
    --inference-config configs/infer_30b.toml \
    --wandb-project dakota-rl-grammar
"""

import argparse
import sys
from pathlib import Path

# Add dakota_rl_training to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dakota_rl_training.train import check_prerequisites
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(
        description="Launch Dakota RL training with reward ledger logging"
    )
    
    # Config file options (for uv rl command)
    parser.add_argument(
        "--trainer-config",
        type=str,
        default=None,
        help="Path to trainer TOML config file"
    )
    parser.add_argument(
        "--orchestrator-config",
        type=str,
        default=None,
        help="Path to orchestrator TOML config file"
    )
    parser.add_argument(
        "--inference-config",
        type=str,
        default=None,
        help="Path to inference TOML config file"
    )
    
    # Direct config options (alternative to config files)
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name (default: Qwen/Qwen3-0.6B - small instruct model for RL)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dakota_rl_training/outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to RL task dataset JSONL (default: uses packaged dataset)"
    )
    
    # Wandb options
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="dakota-rl-grammar",
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Wandb run name (default: auto-generated)"
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run wandb in offline mode"
    )
    
    # GPU options
    parser.add_argument(
        "--trainer-gpu-ids",
        type=str,
        default="0",
        help="Comma-separated GPU IDs for trainer (e.g., '0' or '0,1,2,3')"
    )
    parser.add_argument(
        "--inference-gpu-ids",
        type=str,
        default="0",
        help="Comma-separated GPU IDs for inference (e.g., '0' or '0,1,2,3')"
    )
    
    # Other options
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean output directory before starting"
    )
    
    args = parser.parse_args()
    load_dotenv()
    
    print("\n" + "="*80)
    print(" DAKOTA RL TRAINING WITH REWARD LEDGER LOGGING")
    print("="*80)
    
    # Check prerequisites
    ready, issues = check_prerequisites()
    if not ready:
        print("\nPrerequisites check failed:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    
    print("\nAll prerequisites met!")
    
    # Build uv rl command
    cmd_parts = ["uv", "run", "rl"]
    
    # Add config files if provided
    if args.trainer_config:
        cmd_parts.extend(["--trainer", "@", args.trainer_config])
    if args.orchestrator_config:
        cmd_parts.extend(["--orchestrator", "@", args.orchestrator_config])
    if args.inference_config:
        cmd_parts.extend(["--inference", "@", args.inference_config])
    
    # Add wandb flags
    cmd_parts.extend(["--wandb.project", args.wandb_project])
    if args.wandb_name:
        cmd_parts.extend(["--wandb.name", args.wandb_name])
    if args.wandb_offline:
        cmd_parts.extend(["--wandb.offline", "true"])
    
    # Add GPU IDs
    cmd_parts.extend(["--trainer-gpu-ids", args.trainer_gpu_ids])
    cmd_parts.extend(["--inference-gpu-ids", args.inference_gpu_ids])
    
    # Add output dir
    cmd_parts.extend(["--output-dir", args.output_dir])
    
    # Add clean flag if requested
    if args.clean:
        cmd_parts.append("--clean")
    
    # If no config files provided, we need to create them or use defaults
    # For now, print the command and instructions
    print("\n" + "="*80)
    print(" TRAINING COMMAND")
    print("="*80)
    print("\nRun this command:")
    print()
    print(" ".join(cmd_parts))
    print()
    print("="*80)
    print(" NOTES")
    print("="*80)
    print("""
1. Reward ledger logging is automatically enabled when using the DakotaGrammarRubric
2. Ledger data will be logged to:
   - W&B: under 'ledger/*' namespace
   - CSV: wandb_analysis/reward_ledger.csv
3. After training, generate visualizations:
   python scripts/analysis/plot_reward_ledger.py
   python scripts/analysis/make_ledger_snippet.py
4. The ledger exposes all reward components for transparency
5. Starting from clean baseline: {model_name}
    """.format(model_name=args.model))
    
    print("\nTo actually launch training, you need to:")
    print("1. Ensure config files exist (or create them)")
    print("2. Run the command above")
    print("3. Or use the Python API directly (see train.py)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

