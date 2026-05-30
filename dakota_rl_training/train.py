#!/usr/bin/env python3
"""
Dakota Grammar RL Training Script
Actual implementation using PrimeIntellect prime-rl framework.

Replaces the stub with real trainer wiring that loads the Dakota environment
and runs GRPO training with curriculum learning.
"""

import argparse
import importlib.util
import logging
import site
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ENVIRONMENT_PACKAGE_DIR = ROOT_DIR / "environments" / "dakota_grammar_translation"

# Prefer the pip-installed `verifiers` package over the legacy local folder that
# shares the same import name inside `dakota_rl_training/`.
_original_sys_path = list(sys.path)
_site_paths = [site.getusersitepackages(), *site.getsitepackages()]
_clean_path = [p for p in _original_sys_path if Path(p or ".").resolve() != SCRIPT_DIR]
sys.path = [p for p in _site_paths if p] + _clean_path
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(ENVIRONMENT_PACKAGE_DIR) not in sys.path:
    sys.path.append(str(ENVIRONMENT_PACKAGE_DIR))

# Try to import prime_rl components
try:
    from prime_rl.rl import RLConfig, rl
    from prime_rl.utils.pydantic_config import parse_argv
    PRIME_RL_AVAILABLE = True
except ImportError:
    PRIME_RL_AVAILABLE = False
    RLConfig = None
    rl = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_prerequisites() -> tuple[bool, list[str]]:
    """Check if all prerequisites are met."""
    issues = []
     
    if not PRIME_RL_AVAILABLE:
        issues.append("prime_rl not installed. Install with: pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git")
    
    if importlib.util.find_spec("verifiers") is None:
        issues.append("verifiers not installed. Install with: pip install git+https://github.com/PrimeIntellect-ai/verifiers.git")
    
    # Check if environment package is installed or available directly from the repo.
    if importlib.util.find_spec("dakota_grammar_translation") is None:
        issues.append(
            "dakota_grammar_translation environment could not be imported from the repo. "
            "Install with: pip install -e environments/dakota_grammar_translation"
        )
    
    # Check if datasets exist
    datasets_dir = Path("dakota_rl_training/datasets")
    if not (datasets_dir / "grammar_tasks_complete.jsonl").exists():
        issues.append(
            f"Dataset not found: {datasets_dir}/grammar_tasks_complete.jsonl. "
            "Run: python convert_rules_to_primeintellect.py"
        )
    
    return len(issues) == 0, issues


def create_rl_config(
    config_path: Optional[Path] = None,
    output_dir: Path = Path("dakota_rl_training/outputs"),
    model_name: str = "Qwen/Qwen3-0.6B",
    max_steps: Optional[int] = None,
    env_dataset_path: Optional[str] = None,
    wandb_project: str = "dakota-rl-grammar",
    wandb_name: Optional[str] = None,
    local: bool = False,
) -> RLConfig:
    """
    Create RLConfig for Dakota grammar training.
    
    Args:
        config_path: Path to TOML config file (if None, uses defaults)
        output_dir: Directory for outputs (checkpoints, logs, etc.)
        model_name: HuggingFace model name
        max_steps: Maximum training steps (None = unlimited)
        env_dataset_path: Path to RL task dataset JSONL
        wandb_project: Weights & Biases project name
        wandb_name: Weights & Biases run name
        local: If True, use single GPU local training
    """
    if not PRIME_RL_AVAILABLE:
        raise ImportError("prime_rl not available. Install with: pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git")
    
    # If config file provided, parse it
    if config_path and config_path.exists():
        logger.info(f"Loading config from {config_path}")
        # Parse command line args as if they were from config file
        sys.argv = ["train.py", "@", str(config_path)]
        config = parse_argv(RLConfig)
        return config
    
    # Otherwise create default config programmatically
    from prime_rl.trainer.rl.config import RLTrainerConfig, LossConfig, AdamWConfig, ConstantSchedulerConfig
    from prime_rl.trainer.config import ModelConfig, CheckpointConfig, WeightCheckpointConfig, LogConfig
    from prime_rl.orchestrator.config import OrchestratorConfig, EnvironmentConfig, BufferConfig, ClientConfig
    from prime_rl.inference.config import InferenceConfig, EngineConfig
    from prime_rl.utils.config import WandbMonitorConfig
    
    # Default dataset path
    if env_dataset_path is None:
        env_dataset_path = str(Path("dakota_rl_training/datasets/grammar_tasks_complete.jsonl").resolve())
    
    # Trainer config
    trainer_config = RLTrainerConfig(
        model=ModelConfig(name=model_name),
        output_dir=output_dir / "trainer",
        max_steps=max_steps,
        async_level=2,
        loss=LossConfig(),
        optim=AdamWConfig(lr=5.0e-6),
        scheduler=ConstantSchedulerConfig(),
        ckpt=CheckpointConfig(interval=50),
        weights=WeightCheckpointConfig(),
        log=LogConfig(),
        wandb=WandbMonitorConfig(project=wandb_project, name=wandb_name) if wandb_project else None,
    )
    
    # Orchestrator config (loads environment)
    orchestrator_config = OrchestratorConfig(
        environment=EnvironmentConfig(
            id="dakota_grammar_translation",
            args={
                "dataset_path": env_dataset_path,
                "max_examples": -1,
                "eval_fraction": 0.1,
                "difficulty_filter": None,  # Can filter: ["easy"], ["medium"], ["hard"]
                "task_filter": None,  # Can filter: ["morphology"], ["translation"], etc.
            }
        ),
        buffer=BufferConfig(),
        model=ModelConfig(name=model_name),
        client=ClientConfig(base_url="http://localhost:8000/v1"),
        output_dir=output_dir / "orchestrator",
        max_steps=max_steps,
        async_level=2,
        seed=42,
        log=LogConfig(),
        wandb=WandbMonitorConfig(project=wandb_project, name=wandb_name) if wandb_project else None,
    )
    
    # Inference config (vLLM server)
    inference_config = InferenceConfig(
        model=ModelConfig(name=model_name),
        engine=EngineConfig(
            tensor_parallel_size=1 if local else 1,  # Adjust for multi-GPU
            max_model_len=8192,
        ),
        server_port=8000,
    )
    
    # Create RL config
    config = RLConfig(
        trainer=trainer_config,
        orchestrator=orchestrator_config,
        inference=inference_config,
        output_dir=output_dir,
        trainer_gpu_ids=[0] if local else [0],  # Adjust for multi-GPU
        inference_gpu_ids=[0] if local else [0],
        log=LogConfig(),
        clean=False,  # Set True to clean output dir before starting
    )
    
    return config


def main():
    """Main entry point for RL training."""
    parser = argparse.ArgumentParser(
        description="Train Dakota grammar model with RL using PrimeIntellect"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config file (if None, uses defaults)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dakota_rl_training/outputs",
        help="Output directory for checkpoints, logs, etc."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (None = unlimited)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to RL task dataset JSONL"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="dakota-rl-grammar",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run local training (single GPU)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check prerequisites, don't train"
    )
    
    args = parser.parse_args()
    load_dotenv()
    
    print("\n" + "="*70)
    print(" DAKOTA GRAMMAR RL TRAINING")
    print("="*70)
    
    # Check prerequisites
    ready, issues = check_prerequisites()
    if not ready:
        print("\n  Prerequisites check failed:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease install missing dependencies and try again.")
        return 1
    
    if args.check_only:
        print("\n All prerequisites met!")
        return 0
    
    print("\n All prerequisites met!")
    
    # Create config
    config_path = Path(args.config) if args.config else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        config = create_rl_config(
            config_path=config_path,
            output_dir=output_dir,
            model_name=args.model,
            max_steps=args.max_steps,
            env_dataset_path=args.dataset,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            local=args.local,
        )
        
        logger.info("Starting RL training...")
        logger.info(f"Model: {args.model}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Dataset: {args.dataset or 'default'}")
        logger.info(f"Max steps: {args.max_steps or 'unlimited'}")
        
        # Run training
        rl(config)
        
        logger.info("Training completed!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
