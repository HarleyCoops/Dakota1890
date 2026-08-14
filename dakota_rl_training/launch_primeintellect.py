#!/usr/bin/env python3
"""
Launch Dakota RL Training on PrimeIntellect
Uses PI_API_KEY from .env for authentication
"""

import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

class PrimeIntellectTrainer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.primeintellect.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def load_config(self, config_path: str) -> dict:
        """Load training configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def load_dataset(self, dataset_path: str) -> list:
        """Load JSONL dataset"""
        tasks = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                tasks.append(json.loads(line))
        return tasks

    def create_training_job(self, config: dict, dataset_path: str):
        """Create a new training job on PrimeIntellect"""

        # Load dataset
        dataset = self.load_dataset(dataset_path)

        # Prepare training request
        training_request = {
            "job_name": "dakota_grammar_rl_training",
            "model": {
                "base_model": config['model']['base'],
                "lora_config": {
                    "rank": config['model']['lora_rank'],
                    "alpha": config['model']['lora_alpha'],
                    "dropout": config['model']['lora_dropout'],
                    "target_modules": config['model']['target_modules']
                }
            },
            "training": {
                "algorithm": config['training']['algorithm'],
                "num_epochs": config['training']['num_epochs'],
                "batch_size": config['training']['batch_size'],
                "learning_rate": config['training']['learning_rate'],
                "warmup_steps": config['training']['warmup_steps']
            },
            "dataset": {
                "tasks": dataset,
                "train_split": config['dataset']['train_split'],
                "val_split": config['dataset']['val_split']
            },
            "curriculum": config.get('curriculum', {}),
            "verifier": {
                "use_toploc": config['verifier']['use_toploc'],
                "checkpoint_frequency": config['verifier']['checkpoint_frequency']
            },
            "logging": {
                "wandb_project": config['logging']['wandb_project'],
                "metrics": config['logging']['metrics']
            }
        }

        # Submit training job
        print("=" * 70)
        print("DAKOTA RL TRAINING - PRIMEINTELLECT LAUNCH")
        print("=" * 70)
        print(f"\nModel: {config['model']['base']}")
        print(f"Algorithm: {config['training']['algorithm']}")
        print(f"Dataset: {len(dataset)} tasks")
        print(f"Epochs: {config['training']['num_epochs']}")
        print(f"Curriculum: {config['curriculum']['enabled']}")
        print(f"TOPLOC Verification: {config['verifier']['use_toploc']}")

        print("\n[INFO] Submitting training job to PrimeIntellect...")

        try:
            response = requests.post(
                f"{self.base_url}/training/jobs",
                headers=self.headers,
                json=training_request,
                timeout=30
            )

            if response.status_code == 200 or response.status_code == 201:
                result = response.json()
                job_id = result.get('job_id')

                print("\n[SUCCESS] Training job created!")
                print(f"Job ID: {job_id}")
                print(f"Status: {result.get('status', 'pending')}")

                if 'dashboard_url' in result:
                    print(f"\nMonitor training at: {result['dashboard_url']}")

                return job_id
            else:
                print("\n[ERROR] Failed to create training job")
                print(f"Status: {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"\n[ERROR] Network error: {e}")
            print("\nAlternative launch methods:")
            print("1. Use PrimeIntellect web UI: https://app.primeintellect.ai")
            print("2. Use prime-rl CLI: prime-rl train --config configs/training_config.yaml")
            return None

    def monitor_job(self, job_id: str):
        """Monitor training job status"""
        try:
            response = requests.get(
                f"{self.base_url}/training/jobs/{job_id}",
                headers=self.headers,
                timeout=10
            )

            if response.status_code == 200:
                status = response.json()
                print(f"\nJob Status: {status.get('status')}")
                print(f"Progress: {status.get('progress', 0):.1f}%")

                if 'metrics' in status:
                    metrics = status['metrics']
                    print("\nCurrent Metrics:")
                    print(f"  Reward: {metrics.get('reward/mean', 0):.3f}")
                    print(f"  Char Accuracy: {metrics.get('char_accuracy', 0):.2%}")
                    print(f"  Affix Accuracy: {metrics.get('affix_accuracy', 0):.2%}")

                return status
            else:
                print(f"[ERROR] Could not fetch job status: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Network error: {e}")
            return None

def main():
    # Check for API key
    api_key = os.getenv("PI_API_KEY")
    if not api_key:
        print("[ERROR] PI_API_KEY not found in .env file")
        print("\nPlease add your PrimeIntellect API key to .env:")
        print("PI_API_KEY=your_key_here")
        return

    # Initialize trainer
    trainer = PrimeIntellectTrainer(api_key)

    # Load configuration
    config_path = Path(__file__).parent / "configs" / "training_config.yaml"
    config = trainer.load_config(str(config_path))

    # Determine dataset path
    if config['curriculum']['enabled']:
        # Start with easy tasks
        dataset_path = Path(__file__).parent / config['curriculum']['stages'][0]['dataset']
        print(f"\n[INFO] Using curriculum learning - starting with: {dataset_path.name}")
    else:
        # Use complete dataset
        dataset_path = Path(__file__).parent / "datasets" / "grammar_tasks_complete_v2.jsonl"

    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return

    # Create training job
    job_id = trainer.create_training_job(config, str(dataset_path))

    if job_id:
        print("\n" + "=" * 70)
        print("TRAINING LAUNCHED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nJob ID: {job_id}")
        print("\nNext steps:")
        print("1. Monitor training at: https://app.primeintellect.ai")
        print("2. Check metrics in Weights & Biases")
        print("3. Wait for curriculum stages to complete")
        print("\nExpected timeline:")
        print("  - Easy tasks: ~2-4 hours")
        print("  - Medium tasks: ~3-5 hours")
        print("  - Hard tasks: ~1-2 hours")
        print("  - Total: 6-11 hours")
    else:
        print("\n[INFO] Could not launch via API. Please use web UI:")
        print("https://app.primeintellect.ai")
        print("\nUpload these files:")
        print(f"  1. {config_path}")
        print(f"  2. {dataset_path}")

if __name__ == "__main__":
    main()
