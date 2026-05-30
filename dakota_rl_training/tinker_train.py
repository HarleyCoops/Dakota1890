#!/usr/bin/env python3
"""Launch Dakota RL training on Tinker."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
import json
import site

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Preserve original sys.path and force-load pip verifiers before adding local paths
_orig_sys_path = list(sys.path)
_site_paths = [site.getusersitepackages(), *site.getsitepackages()]
_clean_path = [p for p in _orig_sys_path if "dakota_rl_training" not in p]
sys.path = [p for p in _site_paths if p] + _clean_path
import verifiers  # type: ignore  # noqa: F401
# Restore sys.path with project paths added
sys.path = _orig_sys_path

# Add project root and environments directory to sys.path to allow imports
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT_DIR / "environments"
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(ENV_DIR) not in sys.path:
    sys.path.append(str(ENV_DIR))

# Add dakota_rl_training to path
RL_TRAINING_DIR = ROOT_DIR / "dakota_rl_training"
if str(RL_TRAINING_DIR) not in sys.path:
    sys.path.append(str(RL_TRAINING_DIR))

from dotenv import load_dotenv
import weave
from tinker_cookbook.rl import train

from dakota_rl_training.tinker_integration import (
    DakotaGrammarDatasetBuilder,
    export_reward_ledger,
)

logger = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Dakota grammar RL policy on Tinker.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="Base model to fine-tune.")
    parser.add_argument("--log-path", default="dakota_rl_training/outputs/tinker_run", help="Directory for logs/checkpoints.")
    parser.add_argument("--dataset-path", default=None, help="Optional override JSONL dataset.")
    parser.add_argument("--eval-path", default=None, help="Optional eval split JSONL.")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of env groups per batch.")
    parser.add_argument("--group-size", type=int, default=16, help="Rollouts per GRPO group.")
    parser.add_argument("--max-examples", type=int, default=-1, help="Limit number of training examples.")
    parser.add_argument("--eval-examples", type=int, default=-1, help="Limit number of eval examples.")
    parser.add_argument("--eval-fraction", type=float, default=0.1, help="Eval split when eval_path not provided.")
    parser.add_argument("--system-prompt", default=None, help="Override default Dakota system prompt.")
    parser.add_argument("--difficulty-filter", nargs="*", default=None, help="Filter dataset difficulties (case-insensitive).")
    parser.add_argument("--task-filter", nargs="*", default=None, help="Filter dataset task types.")
    parser.add_argument("--seed", type=int, default=42, help="Dataset shuffling seed.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max sampled tokens per completion.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--learning-rate", type=float, default=4e-5, help="Optimizer learning rate.")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank for Tinker training client.")
    parser.add_argument("--loss-fn", choices=["importance_sampling", "ppo"], default="importance_sampling", help="RL loss function.")
    parser.add_argument("--num-substeps", type=int, default=1, help="Optimizer steps per training iteration.")
    parser.add_argument("--kl-penalty-coef", type=float, default=0.0, help="KL penalty coefficient.")
    parser.add_argument("--kl-discount-factor", type=float, default=0.0, help="KL penalty discount.")
    parser.add_argument("--eval-every", type=int, default=20, help="How often to run eval batches.")
    parser.add_argument("--save-every", type=int, default=20, help="How often to checkpoint weights.")
    parser.add_argument("--num-groups-to-log", type=int, default=4, help="Trajectory groups to pretty-print/logtree.")
    parser.add_argument("--wandb-project", default="thinking-machines-qwen3-30b", help="Weights & Biases project.")
    parser.add_argument("--wandb-name", default=None, help="Weights & Biases run name.")
    parser.add_argument("--base-url", default=None, help="Custom Tinker API base URL.")
    parser.add_argument("--ledger-csv", default="wandb_analysis/reward_ledger_tinker.csv", help="Output CSV for reward ledger.")
    parser.add_argument("--async-groups", type=int, default=0, help="Enable async mode with this many groups per batch.")
    parser.add_argument("--async-max-steps", type=int, default=4, help="Maximum policy lag when async mode enabled.")
    parser.add_argument("--stream-groups-per-batch", type=int, default=0, help="Streaming minibatch total groups.")
    parser.add_argument("--stream-num-minibatches", type=int, default=2, help="Streaming minibatch subdivision.")
    parser.add_argument(
        "--sync-metrics-to-wandb",
        action="store_true",
        help="After training, replay metrics.jsonl to the W&B run (useful when Tinker workers are sandboxed).",
    )
    return parser


def build_dataset_builder(args: argparse.Namespace) -> DakotaGrammarDatasetBuilder:
    return DakotaGrammarDatasetBuilder(
        model_name=args.model_name,
        batch_size=args.batch_size,
        group_size=args.group_size,
        dataset_path=args.dataset_path,
        eval_path=args.eval_path,
        max_examples=args.max_examples,
        eval_examples=args.eval_examples,
        eval_fraction=args.eval_fraction,
        system_prompt=args.system_prompt,
        difficulty_filter=args.difficulty_filter,
        task_filter=args.task_filter,
        seed=args.seed,
    )


def build_config(args: argparse.Namespace) -> train.Config:
    dataset_builder = build_dataset_builder(args)
    async_config = None
    if args.async_groups > 0:
        async_config = train.AsyncConfig(
            max_steps_off_policy=args.async_max_steps,
            groups_per_batch=args.async_groups,
        )
    stream_config = None
    if args.stream_groups_per_batch > 0:
        stream_config = train.StreamMinibatchConfig(
            groups_per_batch=args.stream_groups_per_batch,
            num_minibatches=args.stream_num_minibatches,
        )

    return train.Config(
        learning_rate=args.learning_rate,
        dataset_builder=dataset_builder,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        lora_rank=args.lora_rank,
        kl_penalty_coef=args.kl_penalty_coef,
        kl_discount_factor=args.kl_discount_factor,
        loss_fn=args.loss_fn,
        num_substeps=args.num_substeps,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        log_path=args.log_path,
        base_url=args.base_url,
        eval_every=args.eval_every,
        save_every=args.save_every,
        remove_constant_reward_groups=True,
        async_config=async_config,
        stream_minibatch_config=stream_config,
        num_groups_to_log=args.num_groups_to_log,
    )


@weave.op()
async def run_training(cfg: train.Config) -> None:
    await train.main(cfg)


def _find_wandb_metadata(log_path: Path):
    """Locate the wandb-metadata.json file generated by Tinker."""

    for meta_path in Path(log_path).glob("wandb/run-*/files/wandb-metadata.json"):
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            run_dir = meta_path.parents[1].name
            run_id = data.get("id") or run_dir.rsplit("-", 1)[-1]
            run_name = data.get("name")
            entity = data.get("entity") or data.get("username")
            if run_id:
                return run_id, run_name, entity
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read %s: %s", meta_path, exc)
    return None


def _sync_metrics_to_wandb(metrics_path: Path, project: str, run_id: str, run_name: str | None, entity: str | None):
    """Replay metrics.jsonl into the matching W&B run (resumes if run exists)."""

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; cannot sync metrics.")
        return

    if not metrics_path.exists():
        logger.warning("metrics.jsonl missing; skipping sync.")
        return

    logger.info("Syncing metrics from %s to W&B run %s (project=%s)", metrics_path, run_id, project)
    wandb.init(project=project, id=run_id, resume="allow", name=run_name, entity=entity, settings=wandb.Settings(start_method="thread"))

    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = entry.get("step")
            wandb.log(entry, step=step)

    wandb.finish()
    logger.info("Completed W&B sync for %s", run_id)


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    load_dotenv()

    cfg = build_config(args)
    Path(cfg.log_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Tinker RL training run at %s", cfg.log_path)

    # Initialize Weave
    if cfg.wandb_project:
        weave.init(cfg.wandb_project)

    asyncio.run(run_training(cfg))
    logger.info("Training complete.")

    metrics_path = Path(cfg.log_path) / "metrics.jsonl"
    ledger_csv = Path(args.ledger_csv)
    if metrics_path.exists():
        export_reward_ledger(metrics_path, ledger_csv)
        logger.info("Reward ledger exported to %s", ledger_csv)
    else:
        logger.warning("metrics.jsonl not found; skipping ledger export.")

    if args.sync_metrics_to_wandb and cfg.wandb_project:
        wandb_meta = _find_wandb_metadata(Path(cfg.log_path))
        if wandb_meta is None:
            logger.warning("Could not locate W&B metadata; skipping metrics sync.")
        else:
            run_id, run_name, entity = wandb_meta
            _sync_metrics_to_wandb(metrics_path, cfg.wandb_project, run_id, run_name, entity)
    elif cfg.wandb_project:
        logger.warning(
            "W&B metrics replay is disabled. Local metrics.jsonl may contain ledger channels "
            "that are absent from the public W&B run; pass --sync-metrics-to-wandb for audit runs."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
