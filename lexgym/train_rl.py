"""Stage 2: RL on Tinker over the verifiable lexgym tasks.

    python -m lexgym.train_rl --model-name <tinker model> --load-checkpoint-path tinker://.../weights/final

Start from the SFT checkpoint. From a cold start the model cannot sample a
Dakota word it does not know, so lexical tasks give no gradient.
"""
from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv
from tinker_cookbook.rl import train

from .tinker_env import LexgymDatasetBuilder, pick_renderer_name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--load-checkpoint-path", default=None, help="SFT state path (tinker://...).")
    ap.add_argument("--task-dir", default="data/lexgym_tasks")
    ap.add_argument("--log-path", default="lexgym_runs/rl")
    ap.add_argument("--train-tasks", nargs="*", default=None)
    ap.add_argument("--train-limit-per-task", type=int, default=-1)
    ap.add_argument("--eval-limit-per-task", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--trace-dir", default=None, help="Rollout traces (default: <log-path>/traces). Use 'off' to disable.")
    ap.add_argument("--stage", default="rl", help="Label stored in every trace row, for example rl-after-sft.")
    ap.add_argument("--wandb-project", default="riggs1890-lexgym")
    ap.add_argument("--wandb-name", default=None)
    a = ap.parse_args()
    load_dotenv()
    renderer_name = pick_renderer_name(a.model_name)
    builder = LexgymDatasetBuilder(
        model_name=a.model_name, task_dir=a.task_dir, batch_size=a.batch_size, group_size=a.group_size,
        train_tasks=a.train_tasks, train_limit_per_task=a.train_limit_per_task,
        eval_limit_per_task=a.eval_limit_per_task, renderer_name=renderer_name,
        trace_dir=None if a.trace_dir == "off" else (a.trace_dir or f"{a.log_path}/traces"),
        run_id=a.wandb_name or a.log_path.rstrip("/").split("/")[-1], stage=a.stage,
        load_checkpoint_path=a.load_checkpoint_path, temperature=a.temperature, max_tokens=a.max_tokens)
    cfg = train.Config(
        learning_rate=a.learning_rate, dataset_builder=builder, model_name=a.model_name,
        recipe_name="lexgym_riggs1890_rl", max_tokens=a.max_tokens, temperature=a.temperature,
        lora_rank=a.lora_rank, log_path=a.log_path, load_checkpoint_path=a.load_checkpoint_path,
        renderer_name=renderer_name, eval_every=a.eval_every, save_every=a.save_every,
        wandb_project=a.wandb_project or None, wandb_name=a.wandb_name, max_steps=a.max_steps,
        # False: with True the cookbook drops groups before tagging, and the
        # per-task train metrics are then attributed to the wrong task.
        remove_constant_reward_groups=False,
    )
    asyncio.run(train.main(cfg))


if __name__ == "__main__":
    main()
