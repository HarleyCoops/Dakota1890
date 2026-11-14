"""Stage 2: RL on Tinker over the verifiable lexgym tasks.

    python -m lexgym.train_rl --model-name <tinker model> --load-checkpoint-path tinker://.../weights/final

Start from the SFT checkpoint. From a cold start the model cannot sample a
Dakota word it does not know, so lexical tasks give no gradient.

Reward arms (--rl-arm), each compared against the unchanged baseline loop on
the same base model, seed set, and evaluation sets:

- baseline: the composition the pilots and the 13 November smoke ran.
- a1: gated composition. `char_dice` leaves the en_to_dak reward (1.0 exact,
  0.7 n / ŋ, else 0) and stays in the ledger as a diagnostic. scoring.py
  selects it through task["arm"] = "gated"; SCORER_VERSIONS marks it in the
  traces.
- a2: a1 plus a scaled advantage. The group-centred advantages are divided by
  the group standard deviation with a floor of --sigma-min, replacing the
  cookbook's mean-centred compute_advantages. tinker-cookbook 0.6.1 calls it
  as a module global of tinker_cookbook.rl.train (prepare_minibatch), so the
  rebind below reaches the training loop.
- c1: a1 plus a KL penalty toward the supervised checkpoint, through the
  cookbook's kl_penalty_coef and KLReferenceConfig. Requires
  --load-checkpoint-path: the reference is the state this run started from,
  so the penalty holds the supervised skills while RL learns behaviour.

The measure for every arm is held-out exact match per task type and recall
exact match. The training reward is not a measure, because the arms alter it.
"""
from __future__ import annotations

import argparse
import asyncio

import torch
from dotenv import load_dotenv
from tinker_cookbook.rl import train
from tinker_cookbook.rl.train import KLReferenceConfig

from .advantages import group_advantages
from .tinker_env import LexgymDatasetBuilder, pick_renderer_name

SCORING_ARM = {"baseline": "baseline", "a1": "gated", "a2": "gated", "c1": "gated"}


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
    ap.add_argument("--rl-arm", choices=sorted(SCORING_ARM), default="baseline",
                    help="Which GRPO arm this run is (see the module docstring).")
    ap.add_argument("--sigma-min", type=float, default=0.25,
                    help="a2 only: floor on the group standard deviation when scaling advantages.")
    ap.add_argument("--kl-penalty-coef", type=float, default=0.05,
                    help="c1 only: KL penalty coefficient against the supervised checkpoint.")
    ap.add_argument("--kl-discount-factor", type=float, default=0.0,
                    help="c1 only: position discount for the KL penalty.")
    a = ap.parse_args()
    load_dotenv()

    if a.rl_arm == "c1" and not a.load_checkpoint_path:
        raise SystemExit(
            "--rl-arm c1 penalises KL against the supervised checkpoint, so it needs "
            "--load-checkpoint-path"
        )
    if a.rl_arm == "a2":
        def compute_advantages_scaled(trajectory_groups):
            return [
                torch.tensor(group_advantages(list(group.get_total_rewards()), a.sigma_min))
                for group in trajectory_groups
            ]
        # prepare_minibatch looks this up as a module global, so the rebind reaches it.
        train.compute_advantages = compute_advantages_scaled

    renderer_name = pick_renderer_name(a.model_name)
    builder = LexgymDatasetBuilder(
        model_name=a.model_name, task_dir=a.task_dir, batch_size=a.batch_size, group_size=a.group_size,
        train_tasks=a.train_tasks, train_limit_per_task=a.train_limit_per_task,
        eval_limit_per_task=a.eval_limit_per_task, renderer_name=renderer_name,
        trace_dir=None if a.trace_dir == "off" else (a.trace_dir or f"{a.log_path}/traces"),
        run_id=a.wandb_name or a.log_path.rstrip("/").split("/")[-1], stage=a.stage,
        load_checkpoint_path=a.load_checkpoint_path, temperature=a.temperature, max_tokens=a.max_tokens,
        arm=SCORING_ARM[a.rl_arm])
    cfg_kwargs: dict = dict(
        learning_rate=a.learning_rate, dataset_builder=builder, model_name=a.model_name,
        recipe_name="lexgym_riggs1890_rl", max_tokens=a.max_tokens, temperature=a.temperature,
        lora_rank=a.lora_rank, log_path=a.log_path, load_checkpoint_path=a.load_checkpoint_path,
        renderer_name=renderer_name, eval_every=a.eval_every, save_every=a.save_every,
        wandb_project=a.wandb_project or None, wandb_name=a.wandb_name, max_steps=a.max_steps,
        # False: with True the cookbook drops groups before tagging, and the
        # per-task train metrics are then attributed to the wrong task.
        remove_constant_reward_groups=False,
    )
    if a.rl_arm == "c1":
        cfg_kwargs.update(
            kl_penalty_coef=a.kl_penalty_coef,
            kl_discount_factor=a.kl_discount_factor,
            kl_reference_config=KLReferenceConfig(
                base_model=a.model_name, load_checkpoint_path=a.load_checkpoint_path),
        )
    cfg = train.Config(**cfg_kwargs)
    asyncio.run(train.main(cfg))


if __name__ == "__main__":
    main()
