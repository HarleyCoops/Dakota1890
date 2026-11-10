"""Experiment: does vocabulary recall rise with epochs, and with model size?

Trains the supervised stage for N epochs on one base model, then evaluates the
base model and the checkpoint at the end of each epoch with greedy decoding.

    python -m lexgym.recall_curve --model-name Qwen/Qwen3.5-4B --epochs 3

Results go to <log-path>/results.json and, with --upload-repo, to a Hugging Face
dataset together with the evaluation traces.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from .evaluate import evaluate
from .tinker_env import pick_renderer_name
from .train_sft import LexgymSFTBuilder

TASK_FILES = ("sft_train.jsonl", "rl_train.jsonl", "eval_heldout_rule.jsonl", "eval_recall.jsonl",
              "lookup_index.json", "report.json")


def fetch_tasks(repo: str, task_dir: str) -> None:
    """Download the task files from the Hugging Face dataset (folder tasks_jsonl/)."""
    from huggingface_hub import hf_hub_download

    Path(task_dir).mkdir(parents=True, exist_ok=True)
    for name in TASK_FILES:
        target = Path(task_dir) / name
        if not target.exists():
            source = hf_hub_download(repo, f"tasks_jsonl/{name}", repo_type="dataset")
            target.write_bytes(Path(source).read_bytes())


def upload(repo: str | None, log_path: Path, prefix: str) -> None:
    if not repo:
        return
    from huggingface_hub import HfApi

    HfApi().upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(log_path), path_in_repo=prefix,
                          ignore_patterns=["wandb/*", "*.html", "*logtree.json"],
                          commit_message=f"recall curve results: {prefix}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--task-dir", default="data/lexgym_tasks")
    ap.add_argument("--tasks-repo", default=None, help="HF dataset with tasks_jsonl/ (used when task files are missing)")
    ap.add_argument("--log-path", default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--limit-per-task", type=int, default=150)
    ap.add_argument("--wandb-project", default="riggs1890-lexgym")
    ap.add_argument("--upload-repo", default=None, help="HF dataset that receives results and traces")
    ap.add_argument("--experiment", default="recall_curve_2026_11_10")
    a = ap.parse_args()
    load_dotenv()
    slug = a.model_name.split("/")[-1].lower()
    log_path = Path(a.log_path or f"lexgym_runs/{a.experiment}/{slug}")
    prefix = f"{a.experiment}/{slug}"
    if a.tasks_repo:
        fetch_tasks(a.tasks_repo, a.task_dir)

    renderer_name = pick_renderer_name(a.model_name)
    common = ChatDatasetBuilderCommonConfig(model_name_for_tokenizer=a.model_name, renderer_name=renderer_name,
                                            max_length=1024, batch_size=a.batch_size,
                                            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
    builder = LexgymSFTBuilder(common_config=common, file_path=f"{a.task_dir}/sft_train.jsonl")
    steps_per_epoch = len(builder()[0])
    sft_dir = log_path / "sft"
    cfg = train.Config(
        log_path=str(sft_dir), model_name=a.model_name, recipe_name="lexgym_riggs1890_recall_curve",
        renderer_name=renderer_name, dataset_builder=builder, learning_rate=a.learning_rate,
        lora_rank=a.lora_rank, num_epochs=a.epochs, save_every=steps_per_epoch, eval_every=50,
        wandb_project=a.wandb_project or None, wandb_name=f"{a.experiment}-sft-{slug}",
    )

    results: dict = {"experiment": a.experiment, "model_name": a.model_name, "epochs": a.epochs,
                     "steps_per_epoch": steps_per_epoch, "learning_rate": a.learning_rate,
                     "lora_rank": a.lora_rank, "batch_size": a.batch_size, "evaluations": []}

    def run_eval(label: str, epoch: int, sampler_path: str | None) -> None:
        summary = asyncio.run(evaluate(a.model_name, sampler_path, a.task_dir, str(log_path / f"eval_{label}"),
                                       run_id=f"{a.experiment}-{slug}-{label}", stage=f"supervised, {label}",
                                       limit_per_task=a.limit_per_task))
        results["evaluations"].append({"label": label, "epoch": epoch, **summary})
        (log_path / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        if a.wandb_project and os.environ.get("WANDB_API_KEY"):
            import wandb

            run = wandb.init(project=a.wandb_project, name=f"{a.experiment}-eval-{slug}", id=f"{a.experiment}-eval-{slug}".replace("_", "-"),
                             resume="allow", config={k: v for k, v in results.items() if k != "evaluations"})
            flat = {f"eval/{key}/{metric}": value for key, stats in summary["results"].items()
                    for metric, value in stats.items() if metric != "n"}
            run.log({"epoch": epoch, **flat})
            run.finish()
        upload(a.upload_repo, log_path, prefix)
        print(f"[{label}] recall exact = {summary['results'].get('split/recall', {}).get('exact')}, "
              f"held-out exact = {summary['results'].get('split/heldout_rule', {}).get('exact')}", flush=True)

    run_eval("epoch0_base", 0, None)
    asyncio.run(train.main(cfg))
    done = set()
    for line in (sft_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines():
        ckpt = json.loads(line)
        step = ckpt.get("epoch", 0) * steps_per_epoch + ckpt.get("batch", 0)
        epoch = step // steps_per_epoch
        if step % steps_per_epoch or epoch == 0 or epoch in done or not ckpt.get("sampler_path"):
            continue
        done.add(epoch)
        run_eval(f"epoch{epoch}", epoch, ckpt["sampler_path"])
    upload(a.upload_repo, log_path, prefix)


if __name__ == "__main__":
    main()
