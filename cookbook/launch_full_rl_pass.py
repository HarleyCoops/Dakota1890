"""Run the full RL pass of lexgym as a chain of Hugging Face Jobs.

    python cookbook/launch_full_rl_pass.py --model-name Qwen/Qwen3.5-4B \
        --initial-checkpoint tinker://536d3512-c611-5499-9dda-869a1d74ad09:train:0/weights/final

One full pass is 2,154 steps at batch 32 over the 68,911 RL tasks. At the measured
smoke rate of about $0.027 per step and 5-6 minutes per step, a pass costs roughly
$60 on Tinker and runs about seven days, which no single Job timeout holds. So this
launcher walks the pass in chunks: each chunk trains N steps from the previous
chunk's saved Tinker state, uploads its run folder to the traces dataset, and is
followed by the next chunk starting from the checkpoint saved at the end of it.

The chain survives interruptions. Progress (the last Tinker state path and the
steps done) is kept in a local JSON beside this script. If a chunk dies or is
cancelled early, the launcher harvests the last saved checkpoint from the job log
or from the uploaded run folder, and the next chunk resumes from there. Re-running
the same command continues the pass where it stopped; pass --reset to start over.

Configuration matches the 13 November smoke: renderer without a thinking block,
answers up to 800 tokens, batch 32, group 8, LoRA rank 32, learning rate 2e-5,
evaluation and checkpoint save every 100 steps.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    from huggingface_hub import HfApi
except ImportError:
    raise SystemExit("pip install huggingface_hub python-dotenv")

from launch_rl_job import REPO_URL, slug

# main after the 800-token change and the launch_full_rl_pass.py commit
DEFAULT_COMMIT = "d5654f7f4d048c0bad0fab895cde5be4322128fd"

TERMINAL = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}


def build_script(args, chunk_idx: int, steps: int, load_path: str | None) -> str:
    log_path = f"{args.log_path_prefix}-chunk{chunk_idx:02d}"
    prefix = f"{args.experiment}/{slug(args.model_name)}/{args.stage}/chunk{chunk_idx:02d}"
    train = [
        "python", "-m", "lexgym.train_rl",
        "--model-name", args.model_name,
        "--task-dir", "data/lexgym_tasks",
        "--log-path", log_path,
        "--batch-size", str(args.batch_size),
        "--group-size", str(args.group_size),
        "--learning-rate", str(args.learning_rate),
        "--eval-every", str(args.eval_every),
        "--save-every", str(args.save_every),
        "--eval-limit-per-task", str(args.eval_limit_per_task),
        "--max-tokens", str(args.max_tokens),
        "--stage", f"{args.stage}-chunk{chunk_idx:02d}",
        "--wandb-name", f"{args.experiment}-{args.stage}-chunk{chunk_idx:02d}",
        "--max-steps", str(steps),
    ]
    if load_path:
        train += ["--load-checkpoint-path", load_path]

    fetch = (f"from lexgym.recall_curve import fetch_tasks; "
             f"fetch_tasks({args.tasks_repo!r}, 'data/lexgym_tasks')")
    upload = (f"from pathlib import Path; from lexgym.recall_curve import upload; "
              f"upload({args.upload_repo!r}, Path({log_path!r}), {prefix!r})")
    return "\n".join([
        "set -e",
        "pip install -q --index-url https://download.pytorch.org/whl/cpu torch",
        f"pip install -q tinker=={args.tinker_version} tinker-cookbook=={args.cookbook_version} "
        "wandb python-dotenv huggingface_hub datasets",
        "mkdir repo && cd repo && git init -q",
        f"GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 {REPO_URL} {args.commit}",
        "GIT_LFS_SKIP_SMUDGE=1 git checkout -q FETCH_HEAD",
        f"python -c {shlex.quote(fetch)}",
        f"trap {shlex.quote('python -c ' + shlex.quote(upload))} EXIT",
        shlex.join(train),
    ])


def wait_terminal(api: HfApi, job_id: str, poll_seconds: int) -> str:
    stage = "SCHEDULING"
    while stage not in TERMINAL:
        time.sleep(poll_seconds)
        try:
            info = api.inspect_job(job_id)
            stage = str(info.status.stage)
            running = getattr(info.durations, "running_secs", None) if info.durations else None
            print(f"  job {job_id}: {stage}" + (f", {int(running) // 60} min running" if running else ""))
        except Exception as exc:  # noqa: BLE001 - transient API errors should not end the chain
            print(f"  status check failed ({exc}); retrying")
    return stage


def harvest_state_paths(api: HfApi, job_id: str, upload_repo: str, prefix: str) -> list[str]:
    """State paths saved during a chunk, newest last, from logs and from the uploaded folder."""
    paths: list[str] = []
    log_text = ""
    try:
        log_text = "\n".join(str(line) for line in api.fetch_job_logs(job_id))
        paths += re.findall(r"tinker://[0-9a-f-]+:train:\d+/weights/[A-Za-z0-9_/.-]+", log_text)
    except Exception as exc:  # noqa: BLE001 - logs may be gone or unavailable
        print(f"  could not read job logs ({exc})")
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(repo_id=upload_repo, repo_type="dataset",
                                filename=f"{prefix}/checkpoints.jsonl")
        for row in (l for l in Path(local).read_text(encoding="utf-8").splitlines() if l.strip()):
            record = json.loads(row)
            if isinstance(record, dict) and record.get("state_path"):
                paths.append(record["state_path"])
    except Exception as exc:  # noqa: BLE001 - the chunk may have failed before any upload
        print(f"  could not read uploaded checkpoints ({exc})")
    return sorted(set(paths))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--initial-checkpoint", required=True)
    ap.add_argument("--experiment", default="rl_full_2026_11_14")
    ap.add_argument("--stage", default="rl-full")
    ap.add_argument("--total-steps", type=int, default=2154, help="Steps in one pass over all RL tasks.")
    ap.add_argument("--steps-per-chunk", type=int, default=350)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--eval-limit-per-task", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--timeout", default="48h")
    ap.add_argument("--tinker-version", default="0.39.0")
    ap.add_argument("--cookbook-version", default="0.6.1")
    ap.add_argument("--tasks-repo", default="HarleyCooper/riggs1890-lexicon-graph")
    ap.add_argument("--upload-repo", default="HarleyCooper/riggs1890-lexgym-traces")
    ap.add_argument("--flavor", default="cpu-basic")
    ap.add_argument("--log-path-prefix", default="lexgym_runs/rl_full_2026_11_14")
    ap.add_argument("--commit", default=DEFAULT_COMMIT)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--progress-file", default="rl_full_pass_progress.json")
    ap.add_argument("--reset", action="store_true", help="Ignore any saved progress and start from the initial checkpoint.")
    a = ap.parse_args()
    load_dotenv()

    progress_path = Path(__file__).resolve().parent.parent / a.progress_file
    if a.reset and progress_path.exists():
        progress_path.unlink()

    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        print(f"resuming: {progress['steps_done']} of {a.total_steps} steps done, "
              f"from {progress['state_path']}")
    else:
        progress = {"steps_done": 0, "state_path": a.initial_checkpoint, "chunks": []}

    secrets = {k: os.environ[k] for k in ("TINKER_API_KEY", "WANDB_API_KEY", "HF_TOKEN")}
    api = HfApi()

    try:
        while progress["steps_done"] < a.total_steps:
            chunk_idx = len(progress["chunks"]) + 1
            steps = min(a.steps_per_chunk, a.total_steps - progress["steps_done"])
            print(f"chunk {chunk_idx}: {steps} steps from {progress['state_path']}")
            job = api.run_job(image="python:3.12",
                              command=["bash", "-c", build_script(a, chunk_idx, steps, progress["state_path"])],
                              secrets=secrets, flavor=a.flavor, timeout=a.timeout)
            chunk = {"index": chunk_idx, "steps": steps, "job_id": job.id, "job_url": job.url}
            print(f"  submitted {job.id} {job.url}")
            stage = wait_terminal(api, job.id, a.poll_seconds)
            chunk["stage"] = stage
            if stage == "COMPLETED":
                progress["steps_done"] += steps
            else:
                print(f"  chunk {chunk_idx} ended in {stage}; rescuing the last saved checkpoint")
            prefix = f"{a.experiment}/{slug(a.model_name)}/{a.stage}/chunk{chunk_idx:02d}"
            paths = harvest_state_paths(api, job.id, a.upload_repo, prefix)
            if paths:
                progress["state_path"] = paths[-1]
                print(f"  last checkpoint: {progress['state_path']}")
            else:
                print("  no checkpoint found; the next chunk starts from the previous one")
            progress["chunks"].append(chunk)
            progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print(f"progress written: {progress_path} "
                  f"({progress['steps_done']} of {a.total_steps} steps)")
    except KeyboardInterrupt:
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        print(f"interrupted; progress written to {progress_path}. Re-run the same command to continue.")

    print(f"pass complete: {progress['steps_done']} steps, final state {progress['state_path']}")
    print("next: evaluate the final state, then export the adapter "
          "(tinker checkpoint push-hf).")


if __name__ == "__main__":
    main()
