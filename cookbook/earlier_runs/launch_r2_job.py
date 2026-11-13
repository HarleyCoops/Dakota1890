"""Launch run R2 as one detached Hugging Face Job: tool-safe SFT warm start, then a full RL pass.

    python launch_r2_job.py --model-name Qwen/Qwen3.5-4B

Stages inside the job (CPU only; Tinker does the GPU work):
  1. Build sft_mixed.jsonl with build_mixed_sft.py (single-turn rows plus solved lookup conversations,
     one row per assistant turn).
  2. SFT on that file with the unchanged lexgym.train_sft (loss on the last assistant message of a row).
  3. Evaluate the SFT checkpoint on the 1,500 evaluation tasks.
  4. RL (one pass over rl_train.jsonl) from the SFT checkpoint.
The run folder is uploaded to the traces dataset when the job ends for any reason.

Reads TINKER_API_KEY, WANDB_API_KEY and HF_TOKEN from the environment or a local .env file.
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import shlex
from pathlib import Path

from huggingface_hub import HfApi

REPO_URL = "https://github.com/HarleyCoops/Dakota1890"
DEFAULT_COMMIT = "a281807bedcda2368411aced939f51d516874f7e"  # main on 2026-11-11


def slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model_name.split("/")[-1].lower()).strip("-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--experiment", default="rl_full_2026_11_12")
    ap.add_argument("--stage", default="r2-mixed-sft-then-rl")
    ap.add_argument("--commit", default=DEFAULT_COMMIT)
    ap.add_argument("--sft-epochs", type=int, default=3)
    ap.add_argument("--one-hop", type=int, default=3000)
    ap.add_argument("--rl-max-steps", type=int, default=None, help="Default: one pass over the RL tasks.")
    ap.add_argument("--rl-learning-rate", type=float, default=2e-5)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--eval-limit-per-task", type=int, default=150)
    ap.add_argument("--tinker-version", default="0.39.0")
    ap.add_argument("--cookbook-version", default="0.6.1")
    ap.add_argument("--tasks-repo", default="HarleyCooper/riggs1890-lexicon-graph")
    ap.add_argument("--upload-repo", default="HarleyCooper/riggs1890-lexgym-traces")
    ap.add_argument("--flavor", default="cpu-basic")
    ap.add_argument("--timeout", default="48h")
    a = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    secrets = {k: os.environ[k] for k in ("TINKER_API_KEY", "WANDB_API_KEY", "HF_TOKEN")}
    name = f"{a.experiment}-{a.stage}-{slug(a.model_name)}"
    run = f"lexgym_runs/{name}"
    prefix = f"{a.experiment}/{slug(a.model_name)}/{a.stage}"
    builder = base64.b64encode((Path(__file__).parent.parent / "build_mixed_sft.py").read_bytes()).decode()
    q = shlex.quote

    fetch = f"from lexgym.recall_curve import fetch_tasks; fetch_tasks({a.tasks_repo!r}, 'data/lexgym_tasks')"
    upload = (f"from pathlib import Path; from lexgym.recall_curve import upload; "
              f"upload({a.upload_repo!r}, Path({run!r}), {prefix!r})")
    last = ("import json,sys; rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]; "
            "print(rows[-1][sys.argv[2]])")
    sft = ["python", "-m", "lexgym.train_sft", "--model-name", a.model_name,
           "--sft-file", "data/lexgym_tasks/sft_mixed.jsonl", "--log-path", f"{run}/sft",
           "--num-epochs", str(a.sft_epochs), "--save-every", "200", "--wandb-name", f"{name}-sft"]
    rl = ["python", "-m", "lexgym.train_rl", "--model-name", a.model_name, "--task-dir", "data/lexgym_tasks",
          "--log-path", f"{run}/rl", "--learning-rate", str(a.rl_learning_rate),
          "--eval-every", str(a.eval_every), "--save-every", str(a.save_every),
          "--eval-limit-per-task", str(a.eval_limit_per_task), "--stage", "rl-after-mixed-sft",
          "--wandb-name", f"{name}-rl"]
    if a.rl_max_steps is not None:
        rl += ["--max-steps", str(a.rl_max_steps)]

    script = "\n".join([
        "set -e",
        "pip install -q --index-url https://download.pytorch.org/whl/cpu torch",
        f"pip install -q tinker=={a.tinker_version} tinker-cookbook=={a.cookbook_version} "
        "wandb python-dotenv huggingface_hub datasets",
        "mkdir repo && cd repo && git init -q",
        f"GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 {REPO_URL} {a.commit}",
        "GIT_LFS_SKIP_SMUDGE=1 git checkout -q FETCH_HEAD",
        f"python -c {q(fetch)}",
        f"mkdir -p {run}",
        f"trap {q('python -c ' + q(upload))} EXIT",
        "echo '== stage 1: build the mixed SFT file =='",
        f"echo {builder} | base64 -d > build_mixed_sft.py",
        f"python build_mixed_sft.py --one-hop {a.one_hop}",
        f"cp build_mixed_sft.py data/lexgym_tasks/sft_mixed.report.json {run}/",
        "echo '== stage 2: SFT on the mixed file =='",
        shlex.join(sft),
        f"STATE=$(python -c {q(last)} {run}/sft/checkpoints.jsonl state_path)",
        f"SAMPLER=$(python -c {q(last)} {run}/sft/checkpoints.jsonl sampler_path)",
        'echo "SFT state: $STATE"',
        "echo '== stage 3: evaluate the SFT checkpoint =='",
        f"python -m lexgym.evaluate --model-name {q(a.model_name)} --sampler-path \"$SAMPLER\" "
        f"--out {run}/eval_after_sft --stage sft-mixed",
        f"cat {run}/eval_after_sft/summary.json || true",
        "echo '== stage 4: RL from the SFT checkpoint =='",
        shlex.join(rl) + ' --load-checkpoint-path "$STATE"',
    ])
    job = HfApi().run_job(image="python:3.12", command=["bash", "-c", script], secrets=secrets,
                          flavor=a.flavor, timeout=a.timeout)
    print(job.id, job.url)
    print("W&B run names:", f"{name}-sft", "and", f"{name}-rl")
    print("Traces prefix:", f"{a.upload_repo}/{prefix}")


if __name__ == "__main__":
    main()
