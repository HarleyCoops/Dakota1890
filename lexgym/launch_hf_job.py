"""Launch a lexgym experiment as a detached Hugging Face Job (CPU only; Tinker does the GPU work).

    python -m lexgym.launch_hf_job --model-name Qwen/Qwen3.5-4B --commit <git sha>

Reads TINKER_API_KEY, WANDB_API_KEY and HF_TOKEN from the environment and passes
them to the job as secrets. The job installs the pinned Tinker packages, checks
out the given commit of this repository, and runs `lexgym.recall_curve`.
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi

REPO_URL = "https://github.com/HarleyCoops/Dakota1890"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--commit", required=True, help="git commit of this repository to run")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--experiment", default="recall_curve_2026_11_10")
    ap.add_argument("--tasks-repo", default="HarleyCooper/riggs1890-lexicon-graph")
    ap.add_argument("--upload-repo", default="HarleyCooper/riggs1890-lexgym-traces")
    ap.add_argument("--flavor", default="cpu-basic")
    ap.add_argument("--timeout", default="10h")
    a = ap.parse_args()
    secrets = {k: os.environ[k] for k in ("TINKER_API_KEY", "WANDB_API_KEY", "HF_TOKEN")}
    script = " && ".join([
        "set -e",
        "pip install -q --index-url https://download.pytorch.org/whl/cpu torch",
        "pip install -q tinker==0.39.0 tinker-cookbook==0.6.1 wandb python-dotenv huggingface_hub datasets",
        "mkdir repo && cd repo && git init -q",
        f"GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 {REPO_URL} {a.commit}",
        "GIT_LFS_SKIP_SMUDGE=1 git checkout -q FETCH_HEAD",
        f"python -m lexgym.recall_curve --model-name {a.model_name} --epochs {a.epochs} "
        f"--experiment {a.experiment} --tasks-repo {a.tasks_repo} --upload-repo {a.upload_repo}",
    ])
    job = HfApi().run_job(image="python:3.12", command=["bash", "-c", script], secrets=secrets,
                          flavor=a.flavor, timeout=a.timeout)
    print(job.id, job.url)


if __name__ == "__main__":
    main()
