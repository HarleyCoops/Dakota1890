"""Launch the SFT learning-curve run as one detached Hugging Face Job.

    python launch_sft_curve_job.py --model-name thinkingmachines/Inkling-2-Small

Stages inside the job (CPU only; Tinker does the GPU work):
  1. Build the mixed SFT file with build_mixed_sft.py (single-turn rows plus solved lookup
     conversations) and use it as sft_train.jsonl.
  2. Run lexgym.recall_curve: evaluate the base model on the 1,500 evaluation tasks, train
     SFT for N epochs, and evaluate the checkpoint at the end of each epoch.
Results go to results.json, to the W&B project, and to the traces dataset after each evaluation.

Thinking is off. For Qwen the repo already picks the renderer without a thinking block. For
Inkling the job sets the renderer's default reasoning effort to 0.0 for training and sampling.

Reads TINKER_API_KEY, WANDB_API_KEY and HF_TOKEN from the environment or a local .env file.
"""
from __future__ import annotations

import argparse
import base64
import os
import shlex
from pathlib import Path

from huggingface_hub import HfApi

REPO_URL = "https://github.com/HarleyCoops/Dakota1890"
DEFAULT_COMMIT = "a281807bedcda2368411aced939f51d516874f7e"  # main on 2026-11-11

# Sets the Inkling renderer's default reasoning effort to 0.0 and checks the edit.
EFFORT_ZERO = [
    'TML=$(python -c "import importlib.util as u; print(u.find_spec(\'tinker_cookbook.renderers.tml_v0\').origin)")',
    "sed -i 's/^DEFAULT_EFFORT: float = 0.9$/DEFAULT_EFFORT: float = 0.0/' \"$TML\"",
    "grep -q '^DEFAULT_EFFORT: float = 0.0$' \"$TML\"",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--experiment", default="sft_mixed_thinking_off_2026_11_12")
    ap.add_argument("--commit", default=DEFAULT_COMMIT)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--one-hop", type=int, default=3000)
    ap.add_argument("--limit-per-task", type=int, default=150)
    ap.add_argument("--eval-max-tokens", type=int, default=2048)
    ap.add_argument("--tinker-version", default="0.39.0")
    ap.add_argument("--cookbook-version", default="0.6.1")
    ap.add_argument("--tasks-repo", default="HarleyCooper/riggs1890-lexicon-graph")
    ap.add_argument("--upload-repo", default="HarleyCooper/riggs1890-lexgym-traces")
    ap.add_argument("--flavor", default="cpu-basic")
    ap.add_argument("--timeout", default="24h")
    a = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    secrets = {k: os.environ[k] for k in ("TINKER_API_KEY", "WANDB_API_KEY", "HF_TOKEN")}
    slug = a.model_name.split("/")[-1].lower()
    run = f"lexgym_runs/{a.experiment}/{slug}"
    prefix = f"{a.experiment}/{slug}"
    builder = base64.b64encode((Path(__file__).parent / "build_mixed_sft.py").read_bytes()).decode()
    q = shlex.quote

    fetch = f"from lexgym.recall_curve import fetch_tasks; fetch_tasks({a.tasks_repo!r}, 'data/lexgym_tasks')"
    upload = (f"from pathlib import Path; from lexgym.recall_curve import upload; "
              f"upload({a.upload_repo!r}, Path({run!r}), {prefix!r})")
    curve = ["python", "-m", "lexgym.recall_curve", "--model-name", a.model_name, "--epochs", str(a.epochs),
             "--learning-rate", str(a.learning_rate), "--limit-per-task", str(a.limit_per_task),
             "--experiment", a.experiment, "--upload-repo", a.upload_repo]

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
        "mv data/lexgym_tasks/sft_train.jsonl data/lexgym_tasks/sft_train_single_turn.jsonl",
        "mv data/lexgym_tasks/sft_mixed.jsonl data/lexgym_tasks/sft_train.jsonl",
        f"cp build_mixed_sft.py data/lexgym_tasks/sft_mixed.report.json {run}/",
        "echo '== thinking off, and a larger answer limit for evaluation =='",
        *EFFORT_ZERO,
        f"sed -i 's/max_tokens: int = 96, concurrency/max_tokens: int = {a.eval_max_tokens}, concurrency/' lexgym/evaluate.py",
        f"grep -q 'max_tokens: int = {a.eval_max_tokens}, concurrency' lexgym/evaluate.py",
        "echo '== stage 2: base evaluation, SFT, and one evaluation per epoch =='",
        shlex.join(curve),
    ])
    job = HfApi().run_job(image="python:3.12", command=["bash", "-c", script], secrets=secrets,
                          flavor=a.flavor, timeout=a.timeout)
    print(job.id, job.url)
    print("W&B run names:", f"{a.experiment}-sft-{slug}", "and", f"{a.experiment}-eval-{slug}")
    print("Traces prefix:", f"{a.upload_repo}/{prefix}")


if __name__ == "__main__":
    main()
