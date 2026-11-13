"""Launch a full RL pass of lexgym as a detached Hugging Face Job (CPU only; Tinker does the GPU work).

    python launch_rl_job.py --model-name Qwen/Qwen3.5-4B
    python launch_rl_job.py --model-name Qwen/Qwen3.5-4B \
        --load-checkpoint-path tinker://536d3512-c611-5499-9dda-869a1d74ad09:train:0/weights/final \
        --stage rl-after-sft3
    python launch_rl_job.py --model-name thinkingmachines/Inkling-2-Small

Reads TINKER_API_KEY, WANDB_API_KEY and HF_TOKEN from the environment and passes
them to the job as secrets. The job checks out one commit of HarleyCoops/Dakota1890,
downloads the task files, runs `lexgym.train_rl`, and uploads the run folder to the
traces dataset. The upload also runs when training fails, so checkpoint paths are kept.

Without --max-steps the run is one pass over rl_train.jsonl (about 2,150 steps at
batch size 32).
"""
from __future__ import annotations

import argparse
import os
import re
import shlex

from huggingface_hub import HfApi

REPO_URL = "https://github.com/HarleyCoops/Dakota1890"
DEFAULT_COMMIT = "a281807bedcda2368411aced939f51d516874f7e"  # main on 2026-11-11


# The repo's pick_renderer_name prefers a renderer that ends in "disable_thinking".
# These two lines make it return the first recommended renderer (thinking enabled) and check the edit.
THINKING_ON = [
    r"""sed -i 's/return next((n for n in names if n.endswith("disable_thinking")), names\[0\])/return names[0]/' lexgym/tinker_env.py""",
    "grep -q 'return names\\[0\\]$' lexgym/tinker_env.py",
]


# Keeps only the listed task types in the downloaded task files (argv[1]: train types, argv[2]: eval types).
TASK_FILTER = """import json, sys
from pathlib import Path
d = Path('data/lexgym_tasks')
train, evals = set(sys.argv[1].split(',')), set(sys.argv[2].split(','))
for name, keep in (('rl_train.jsonl', train), ('eval_heldout_rule.jsonl', evals), ('eval_recall.jsonl', evals)):
    rows = [l for l in (d / name).read_text(encoding='utf-8').splitlines() if l.strip() and json.loads(l)['task'] in keep]
    (d / name).write_text(''.join(r + chr(10) for r in rows), encoding='utf-8')
    print('task filter:', name, len(rows))
"""


def slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model_name.split("/")[-1].lower()).strip("-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--load-checkpoint-path", default=None, help="Optional SFT state path (tinker://.../weights/...).")
    ap.add_argument("--stage", default="rl-from-base")
    ap.add_argument("--experiment", default="rl_full_2026_11_12")
    ap.add_argument("--commit", default=DEFAULT_COMMIT)
    ap.add_argument("--max-steps", type=int, default=None, help="Default: one pass over the RL tasks.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--eval-limit-per-task", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=25000, help="Sampling limit per turn, thinking included.")
    ap.add_argument("--thinking", choices=["on", "off"], default="on",
                    help="on: use the model's first recommended renderer (thinking enabled). "
                         "off: the repo default, which prefers a renderer without a thinking block.")
    ap.add_argument("--train-tasks", default=None,
                    help="Comma list of task types to train on (default: all). Example: syllabify,pos,root")
    ap.add_argument("--eval-tasks", default=None,
                    help="Comma list of task types to evaluate on (default: same as --train-tasks, or all).")
    ap.add_argument("--tinker-version", default="0.39.0")
    ap.add_argument("--cookbook-version", default="0.6.1")
    ap.add_argument("--tasks-repo", default="HarleyCooper/riggs1890-lexicon-graph")
    ap.add_argument("--upload-repo", default="HarleyCooper/riggs1890-lexgym-traces")
    ap.add_argument("--flavor", default="cpu-basic")
    ap.add_argument("--timeout", default="48h")
    a = ap.parse_args()

    try:  # keys may live in a local .env file, as for lexgym.train_rl
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    secrets = {k: os.environ[k] for k in ("TINKER_API_KEY", "WANDB_API_KEY", "HF_TOKEN")}
    name = f"{a.experiment}-{a.stage}-{slug(a.model_name)}"
    log_path = f"lexgym_runs/{name}"
    prefix = f"{a.experiment}/{slug(a.model_name)}/{a.stage}"

    train = [
        "python", "-m", "lexgym.train_rl",
        "--model-name", a.model_name,
        "--task-dir", "data/lexgym_tasks",
        "--log-path", log_path,
        "--batch-size", str(a.batch_size),
        "--group-size", str(a.group_size),
        "--learning-rate", str(a.learning_rate),
        "--eval-every", str(a.eval_every),
        "--save-every", str(a.save_every),
        "--eval-limit-per-task", str(a.eval_limit_per_task),
        "--max-tokens", str(a.max_tokens),
        "--stage", a.stage,
        "--wandb-name", name,
    ]
    if a.load_checkpoint_path:
        train += ["--load-checkpoint-path", a.load_checkpoint_path]
    if a.max_steps is not None:
        train += ["--max-steps", str(a.max_steps)]

    fetch = (f"from lexgym.recall_curve import fetch_tasks; "
             f"fetch_tasks({a.tasks_repo!r}, 'data/lexgym_tasks')")
    upload = (f"from pathlib import Path; from lexgym.recall_curve import upload; "
              f"upload({a.upload_repo!r}, Path({log_path!r}), {prefix!r})")

    script = "\n".join([
        "set -e",
        "pip install -q --index-url https://download.pytorch.org/whl/cpu torch",
        f"pip install -q tinker=={a.tinker_version} tinker-cookbook=={a.cookbook_version} "
        "wandb python-dotenv huggingface_hub datasets",
        "mkdir repo && cd repo && git init -q",
        f"GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 {REPO_URL} {a.commit}",
        "GIT_LFS_SKIP_SMUDGE=1 git checkout -q FETCH_HEAD",
        f"python -c {shlex.quote(fetch)}",
        *THINKING_ON[: (2 if a.thinking == "on" else 0)],
        *([f"python -c {shlex.quote(TASK_FILTER)} {shlex.quote(a.train_tasks or a.eval_tasks)} "
           f"{shlex.quote(a.eval_tasks or a.train_tasks)}"] if (a.train_tasks or a.eval_tasks) else []),
        # Upload the run folder when training ends for any reason.
        f"trap {shlex.quote('python -c ' + shlex.quote(upload))} EXIT",
        shlex.join(train),
    ])
    job = HfApi().run_job(image="python:3.12", command=["bash", "-c", script], secrets=secrets,
                          flavor=a.flavor, timeout=a.timeout)
    print(job.id, job.url)
    print("W&B run name:", name)
    print("Traces prefix:", f"{a.upload_repo}/{prefix}")


if __name__ == "__main__":
    main()
