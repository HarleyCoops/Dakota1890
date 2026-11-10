"""Evaluate a base model or a Tinker checkpoint on the lexgym evaluation sets.

    python -m lexgym.evaluate --model-name Qwen/Qwen3.5-4B --out runs/eval_base
    python -m lexgym.evaluate --model-name Qwen/Qwen3.5-4B --sampler-path tinker://.../sampler_weights/final --out runs/eval_e1

Greedy decoding by default. Writes one trace row per rollout (docs/TRACE_SCHEMA.md)
and a summary.json with exact-match rate and mean reward per task type and split.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
from pathlib import Path

import tinker
from dotenv import load_dotenv
from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .scoring import SCORER_VERSION
from .tinker_env import (LexgymGroupBuilder, LexiconIndex, TraceWriter, _git_commit, _sha256, load_tasks,
                         pick_renderer_name)

EVAL_FILES = ("eval_heldout_rule.jsonl", "eval_recall.jsonl")


async def evaluate(model_name: str, sampler_path: str | None, task_dir: str, out: str, *, run_id: str = "",
                   stage: str = "", limit_per_task: int = 150, recall_limit_per_task: int = -1,
                   temperature: float = 0.0, max_tokens: int = 96, concurrency: int = 64, seed: int = 42) -> dict:
    d, out_dir = Path(task_dir), Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer_name = pick_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=get_tokenizer(model_name))
    index = LexiconIndex(d / "lookup_index.json")
    tasks = (load_tasks(d / "eval_heldout_rule.jsonl", None, limit_per_task, seed)
             + load_tasks(d / "eval_recall.jsonl", None, recall_limit_per_task, seed))
    trace = TraceWriter(out_dir / "traces", {
        "run_id": run_id, "stage": stage, "base_model": model_name, "load_checkpoint_path": sampler_path,
        "renderer": renderer_name, "temperature": temperature, "max_tokens": max_tokens,
        "train_group_size": 1, "train_batch_size": 0, "seed": seed, "lexgym_commit": _git_commit(),
        "scorer_version": SCORER_VERSION,
        "task_files_sha256": {f: _sha256(d / f) for f in (*EVAL_FILES, "lookup_index.json")},
    })
    service = tinker.ServiceClient()
    client = (service.create_sampling_client(model_path=sampler_path) if sampler_path
              else service.create_sampling_client(base_model=model_name))
    policy = TinkerTokenCompleter(client, max_tokens=max_tokens, temperature=temperature)
    gate = asyncio.Semaphore(concurrency)
    rows: list[dict] = []

    async def one(i: int, task: dict) -> None:
        builder = LexgymGroupBuilder(task, renderer, index, 1, trace,
                                     {"role": "eval", "pass_index": 0, "batch_index": 0, "group_index": i})
        async with gate:
            group = await do_group_rollout(builder, policy)
        ledger = {}
        for transition in group.trajectories_G[0].transitions:
            ledger.update({k.split("ledger/")[-1]: v for k, v in transition.metrics.items() if k.startswith("ledger/")})
        rows.append({"task": task["task"], "split": task["split"], **ledger})

    await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))
    stats: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for key in (f"task/{r['task']}", f"split/{r['split']}", "all"):
            stats[key]["exact"].append(r.get("exact", 0.0))
            stats[key]["reward"].append(r.get("reward", 0.0))
            for extra in ("token_f1", "lookups_used", "target_entry_seen", "parse_ok"):
                if extra in r:
                    stats[key][extra].append(r[extra])
    summary = {"model_name": model_name, "sampler_path": sampler_path, "run_id": run_id, "stage": stage,
               "temperature": temperature, "n_tasks": len(rows),
               "results": {k: {"n": len(v["exact"]), **{m: sum(x) / len(x) for m, x in v.items()}}
                           for k, v in sorted(stats.items())}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--sampler-path", default=None, help="tinker://.../sampler_weights/<name>; omit for the base model")
    ap.add_argument("--task-dir", default="data/lexgym_tasks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-per-task", type=int, default=150)
    ap.add_argument("--recall-limit-per-task", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--stage", default="")
    a = ap.parse_args()
    load_dotenv()
    summary = asyncio.run(evaluate(a.model_name, a.sampler_path, a.task_dir, a.out, run_id=Path(a.out).name,
                                   stage=a.stage, limit_per_task=a.limit_per_task,
                                   recall_limit_per_task=a.recall_limit_per_task, temperature=a.temperature,
                                   max_tokens=a.max_tokens))
    for key, value in summary["results"].items():
        print(f"{key:28s} n={value['n']:4d} exact={value['exact']:.3f} reward={value['reward']:.3f}")


if __name__ == "__main__":
    main()
