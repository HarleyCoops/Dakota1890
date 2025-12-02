#!/usr/bin/env python3
"""
Local evaluation of the Tinker sampler checkpoint against the railroad safety tasks.

Usage (PowerShell):
  $env:TINKER_API_KEY='tml-...'
  python RailroadEngineer1959/railroad_local_eval.py --max-examples 50 --max-tokens 256

Defaults:
  checkpoint: tinker://.../sampler_weights/final
  base-model: Qwen/Qwen3-4B-Instruct-2507
  tasks: RailroadEngineer1959/data/railroad_extracted/safety_tasks_complete.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import sys
from tinker import EncodedTextChunk, ModelInput, SamplingParams, ServiceClient
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Ensure local package import works regardless of CWD
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from RailroadEngineer1959.environments.railroad_1959.railroad_1959.environment import (  # type: ignore
    RailroadRubric,
    DEFAULT_SYSTEM_PROMPT,
)


DEFAULT_CHECKPOINT = "tinker://e9732434-7315-5c9c-8de3-6e7d937b8ec5:train:0/sampler_weights/final"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_TASKS = Path("RailroadEngineer1959/data/railroad_extracted/safety_tasks_complete.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local eval against Tinker sampler checkpoint.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Tinker sampler checkpoint URI.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model for tokenization.")
    parser.add_argument("--tasks-path", default=str(DEFAULT_TASKS), help="Path to tasks JSON.")
    parser.add_argument("--max-examples", type=int, default=50, help="Limit number of examples (-1 for all).")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generation tokens.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = json.loads(Path(args.tasks_path).read_text(encoding="utf-8"))
    if args.max_examples > 0:
        tasks = tasks[: args.max_examples]

    tokenizer = get_tokenizer(args.base_model)
    rubric = RailroadRubric()
    svc = ServiceClient()
    client = svc.create_sampling_client(model_path=args.checkpoint)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)

    scores: list[float] = []
    for task in tasks:
        prompt = (
            f"{DEFAULT_SYSTEM_PROMPT}\n"
            f"Task ID: {task['task_id']}\n"
            f"Scenario: {task['description']}\n"
            "What should your crew do?"
        )
        chunk = EncodedTextChunk(tokens=tokenizer.encode(prompt))
        model_input = ModelInput(chunks=[chunk])
        resp = client.sample(prompt=model_input, num_samples=1, sampling_params=params).result()
        seq = resp.sequences[0]
        text = tokenizer.decode(seq.tokens)

        completion = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
        ]
        reward = rubric.score(completion, task["expected_outcome"], {"description": task["description"]})
        scores.append(reward)

    if not scores:
        print("No scores computed.")
        return 1
    med = sorted(scores)[len(scores) // 2]
    print(f"Evaluated {len(scores)} tasks")
    print(f"Mean reward: {mean(scores):.4f}")
    print(f"Median reward: {med:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
