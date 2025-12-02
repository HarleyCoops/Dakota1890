#!/usr/bin/env python3
"""
Run inference against the published Railroad 1959 Tinker checkpoint.

Examples (PowerShell):
  $env:TINKER_API_KEY='...'  # required
  python RailroadEngineer1959/railroad_inference.py --task-id 713A-001
  python RailroadEngineer1959/railroad_inference.py --prompt "Give the procedure when meeting a westbound freight at a siding."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from tinker import EncodedTextChunk, ModelInput, SamplingParams, ServiceClient
from tinker_cookbook.tokenizer_utils import get_tokenizer


DEFAULT_CHECKPOINT = "tinker://e9732434-7315-5c9c-8de3-6e7d937b8ec5:train:0/sampler_weights/final"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_DATASET = Path("RailroadEngineer1959/data/railroad_extracted/safety_tasks_complete.json")


def load_task(task_id: str, dataset_path: Path) -> dict:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    for entry in data:
        if entry.get("task_id") == task_id:
            return entry
    raise ValueError(f"Task ID {task_id} not found in {dataset_path}")


def build_prompt(task: Optional[dict], user_prompt: Optional[str]) -> str:
    if user_prompt:
        return user_prompt
    if not task:
        raise ValueError("Either --task-id or --prompt must be provided.")
    return (
        "You are a railroad safety student. Respond with the correct procedure.\n"
        f"Task ID: {task.get('task_id')}\n"
        f"Scenario: {task.get('description')}\n"
        "What should your crew do?"
    )


def run_inference(
    checkpoint: str,
    base_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: Optional[str] = None,
) -> str:
    tokenizer = get_tokenizer(base_model)
    chunk = EncodedTextChunk(tokens=tokenizer.encode(prompt))
    model_input = ModelInput(chunks=[chunk])
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
    )

    svc = ServiceClient()
    client = svc.create_sampling_client(model_path=checkpoint)
    resp = client.sample(prompt=model_input, num_samples=1, sampling_params=params).result()
    seq = resp.sequences[0]
    return tokenizer.decode(seq.tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for Railroad 1959 Tinker checkpoint.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Tinker sampler checkpoint URI.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model name for tokenization.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET), help="Path to safety tasks JSON.")
    parser.add_argument("--task-id", default=None, help="Task ID from the dataset to build the prompt.")
    parser.add_argument("--prompt", default=None, help="Custom free-form prompt (overrides --task-id).")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--stop", default=None, help="Optional stop sequence.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task = None
    if args.task_id and not args.prompt:
        task = load_task(args.task_id, Path(args.dataset_path))

    prompt = build_prompt(task, args.prompt)
    output = run_inference(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=args.stop,
    )
    print("\n=== MODEL OUTPUT ===\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
