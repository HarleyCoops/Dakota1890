#!/usr/bin/env python3
"""Smoke-check Dakota RL reward channels before paid reruns."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_PACKAGE = ROOT / "environments" / "dakota_grammar_translation"

os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

if str(ENVIRONMENT_PACKAGE) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENT_PACKAGE))

from dakota_grammar_translation import load_environment  # noqa: E402


LEDGER_KEYS = (
    "w_exact",
    "w_char",
    "w_pattern",
    "w_affix",
    "w_length",
    "contrib_exact",
    "contrib_char",
    "contrib_pattern",
    "contrib_affix",
    "exact_match_raw",
    "char_overlap_raw",
    "pattern_raw",
    "affix_raw",
    "length_penalty_raw",
    "exact_match_norm",
    "char_overlap_norm",
    "pattern_norm",
    "affix_norm",
    "length_penalty_norm",
    "difficulty_multiplier",
    "composite_pre",
    "composite_with_length",
    "composite_with_difficulty",
    "composite_predicted",
    "composite_diff",
    "reward_scalar",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load the packaged Dakota grammar environment, score known-answer "
            "completions, and fail if pattern-bearing tasks do not emit a "
            "nonzero pattern reward."
        )
    )
    parser.add_argument("--dataset-path", default=None, help="Optional JSONL dataset override.")
    parser.add_argument("--task-filter", nargs="*", default=["identify_pattern"], help="Task types to check.")
    parser.add_argument("--max-examples", type=int, default=5, help="Number of filtered examples to score.")
    parser.add_argument("--seed", type=int, default=42, help="Dataset seed.")
    return parser


def selected_ledger_values(ledger: dict[str, Any]) -> dict[str, float | None]:
    return {
        key: float(ledger[key]) if isinstance(ledger.get(key), (int, float)) else None
        for key in LEDGER_KEYS
    }


def score_completion(env: Any, sample: dict[str, Any], completion_text: str) -> dict[str, Any]:
    completion = [{"role": "assistant", "content": completion_text}]
    reward = float(env.rubric.score(completion, sample["answer"], sample["info"]))
    ledger = env.get_reward_ledger() or {}
    return {
        "completion": completion_text,
        "reward": reward,
        "ledger": selected_ledger_values(ledger),
    }


def main() -> int:
    args = build_parser().parse_args()

    env = load_environment(
        dataset_path=args.dataset_path,
        max_examples=args.max_examples,
        eval_fraction=0,
        task_filter=args.task_filter,
        seed=args.seed,
    )

    rows: list[dict[str, Any]] = []
    pattern_nonzero = 0
    exact_nonzero = 0
    first_pattern_sample: dict[str, Any] | None = None

    for sample in env.dataset:
        pattern = sample["info"].get("verification_pattern")
        if not pattern:
            continue
        if first_pattern_sample is None:
            first_pattern_sample = sample
        gold = score_completion(env, sample, sample["answer"])
        ledger = gold["ledger"]
        pattern_nonzero += int((ledger.get("pattern_raw") or 0.0) > 0.0)
        exact_nonzero += int((ledger.get("exact_match_raw") or 0.0) > 0.0)
        rows.append(
            {
                "id": sample["id"],
                "task": sample["task"],
                "answer": sample["answer"],
                "verification_pattern": pattern,
                "gold_completion": gold,
            }
        )

    verbose_probe = None
    if first_pattern_sample is not None:
        verbose_probe = score_completion(
            env,
            first_pattern_sample,
            f"The grammatical pattern is {first_pattern_sample['answer']}.",
        )

    result = {
        "dataset_path": str(args.dataset_path) if args.dataset_path else "packaged default",
        "task_filter": args.task_filter,
        "checked": len(rows),
        "gold_exact_nonzero": exact_nonzero,
        "gold_pattern_nonzero": pattern_nonzero,
        "verbose_probe": verbose_probe,
        "rows": rows,
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))

    if not rows:
        print("ERROR: no pattern-bearing examples were checked.", file=sys.stderr)
        return 1
    if pattern_nonzero != len(rows):
        print("ERROR: pattern_raw stayed zero for at least one checked gold completion.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
