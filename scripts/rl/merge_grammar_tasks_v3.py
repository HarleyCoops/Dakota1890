#!/usr/bin/env python3
"""Merge repaired v2 train JSONL with filtered Adaptive rows into v3.

Recipe (data only; does not change the live rubric):

1. Keep every grammar_tasks_complete_v2.jsonl row.
2. Append Adaptive rows from adaption_adapted_v2_filtered.jsonl whose
   (prompt, answer) pair is not already in v2 and whose prompt is not in
   frozen holdout v1 (grammar_tasks_heldout.jsonl).
3. Append a second copy of every reverse_translation row (2x upsample).

Does not invent Dakota forms. Does not overwrite holdout v1.
Refuses to write v3 if the Adaptive filtered file is missing.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_V2 = ROOT / "dakota_rl_training" / "datasets" / "grammar_tasks_complete_v2.jsonl"
DEFAULT_ADAPTIVE = (
    ROOT / "dakota_rl_training" / "datasets" / "adaption_adapted_v2_filtered.jsonl"
)
DEFAULT_HOLDOUT = ROOT / "dakota_rl_training" / "datasets" / "grammar_tasks_heldout.jsonl"
DEFAULT_OUTPUT = ROOT / "dakota_rl_training" / "datasets" / "grammar_tasks_complete_v3.jsonl"
DEFAULT_REPORT = (
    ROOT / "dakota_rl_training" / "datasets" / "grammar_tasks_complete_v3_report.json"
)

REVERSE_TASK = "reverse_translation"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prompt(row: dict[str, Any]) -> str:
    return str(row.get("prompt") or row.get("question") or "").strip()


def _answer(row: dict[str, Any]) -> str:
    return str(row.get("answer") or "").strip()


def _pair(row: dict[str, Any]) -> tuple[str, str]:
    return (_prompt(row), _answer(row))


def _task_type(row: dict[str, Any]) -> str:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    return str(info.get("task_type") or row.get("task_type") or "")


def merge_v3_rows(
    v2_rows: list[dict[str, Any]],
    adaptive_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build unique v3 rows, then 2x-upsample reverse_translation."""

    v2_pairs = {_pair(row) for row in v2_rows}
    holdout_prompts = {_prompt(row) for row in holdout_rows}

    unique: list[dict[str, Any]] = [copy.deepcopy(row) for row in v2_rows]
    seen_pairs = set(v2_pairs)
    adaptive_already_in_v2 = 0
    adaptive_holdout_blocked = 0
    adaptive_new = 0

    for row in adaptive_rows:
        pair = _pair(row)
        if pair in v2_pairs:
            adaptive_already_in_v2 += 1
            continue
        if _prompt(row) in holdout_prompts:
            adaptive_holdout_blocked += 1
            continue
        if pair in seen_pairs:
            continue
        unique.append(copy.deepcopy(row))
        seen_pairs.add(pair)
        adaptive_new += 1

    reverse_rows = [row for row in unique if _task_type(row) == REVERSE_TASK]
    train_rows = unique + [copy.deepcopy(row) for row in reverse_rows]

    report = {
        "v2_rows": len(v2_rows),
        "adaptive_filtered": len(adaptive_rows),
        "adaptive_already_in_v2": adaptive_already_in_v2,
        "adaptive_holdout_blocked": adaptive_holdout_blocked,
        "adaptive_new": adaptive_new,
        "v3_unique_rows": len(unique),
        "v3_train_rows_after_reverse_upsample": len(train_rows),
        "reverse_translation_unique": len(reverse_rows),
        "holdout_v1_rows": len(holdout_rows),
    }
    return train_rows, report


def build_report(
    counts: dict[str, Any],
    *,
    v3_written: bool,
    adaptive_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    return {
        "source_complete_v2": "dakota_rl_training/datasets/grammar_tasks_complete_v2.jsonl",
        "adaptive_filtered": "dakota_rl_training/datasets/adaption_adapted_v2_filtered.jsonl",
        "frozen_heldout_v1": "dakota_rl_training/datasets/grammar_tasks_heldout.jsonl",
        "merged_complete_v3": "dakota_rl_training/datasets/grammar_tasks_complete_v3.jsonl",
        "heldout_v1_untouched": True,
        "live_rubric_unchanged": True,
        "v3_jsonl_written": v3_written,
        "adaptive_filtered_present": adaptive_path.is_file(),
        "output_path": str(output_path),
        "before": {
            "v2_rows": counts["v2_rows"],
            "adaptive_filtered": counts["adaptive_filtered"],
            "holdout_v1_rows": counts["holdout_v1_rows"],
        },
        "after": {
            "adaptive_already_in_v2": counts["adaptive_already_in_v2"],
            "adaptive_holdout_blocked": counts["adaptive_holdout_blocked"],
            "adaptive_new": counts["adaptive_new"],
            "v3_unique_rows": counts["v3_unique_rows"],
            "reverse_translation_unique": counts["reverse_translation_unique"],
            "v3_train_rows_after_reverse_upsample": counts[
                "v3_train_rows_after_reverse_upsample"
            ],
            "holdout_v1_rows": counts["holdout_v1_rows"],
        },
        "notes": [
            "v3 = all v2 rows + Adaptive rows whose (prompt, answer) is new "
            "and whose prompt is not in holdout v1, then reverse_translation 2x.",
            "Holdout v1 is the cebp9acs eval set and must not be overwritten.",
            "The live DakotaGrammarRubric is unchanged. Empty required_affixes still score 0.0.",
            "This script copies existing attested rows only. It does not invent Dakota forms.",
        ],
    }


def merge_v3_files(
    v2_path: Path,
    adaptive_path: Path,
    holdout_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    if output_path.resolve() == holdout_path.resolve():
        raise ValueError("Refusing to write v3 over frozen holdout v1.")
    if not adaptive_path.is_file():
        raise FileNotFoundError(
            f"Adaptive filtered file is missing: {adaptive_path}. "
            "Do not invent v3 rows. Place adaption_adapted_v2_filtered.jsonl "
            "next to the v2 train JSONL, then rerun this script."
        )

    v2_rows = load_jsonl(v2_path)
    adaptive_rows = load_jsonl(adaptive_path)
    holdout_rows = load_jsonl(holdout_path)
    train_rows, counts = merge_v3_rows(v2_rows, adaptive_rows, holdout_rows)
    write_jsonl(output_path, train_rows)
    report = build_report(
        counts,
        v3_written=True,
        adaptive_path=adaptive_path,
        output_path=output_path,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-path", type=Path, default=DEFAULT_V2)
    parser.add_argument("--adaptive-path", type=Path, default=DEFAULT_ADAPTIVE)
    parser.add_argument("--holdout-path", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    holdout = args.holdout_path.resolve()
    if holdout != DEFAULT_HOLDOUT.resolve() and holdout.name == "grammar_tasks_heldout.jsonl":
        print(
            f"Refusing to use a non-default holdout v1 path: {args.holdout_path}",
            file=sys.stderr,
        )
        return 2

    try:
        report = merge_v3_files(
            args.v2_path.resolve(),
            args.adaptive_path.resolve(),
            args.holdout_path.resolve(),
            args.output_path.resolve(),
            args.report_path.resolve(),
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
