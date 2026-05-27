#!/usr/bin/env python3
"""Audit Tinker metrics.jsonl for Dakota reward ledger channels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_KEYS = [
    "w_exact",
    "w_char",
    "w_pattern",
    "w_affix",
    "w_length",
    "difficulty_multiplier",
    "contrib_exact",
    "contrib_char",
    "contrib_pattern",
    "contrib_affix",
    "exact_match_raw",
    "pattern_raw",
    "char_overlap_raw",
    "affix_raw",
    "length_penalty_raw",
    "exact_match_norm",
    "char_overlap_norm",
    "pattern_norm",
    "affix_norm",
    "length_penalty_norm",
    "composite_pre",
    "composite_with_length",
    "composite_with_difficulty",
    "composite_predicted",
    "reward_scalar",
    "composite_diff",
    "parse_success",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify that Tinker metrics.jsonl contains Dakota reward ledger diagnostics."
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to Tinker metrics.jsonl.",
    )
    parser.add_argument(
        "--require-key",
        nargs="*",
        default=DEFAULT_REQUIRED_KEYS,
        help="Short ledger keys that must be present after the ledger/ prefix.",
    )
    parser.add_argument(
        "--require-nonzero",
        nargs="*",
        default=[],
        help="Short ledger keys that must have at least one nonzero numeric value.",
    )
    return parser


def update_stat(stats: dict[str, dict[str, Any]], key: str, value: Any) -> None:
    if not isinstance(value, (int, float)):
        return

    numeric = float(value)
    stat = stats.setdefault(
        key,
        {
            "count": 0,
            "nonzero": 0,
            "min": numeric,
            "max": numeric,
            "last": numeric,
        },
    )
    stat["count"] += 1
    stat["nonzero"] += int(numeric != 0.0)
    stat["min"] = min(stat["min"], numeric)
    stat["max"] = max(stat["max"], numeric)
    stat["last"] = numeric


def collect_stats(metrics_path: Path) -> tuple[int, int, dict[str, dict[str, Any]]]:
    rows = 0
    ledger_rows = 0
    stats: dict[str, dict[str, Any]] = {}

    with metrics_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            rows += 1
            entry = json.loads(raw_line)
            has_ledger = False
            for key, value in entry.items():
                if "ledger/" not in key:
                    continue
                short_key = key.split("ledger/", 1)[1]
                update_stat(stats, short_key, value)
                has_ledger = True
            ledger_rows += int(has_ledger)

    return rows, ledger_rows, stats


def main() -> int:
    args = build_parser().parse_args()
    metrics_path = Path(args.metrics)

    if not metrics_path.exists():
        print(f"ERROR: metrics file not found: {metrics_path}", file=sys.stderr)
        return 1

    rows, ledger_rows, stats = collect_stats(metrics_path)
    summary = {
        "metrics": str(metrics_path),
        "rows": rows,
        "ledger_rows": ledger_rows,
        "required_keys": args.require_key,
        "required_nonzero": args.require_nonzero,
        "ledger_stats": {key: stats.get(key) for key in sorted(stats)},
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))

    if rows == 0:
        print("ERROR: metrics file has no rows.", file=sys.stderr)
        return 1
    if ledger_rows == 0:
        print("ERROR: metrics file has no ledger rows.", file=sys.stderr)
        return 1

    missing = [key for key in args.require_key if key not in stats]
    if missing:
        print(f"ERROR: missing required ledger keys: {missing}", file=sys.stderr)
        return 1

    still_zero = [
        key
        for key in args.require_nonzero
        if key not in stats or stats[key].get("nonzero", 0) == 0
    ]
    if still_zero:
        print(f"ERROR: required nonzero ledger keys stayed zero: {still_zero}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
