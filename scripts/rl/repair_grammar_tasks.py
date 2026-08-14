#!/usr/bin/env python3
"""Regenerate repaired Dakota train JSONL from attested in-repo sources.

Writes grammar_tasks_complete_v2.jsonl and grammar_tasks_heldout_v2.jsonl.
Does not overwrite the frozen v1 complete/holdout files used by cebp9acs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dakota_extraction.datasets.grammar_task_repair import repair_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: inferred from this script).",
    )
    args = parser.parse_args()
    report = repair_dataset(args.root.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
