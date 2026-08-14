#!/usr/bin/env python3
"""Materialize the registered Dakota train/held-out split (seed=42, 10%)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_PKG = ROOT / "environments" / "dakota_grammar_translation"
if str(ENV_PKG) not in sys.path:
    sys.path.insert(0, str(ENV_PKG))

from dakota_grammar_translation.splits import (
    HELD_OUT_FRACTION,
    SPLIT_ALGORITHM,
    SPLIT_SEED,
    split_entries,
    write_jsonl,
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    source = ROOT / "dakota_rl_training" / "datasets" / "grammar_tasks_complete.jsonl"
    heldout_path = ROOT / "dakota_rl_training" / "datasets" / "grammar_tasks_heldout.jsonl"
    manifest_path = ROOT / "dakota_rl_training" / "datasets" / "splits" / "SPLIT_MANIFEST.json"
    rows = load_jsonl(source)
    split = split_entries(rows, seed=SPLIT_SEED, heldout_fraction=HELD_OUT_FRACTION)
    write_jsonl(heldout_path, split.heldout)
    manifest = {
        "source": "dakota_rl_training/datasets/grammar_tasks_complete.jsonl",
        "heldout_path": "dakota_rl_training/datasets/grammar_tasks_heldout.jsonl",
        "seed": SPLIT_SEED,
        "heldout_fraction": HELD_OUT_FRACTION,
        "algorithm": SPLIT_ALGORITHM,
        "used_for_grpo": False,
        "leaked_gold_stripped_at_load": True,
        "train_count": len(split.train),
        "heldout_count": len(split.heldout),
        "heldout_ids": [row["task_id"] for row in split.heldout],
        "train_ids": [row["task_id"] for row in split.train],
        "prompt_overlap_excluded_from_train": True,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(split.heldout)} held-out rows to {heldout_path}")
    print(f"Train rows after prompt-overlap exclusion: {len(split.train)}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
