from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterator

LEDGER_FIELDS = [
    "step",
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
    "char_overlap_raw",
    "pattern_raw",
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


def _extract_ledger_payload(entry: Dict[str, float]) -> dict[str, float]:
    payload: Dict[str, float] = {}
    for key, value in entry.items():
        # Tinker uses names like "env/all/ledger/affix_raw" and "test/env/all/ledger/affix_raw".
        # We only care about the portion after the final "ledger/" segment.
        if "ledger/" in key:
            payload[key.split("ledger/", 1)[1]] = value
    if "parse_success" not in payload:
        parse_val = entry.get("ledger/parse_success") or entry.get("env/all/ledger/parse_success")
        if parse_val is not None:
            payload["parse_success"] = parse_val
    return payload


def iter_reward_ledger(metrics_log: Path) -> Iterator[dict[str, float]]:
    """Yield ledger rows from a Tinker metrics.jsonl file."""

    with metrics_log.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            entry = json.loads(raw_line)
            ledger = _extract_ledger_payload(entry)
            if not ledger:
                continue
            step = entry.get("step")
            if step is None:
                step = idx
            ledger_row = {"step": step}
            ledger_row.update(ledger)
            yield ledger_row


def export_reward_ledger(metrics_log: Path, output_csv: Path) -> Path:
    """Export ledger values from metrics.jsonl into a CSV consumable by viz scripts."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = list(iter_reward_ledger(metrics_log))
    if not rows:
        if output_csv.exists():
            output_csv.unlink()
        return output_csv

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in LEDGER_FIELDS})
    return output_csv
