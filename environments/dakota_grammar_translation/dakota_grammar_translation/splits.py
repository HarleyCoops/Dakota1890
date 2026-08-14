"""Deterministic train / held-out split for Dakota grammar tasks."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SPLIT_SEED = 42
HELD_OUT_FRACTION = 0.1
SPLIT_ALGORITHM = (
    "Read grammar_tasks_complete.jsonl in file order, skipping blank lines. "
    "Assign task_id = dakota_{sha256(json({prompt, answer, rule_id, index}))[:16]}. "
    "Shuffle row indices with random.Random(42). "
    "The first round(N * 0.1) shuffled indices are held-out; the rest are train. "
    "Held-out rows are never used for GRPO advantages."
)


def assign_stable_task_id(entry: dict[str, Any], idx: int | None = None) -> str:
    stamped = entry.get("task_id")
    if stamped:
        return str(stamped)
    info = entry.get("info") if isinstance(entry.get("info"), dict) else {}
    prompt = str(entry.get("prompt") or entry.get("question") or "")
    answer = str(entry.get("answer") or "")
    rule_id = str(info.get("rule_id") or entry.get("rule_id") or "")
    payload = {
        "answer": answer,
        "index": idx,
        "prompt": prompt,
        "rule_id": rule_id,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"dakota_{digest}"


@dataclass(frozen=True)
class SplitResult:
    train: list[dict[str, Any]]
    heldout: list[dict[str, Any]]


def split_entries(
    rows: Iterable[dict[str, Any]],
    seed: int = SPLIT_SEED,
    heldout_fraction: float = HELD_OUT_FRACTION,
) -> SplitResult:
    annotated: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        copy = dict(row)
        info = dict(copy.get("info") or {}) if isinstance(copy.get("info"), dict) else {}
        copy["info"] = info
        copy["task_id"] = assign_stable_task_id(row, idx)
        annotated.append(copy)

    if len(annotated) <= 1 or heldout_fraction <= 0:
        return SplitResult(train=annotated, heldout=[])

    rng = random.Random(seed)
    order = list(range(len(annotated)))
    rng.shuffle(order)
    n_held = int(round(len(annotated) * heldout_fraction))
    held_indices = set(order[:n_held])
    train: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for idx, row in enumerate(annotated):
        if idx in held_indices:
            heldout.append(row)
        else:
            train.append(row)
    return SplitResult(train=train, heldout=heldout)


def exclude_eval_overlap(
    train_rows: Iterable[dict[str, Any]],
    eval_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    eval_ids = {
        str(row.get("id") or row.get("task_id"))
        for row in eval_rows
        if row.get("id") or row.get("task_id")
    }
    eval_pairs = {
        (
            str(row.get("question") or row.get("prompt") or "").strip(),
            str(row.get("answer") or "").strip(),
        )
        for row in eval_rows
    }
    filtered: list[dict[str, Any]] = []
    for row in train_rows:
        row_id = str(row.get("id") or row.get("task_id") or "")
        pair = (
            str(row.get("question") or row.get("prompt") or "").strip(),
            str(row.get("answer") or "").strip(),
        )
        if row_id and row_id in eval_ids:
            continue
        if pair in eval_pairs:
            continue
        filtered.append(row)
    return filtered


def load_split_manifest(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
