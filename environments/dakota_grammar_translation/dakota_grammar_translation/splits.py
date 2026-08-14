"""Deterministic train / held-out split for Dakota grammar tasks."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SPLIT_SEED = 42
HELD_OUT_FRACTION = 0.1
SPLIT_ALGORITHM = (
    "Read grammar_tasks_complete.jsonl in file order, skipping blank lines. "
    "Assign task_id = dakota_{sha256(json({prompt, answer, rule_id, index}))[:16]}. "
    "Stratify by (task_type × difficulty). Within each stratum, shuffle indices "
    "with random.Random(42) and take round(n * 0.1) held-out rows. "
    "Train then drops every row whose prompt matches a held-out prompt "
    "(same question, different answer still counts as overlap). "
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

    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(annotated):
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        task_type = str(info.get("task_type") or row.get("task_type") or "default")
        difficulty = str(info.get("difficulty") or row.get("difficulty") or "medium")
        buckets[(task_type, difficulty)].append(idx)

    rng = random.Random(seed)
    held_indices: set[int] = set()
    for key in sorted(buckets):
        order = list(buckets[key])
        rng.shuffle(order)
        n_held = int(round(len(order) * heldout_fraction))
        held_indices.update(order[:n_held])

    heldout = [row for idx, row in enumerate(annotated) if idx in held_indices]
    held_prompts = {
        str(row.get("prompt") or row.get("question") or "").strip()
        for row in heldout
    }
    train = [
        row
        for idx, row in enumerate(annotated)
        if idx not in held_indices
        and str(row.get("prompt") or row.get("question") or "").strip() not in held_prompts
    ]
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
    eval_prompts = {prompt for prompt, _answer in eval_pairs}
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
        if pair[0] in eval_prompts:
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
