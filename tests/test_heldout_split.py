"""Held-out Dakota split must be deterministic and disjoint from train."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PKG = ROOT / "environments" / "dakota_grammar_translation"
if str(ENV_PKG) not in sys.path:
    sys.path.insert(0, str(ENV_PKG))

from dakota_grammar_translation.splits import (  # noqa: E402
    HELD_OUT_FRACTION,
    SPLIT_SEED,
    assign_stable_task_id,
    exclude_eval_overlap,
    load_split_manifest,
    split_entries,
)


COMPLETE = ROOT / "dakota_rl_training" / "datasets" / "grammar_tasks_complete.jsonl"
HELD_OUT = ROOT / "dakota_rl_training" / "datasets" / "grammar_tasks_heldout.jsonl"
MANIFEST = ROOT / "dakota_rl_training" / "datasets" / "splits" / "SPLIT_MANIFEST.json"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_split_seed_and_fraction_are_documented() -> None:
    assert SPLIT_SEED == 42
    assert HELD_OUT_FRACTION == 0.1
    manifest = load_split_manifest(MANIFEST)
    assert manifest["seed"] == SPLIT_SEED
    assert manifest["heldout_fraction"] == HELD_OUT_FRACTION
    assert manifest["source"] == "dakota_rl_training/datasets/grammar_tasks_complete.jsonl"
    assert "algorithm" in manifest


def test_registered_split_has_no_train_overlap() -> None:
    complete = _load_jsonl(COMPLETE)
    heldout = _load_jsonl(HELD_OUT)
    manifest = load_split_manifest(MANIFEST)

    complete_ids = [assign_stable_task_id(row, idx) for idx, row in enumerate(complete)]
    heldout_ids = {assign_stable_task_id(row) for row in heldout}
    held_prompts = {row["prompt"].strip() for row in heldout}
    train_ids = []
    for idx, row in enumerate(complete):
        task_id = complete_ids[idx]
        if task_id in heldout_ids:
            continue
        if row["prompt"].strip() in held_prompts:
            continue
        train_ids.append(task_id)

    assert len(complete) == 10576
    assert len(heldout_ids) == len(heldout)
    assert set(manifest["heldout_ids"]) == heldout_ids
    assert set(manifest["train_ids"]) == set(train_ids)
    assert set(train_ids).isdisjoint(heldout_ids)
    assert len(train_ids) + len(heldout_ids) <= len(complete_ids)
    assert len(heldout) == manifest["heldout_count"]
    assert len(train_ids) == manifest["train_count"]


def test_split_entries_is_deterministic() -> None:
    rows = [
        {"prompt": f"p{i}", "answer": f"a{i}", "info": {"rule_id": f"r{i}"}}
        for i in range(20)
    ]
    first = split_entries(rows, seed=42, heldout_fraction=0.1)
    second = split_entries(rows, seed=42, heldout_fraction=0.1)
    assert [row["task_id"] for row in first.heldout] == [row["task_id"] for row in second.heldout]
    other = split_entries(rows, seed=7, heldout_fraction=0.1)
    assert [row["task_id"] for row in first.heldout] != [row["task_id"] for row in other.heldout]


def test_split_is_stratified_by_task_type_and_difficulty() -> None:
    rows = [
        {
            "prompt": f"morph-{i}",
            "answer": f"am{i}",
            "info": {"task_type": "morphology", "difficulty": "easy", "rule_id": f"m{i}"},
        }
        for i in range(20)
    ] + [
        {
            "prompt": f"trans-{i}",
            "answer": f"at{i}",
            "info": {"task_type": "word_translation", "difficulty": "hard", "rule_id": f"t{i}"},
        }
        for i in range(20)
    ]
    split = split_entries(rows, seed=42, heldout_fraction=0.1)
    held_keys = {
        (row["info"]["task_type"], row["info"]["difficulty"])
        for row in split.heldout
    }
    assert ("morphology", "easy") in held_keys
    assert ("word_translation", "hard") in held_keys


def test_registered_split_is_stratified_and_has_no_prompt_overlap() -> None:
    complete = _load_jsonl(COMPLETE)
    heldout = _load_jsonl(HELD_OUT)
    manifest = load_split_manifest(MANIFEST)
    assert "task_type × difficulty" in manifest["algorithm"]
    held_prompts = {row["prompt"].strip() for row in heldout}
    train_id_set = set(manifest["train_ids"])
    train_prompts = {
        row["prompt"].strip()
        for idx, row in enumerate(complete)
        if assign_stable_task_id(row, idx) in train_id_set
    }
    assert train_prompts.isdisjoint(held_prompts)


def test_exclude_eval_overlap_drops_matching_prompts() -> None:
    train = [
        {"id": "a", "question": "q1", "answer": "gold"},
        {"id": "b", "question": "q2", "answer": "other"},
    ]
    eval_rows = [{"id": "heldout-a", "question": "q1", "answer": "gold"}]
    filtered = exclude_eval_overlap(train, eval_rows)
    assert [row["id"] for row in filtered] == ["b"]
