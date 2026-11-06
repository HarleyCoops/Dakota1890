"""v3 merge copies existing rows only; frozen holdout v1 stays byte-identical."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_RL = ROOT / "scripts" / "rl"
if str(SCRIPTS_RL) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_RL))

from merge_grammar_tasks_v3 import (  # noqa: E402
    DEFAULT_ADAPTIVE,
    DEFAULT_OUTPUT,
    merge_v3_files,
    merge_v3_rows,
)

HELD_OUT_V1 = ROOT / "dakota_rl_training" / "datasets" / "grammar_tasks_heldout.jsonl"
COMPLETE_V2 = ROOT / "dakota_rl_training" / "datasets" / "grammar_tasks_complete_v2.jsonl"
TINKER_TRAIN = ROOT / "dakota_rl_training" / "tinker_train.py"
REPORT = ROOT / "dakota_rl_training" / "datasets" / "grammar_tasks_complete_v3_report.json"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _git_blob_sha256(relpath: str) -> str:
    payload = subprocess.check_output(
        ["git", "show", f"HEAD:{relpath}"],
        cwd=ROOT,
    )
    return hashlib.sha256(payload).hexdigest()


def _row(prompt: str, answer: str, task_type: str = "word_translation") -> dict:
    return {
        "prompt": prompt,
        "answer": answer,
        "info": {"task_type": task_type, "difficulty": "easy"},
    }


def test_frozen_holdout_v1_is_byte_identical_to_head() -> None:
    current = hashlib.sha256(HELD_OUT_V1.read_bytes()).hexdigest()
    head = _git_blob_sha256("dakota_rl_training/datasets/grammar_tasks_heldout.jsonl")
    assert current == head
    assert len(_load_jsonl(HELD_OUT_V1)) == 1060


def test_adaptive_filtered_and_v3_jsonl_are_not_invented_in_repo() -> None:
    assert not DEFAULT_ADAPTIVE.is_file()
    assert not DEFAULT_OUTPUT.is_file()


def test_tinker_train_keeps_v2_default_until_v3_is_committed() -> None:
    text = TINKER_TRAIN.read_text(encoding="utf-8")
    assert "grammar_tasks_complete_v2.jsonl" in text
    assert "grammar_tasks_complete_v3.jsonl" not in text
    assert "grammar_tasks_heldout.jsonl" in text


def test_placeholder_report_documents_local_launch_counts() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["heldout_v1_untouched"] is True
    assert report["live_rubric_unchanged"] is True
    assert report["v3_jsonl_written"] is False
    assert report["before"]["v2_rows"] == 9287
    assert report["before"]["adaptive_filtered"] == 523
    assert report["after"]["adaptive_already_in_v2"] == 192
    assert report["after"]["adaptive_holdout_blocked"] == 0
    assert report["after"]["adaptive_new"] == 331
    assert report["after"]["v3_unique_rows"] == 9618
    assert report["after"]["reverse_translation_unique"] == 2265
    assert report["after"]["v3_train_rows_after_reverse_upsample"] == 11883
    assert report["after"]["holdout_v1_rows"] == 1060
    assert len(_load_jsonl(COMPLETE_V2)) == 9287


def test_merge_keeps_v2_adds_new_adaptive_and_blocks_holdout_prompts() -> None:
    v2 = [
        _row("Translate kaška", "to bind"),
        _row("English for pidá", "glad", "reverse_translation"),
    ]
    holdout = [_row("Held-out prompt", "held-out gold")]
    adaptive = [
        _row("Translate kaška", "to bind"),
        _row("Held-out prompt", "a different gold"),
        _row("English for wópida", "gladness", "reverse_translation"),
        _row("Identify šni", "[verb] šni", "identify_pattern"),
    ]

    train, counts = merge_v3_rows(v2, adaptive, holdout)

    assert counts["v2_rows"] == 2
    assert counts["adaptive_filtered"] == 4
    assert counts["adaptive_already_in_v2"] == 1
    assert counts["adaptive_holdout_blocked"] == 1
    assert counts["adaptive_new"] == 2
    assert counts["v3_unique_rows"] == 4
    assert counts["reverse_translation_unique"] == 2
    assert counts["v3_train_rows_after_reverse_upsample"] == 6
    assert counts["holdout_v1_rows"] == 1

    unique_pairs = [(row["prompt"], row["answer"]) for row in train[:4]]
    assert unique_pairs == [
        ("Translate kaška", "to bind"),
        ("English for pidá", "glad"),
        ("English for wópida", "gladness"),
        ("Identify šni", "[verb] šni"),
    ]
    assert [row["answer"] for row in train[4:]] == ["glad", "gladness"]
    assert all(
        (row.get("info") or {}).get("task_type") == "reverse_translation"
        for row in train[4:]
    )
    assert "Held-out prompt" not in {row["prompt"] for row in train}


def test_merge_refuses_missing_adaptive_and_holdout_overwrite(tmp_path: Path) -> None:
    v2_path = tmp_path / "v2.jsonl"
    holdout_path = tmp_path / "grammar_tasks_heldout.jsonl"
    missing_adaptive = tmp_path / "adaption_adapted_v2_filtered.jsonl"
    output_path = tmp_path / "v3.jsonl"
    report_path = tmp_path / "report.json"

    v2_path.write_text(
        json.dumps(_row("Translate kaška", "to bind"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    holdout_bytes = (
        json.dumps(_row("Held-out prompt", "held-out gold"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    holdout_path.write_bytes(holdout_bytes)

    with pytest.raises(FileNotFoundError, match="Do not invent v3 rows"):
        merge_v3_files(v2_path, missing_adaptive, holdout_path, output_path, report_path)
    assert not output_path.exists()
    assert holdout_path.read_bytes() == holdout_bytes

    with pytest.raises(ValueError, match="holdout"):
        merge_v3_files(v2_path, missing_adaptive, holdout_path, holdout_path, report_path)
    assert holdout_path.read_bytes() == holdout_bytes


def test_merge_writes_v3_from_existing_rows_only(tmp_path: Path) -> None:
    v2_path = tmp_path / "v2.jsonl"
    adaptive_path = tmp_path / "adaption_adapted_v2_filtered.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    output_path = tmp_path / "v3.jsonl"
    report_path = tmp_path / "report.json"

    v2_rows = [
        _row("Translate kaška", "to bind"),
        _row("English for pidá", "glad", "reverse_translation"),
    ]
    adaptive_rows = [
        _row("English for wópida", "gladness", "reverse_translation"),
    ]
    holdout_rows = [_row("Held-out prompt", "held-out gold")]

    v2_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in v2_rows),
        encoding="utf-8",
    )
    adaptive_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in adaptive_rows),
        encoding="utf-8",
    )
    holdout_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in holdout_rows),
        encoding="utf-8",
    )

    report = merge_v3_files(v2_path, adaptive_path, holdout_path, output_path, report_path)
    written = _load_jsonl(output_path)
    assert report["v3_jsonl_written"] is True
    assert report["after"]["v3_unique_rows"] == 3
    assert report["after"]["v3_train_rows_after_reverse_upsample"] == 5
    assert [row["answer"] for row in written] == [
        "to bind",
        "glad",
        "gladness",
        "glad",
        "gladness",
    ]
    assert holdout_path.read_text(encoding="utf-8").count("\n") == 1
