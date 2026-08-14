"""Repaired train JSONL must be scorable; frozen holdout v1 must stay byte-identical."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PKG = ROOT / "environments" / "dakota_grammar_translation"
if str(ENV_PKG) not in sys.path:
    sys.path.insert(0, str(ENV_PKG))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dakota_extraction.datasets.grammar_task_repair import (  # noqa: E402
    is_dakota_gold,
    is_placeholder_gold,
    load_attested_lexicon,
    required_affixes_scorable,
)
from dakota_grammar_translation.train_reward import _affix_bearing_tokens  # noqa: E402

COMPLETE_V1 = ROOT / "dakota_rl_training" / "datasets" / "grammar_tasks_complete.jsonl"
COMPLETE_V2 = ROOT / "dakota_rl_training" / "datasets" / "grammar_tasks_complete_v2.jsonl"
HELD_OUT_V1 = ROOT / "dakota_rl_training" / "datasets" / "grammar_tasks_heldout.jsonl"
HELD_OUT_V2 = ROOT / "dakota_rl_training" / "datasets" / "grammar_tasks_heldout_v2.jsonl"


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


def test_frozen_holdout_v1_is_byte_identical_to_head() -> None:
    current = hashlib.sha256(HELD_OUT_V1.read_bytes()).hexdigest()
    head = _git_blob_sha256("dakota_rl_training/datasets/grammar_tasks_heldout.jsonl")
    assert current == head
    assert len(_load_jsonl(HELD_OUT_V1)) == 1060


def test_v2_train_file_exists_and_is_not_the_frozen_complete() -> None:
    assert COMPLETE_V2.is_file()
    assert COMPLETE_V1.is_file()
    assert COMPLETE_V2.resolve() != COMPLETE_V1.resolve()
    v1 = hashlib.sha256(COMPLETE_V1.read_bytes()).hexdigest()
    v2 = hashlib.sha256(COMPLETE_V2.read_bytes()).hexdigest()
    assert v1 != v2


def test_v2_has_no_placeholder_gold() -> None:
    for idx, row in enumerate(_load_jsonl(COMPLETE_V2)):
        gold = str(row.get("answer") or "")
        assert not is_placeholder_gold(gold), f"row {idx} still has placeholder gold: {gold[:80]}"


def test_v2_nonempty_required_affixes_are_scorable_in_gold() -> None:
    for idx, row in enumerate(_load_jsonl(COMPLETE_V2)):
        info = row.get("info") or {}
        affixes = list(info.get("required_affixes") or [])
        if not affixes:
            continue
        gold = str(row.get("answer") or "")
        assert all(str(affix).strip() for affix in affixes), f"row {idx} has empty affix tokens"
        assert required_affixes_scorable(gold, affixes), (
            f"row {idx} dead affix labels {affixes!r} gold={gold!r}"
        )
        for affix in affixes:
            assert _affix_bearing_tokens(gold, affix), (
                f"row {idx} affix {affix!r} missing from gold {gold!r}"
            )


def test_v2_reverse_translation_gold_is_dakota() -> None:
    lexicon = load_attested_lexicon(ROOT)
    reverse_rows = [
        row
        for row in _load_jsonl(COMPLETE_V2)
        if (row.get("info") or {}).get("task_type") == "reverse_translation"
    ]
    assert reverse_rows, "repaired train should keep attested EN→Dakota rows"
    for idx, row in enumerate(reverse_rows):
        gold = str(row.get("answer") or "")
        assert is_dakota_gold(gold, lexicon), (
            f"reverse_translation row {idx} gold is not attested Dakota: {gold!r}"
        )


def test_heldout_v2_is_a_new_file_and_v1_unchanged() -> None:
    assert HELD_OUT_V2.is_file()
    assert HELD_OUT_V2.resolve() != HELD_OUT_V1.resolve()
    v1_prompts = {row["prompt"].strip() for row in _load_jsonl(HELD_OUT_V1)}
    for row in _load_jsonl(HELD_OUT_V2):
        assert row["prompt"].strip() in v1_prompts
        assert not is_placeholder_gold(str(row.get("answer") or ""))
