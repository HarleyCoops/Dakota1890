"""Integration checks for the packaged Dakota grammar environment."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_PACKAGE = ROOT / "environments" / "dakota_grammar_translation"

if str(ENVIRONMENT_PACKAGE) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENT_PACKAGE))

from dakota_grammar_translation import load_environment
from dakota_grammar_translation.environment import _prepare_records


def test_packaged_environment_loads_with_default_dataset() -> None:
    """The published environment should load from its packaged dataset without repo hacks."""
    env = load_environment(max_examples=4, eval_fraction=0, seed=42)

    assert len(env.dataset) == 4
    sample = env.dataset[0]
    assert sample["question"]
    assert sample["answer"]
    assert "task_type" in sample["info"]
    assert sample["info"]["task_type"] != "default"
    assert sample["info"]["verification_pattern"] == "[verb] šni"


def test_prepare_records_reads_nested_info_schema() -> None:
    """The packaged dataset stores reward metadata under entry['info']; keep it intact."""
    entry = {
        "prompt": "Identify the pattern.",
        "answer": "[verb] šni",
        "info": {
            "task_type": "identify_pattern",
            "difficulty": "hard",
            "rule_id": "grammar_p17_r1",
            "pattern": "[verb] šni",
            "hints": ["šni"],
            "required_affixes": ["-šni"],
            "special_chars": ["š"],
        },
    }

    records = _prepare_records([entry], include_hints=True)

    assert len(records) == 1
    sample = records[0]
    assert sample["task"] == "identify_pattern"
    assert sample["info"]["difficulty"] == "hard"
    assert sample["info"]["rule_id"] == "grammar_p17_r1"
    assert sample["info"]["verification_pattern"] == "[verb] šni"
    assert sample["info"]["hints"] == ["šni"]
    assert sample["info"]["required_affixes"] == ["-šni"]
    assert sample["info"]["special_chars"] == ["š"]


def test_reward_ledger_is_emitted_for_known_answer() -> None:
    """Scoring a correct answer should populate the reward ledger with component values."""
    env = load_environment(max_examples=2, eval_fraction=0, seed=42)
    sample = env.dataset[0]
    completion = [{"role": "assistant", "content": sample["answer"]}]

    reward = env.rubric.score(completion, sample["answer"], sample["info"])
    ledger = env.get_reward_ledger()

    assert ledger is not None
    assert reward == pytest.approx(ledger["reward_scalar"])
    assert ledger["exact_match_raw"] == pytest.approx(1.0)
    assert "char_overlap_raw" in ledger
    assert "difficulty_multiplier" in ledger


def test_bracketed_grammar_patterns_are_matched_literally() -> None:
    """Bracketed Dakota grammar placeholders should not be lost to regex semantics."""
    env = load_environment(max_examples=2, eval_fraction=0, seed=42)
    sample = env.dataset[0]
    assert sample["info"]["verification_pattern"] == "[verb] šni"

    completion = [{"role": "assistant", "content": sample["answer"]}]
    env.rubric.score(completion, sample["answer"], sample["info"])
    ledger = env.get_reward_ledger()

    assert ledger is not None
    assert ledger["pattern_raw"] == pytest.approx(1.0)
