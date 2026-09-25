"""Grant-clean Dakota train reward: hack probes must fail; exact gold must pass.

The published Tinker loop scores via
``environments/dakota_grammar_translation/.../environment.py``, not
``dakota_rl_training/verifiers/rubrics.py``. Both copies of the old heuristic
are reproduced here as ``legacy_*`` so the tests document which hacks the
historical reward actually paid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV_PKG = ROOT / "environments" / "dakota_grammar_translation"
if str(ENV_PKG) not in sys.path:
    sys.path.insert(0, str(ENV_PKG))

from dakota_grammar_translation.legacy_reward import (  # noqa: E402
    legacy_affix_accuracy,
    legacy_character_preservation,
    legacy_composite,
    legacy_semantic_accuracy,
)
from dakota_grammar_translation.train_reward import (  # noqa: E402
    PASS_THRESHOLD,
    WEIGHTS,
    exact_match_score,
    extract_final_answer,
    pattern_score,
    score_train_reward,
)
from dakota_grammar_translation.prompts import strip_leaked_supervision  # noqa: E402


KINSHIP_TASK = {
    "prompt": (
        "Apply the Dakota kinship suffix -ku to suŋka (younger brother) "
        "as in Riggs 1890."
    ),
    "answer": "Dawid suŋkaku",
    "info": {
        "task_type": "morphology",
        "base_form": "suŋka",
        "required_affixes": ["-ku"],
        "special_chars": ["ŋ"],
        "difficulty": "advanced",
        "verification_pattern": None,
        "hints": [],
    },
}

GOLD = KINSHIP_TASK["answer"]
INFO = KINSHIP_TASK["info"]

GOLD_STUFFED_COT = (
    "I remember the gold form Dawid suŋkaku from the dictionary entry. "
    "The suffix -ku marks a kinship genitive and ŋ must be written. "
    "After considering several stems I will not commit to a surface form."
)
CHAR_SPRINKLE = (
    "Dakota uses special letters. Here is one of them: ŋ. "
    "That should be enough to show orthographic awareness."
)
AFFIX_WRONG_STEM = "wicaštaku"
EMPTY = ""
EXACT_GOLD = "Dawid suŋkaku"
BOXED_GOLD = "Reasoning about kinship.\n\\boxed{Dawid suŋkaku}"
FINAL_ANSWER_GOLD = "Let me think about the suffix.\nFinal answer is Dawid suŋkaku"


def _new(response: str) -> dict:
    return score_train_reward(response, GOLD, INFO)


def _legacy(response: str) -> dict:
    return legacy_composite(response, GOLD, INFO)


def test_extract_final_answer_prefers_boxed_then_final_then_last_line() -> None:
    assert extract_final_answer("foo\n\\boxed{Dawid suŋkaku}\n") == "Dawid suŋkaku"
    assert extract_final_answer(FINAL_ANSWER_GOLD) == "Dawid suŋkaku"
    assert extract_final_answer("line one\nline two") == "line two"
    assert extract_final_answer("Dawid suŋkaku") == "Dawid suŋkaku"
    assert extract_final_answer("") == ""


def test_extract_final_answer_empty_marker_does_not_indexerror() -> None:
    """Whitespace-only FINAL_ANSWER_RE captures must not crash scoring.

    Grant-clean v3b (W&B h67qxtne) died when the last match was
    ``Final answer is`` / ``Final answer:`` followed only by whitespace:
    ``group(1).strip()`` was ``""`` and ``splitlines()[0]`` raised
    ``IndexError``. Walk earlier matches, then the last non-empty line.
    """
    no_span = extract_final_answer("reasoning about the suffix\nFinal answer is")
    assert no_span == "Final answer is"

    whitespace_only = extract_final_answer("reasoning about the suffix\nFinal answer:   ")
    assert whitespace_only == "Final answer:"

    # ``(?:is|:)`` is exclusive, so "is:" leaves the colon in the capture.
    normal = extract_final_answer("reasoning about the suffix\nFinal answer is: foo")
    assert normal == ": foo"

    earlier = extract_final_answer("Final answer is Dawid suŋkaku\nFinal answer is   ")
    assert earlier == "Dawid suŋkaku"

    scored = score_train_reward("Final answer:   \n", GOLD, INFO)
    assert scored["answer_span"] == "Final answer:"
    assert scored["passed"] is False


def test_legacy_semantic_pays_gold_substring_in_fluff() -> None:
    assert legacy_semantic_accuracy(GOLD_STUFFED_COT, GOLD) == pytest.approx(1.0)


def test_legacy_affix_pays_suffix_on_wrong_stem() -> None:
    assert legacy_affix_accuracy(AFFIX_WRONG_STEM, ["-ku"]) == pytest.approx(1.0)


def test_legacy_character_pays_sprinkled_eng() -> None:
    assert legacy_character_preservation(CHAR_SPRINKLE, ["ŋ"]) == pytest.approx(1.0)


def test_empty_required_affixes_do_not_score_one() -> None:
    info = dict(INFO)
    info["required_affixes"] = []
    result = score_train_reward(EXACT_GOLD, GOLD, info)
    assert result["affix"] == pytest.approx(0.0)
    assert result["exact_match"] == pytest.approx(1.0)
    assert result["passed"] is True


def test_hint_echo_does_not_pay_pattern() -> None:
    info = dict(INFO)
    info["verification_pattern"] = None
    info["hints"] = ["suŋkaku", "ŋ", "Dawid"]
    echoed = "suŋkaku ŋ Dawid"
    result = score_train_reward(echoed, GOLD, info)
    assert pattern_score(echoed, info) == pytest.approx(0.0)
    assert result["pattern"] == pytest.approx(0.0)
    assert result["exact_match"] == pytest.approx(0.0)


def test_gold_in_cot_is_not_exact_match() -> None:
    result = _new(GOLD_STUFFED_COT)
    assert result["answer_span"] != GOLD
    assert GOLD.lower() in GOLD_STUFFED_COT.lower()
    assert result["exact_match"] == pytest.approx(0.0)
    assert exact_match_score(result["answer_span"], GOLD) == pytest.approx(0.0)


def test_train_scalar_uses_live_tinker_weights_not_semantic_rubric() -> None:
    assert WEIGHTS == {"exact": 0.4, "char": 0.2, "pattern": 0.15, "affix": 0.1}
    assert "semantic" not in WEIGHTS


def test_strip_leaked_pattern_examples_from_prompt() -> None:
    prompt = (
        "Identify the grammatical pattern in this Dakota rule:\n\n"
        "Negation with šni\n\n"
        "Examples:\n  - [verb] šni\n"
    )
    cleaned = strip_leaked_supervision(prompt, gold="[verb] šni", pattern="[verb] šni")
    assert "[verb] šni" not in cleaned
    assert "Negation with šni" in cleaned


def test_new_semantic_does_not_pay_buried_gold() -> None:
    result = _new(GOLD_STUFFED_COT)
    assert result["answer_span"] != GOLD
    assert GOLD.lower() in GOLD_STUFFED_COT.lower()
    assert result["semantic"] < 1.0
    assert result["semantic"] == pytest.approx(0.0)


def test_new_affix_does_not_pay_wrong_stem() -> None:
    result = _new(AFFIX_WRONG_STEM)
    assert result["affix"] == pytest.approx(0.0)


def test_new_character_does_not_pay_sprinkle() -> None:
    result = _new(CHAR_SPRINKLE)
    assert result["char"] < 0.5
    assert result["special_char"] < 1.0 or result["char"] < 1.0


def test_hack_probes_old_reward_passes_gold_stuff_and_wrong_stem() -> None:
    stuffed = _legacy(GOLD_STUFFED_COT)
    wrong_stem = _legacy(AFFIX_WRONG_STEM)
    assert stuffed["semantic"] == pytest.approx(1.0)
    assert stuffed["reward_scalar"] >= PASS_THRESHOLD
    assert wrong_stem["affix"] == pytest.approx(1.0)
    assert wrong_stem["reward_scalar"] >= PASS_THRESHOLD


def test_hack_probes_new_reward_fails_attacks_and_passes_exact_gold() -> None:
    probes = {
        "gold_stuffed_cot": GOLD_STUFFED_COT,
        "char_sprinkle": CHAR_SPRINKLE,
        "affix_wrong_stem": AFFIX_WRONG_STEM,
        "empty": EMPTY,
        "exact_gold": EXACT_GOLD,
    }
    for name, response in probes.items():
        result = _new(response)
        if name == "exact_gold":
            assert result["exact_match"] == pytest.approx(1.0)
            assert result["affix"] == pytest.approx(1.0)
            assert result["char"] == pytest.approx(1.0)
            assert result["reward_scalar"] >= PASS_THRESHOLD
            assert result["passed"] is True
        else:
            assert result["reward_scalar"] < PASS_THRESHOLD, name
            assert result["passed"] is False, name


def test_legitimate_final_answer_span_still_scores() -> None:
    for response in (BOXED_GOLD, FINAL_ANSWER_GOLD):
        result = _new(response)
        assert result["answer_span"] == GOLD
        assert result["exact_match"] == pytest.approx(1.0)
        assert result["passed"] is True


def test_difficulty_does_not_inflate_train_reward() -> None:
    easy_info = dict(INFO)
    easy_info["difficulty"] = "basic"
    hard = _new(EXACT_GOLD)
    easy = score_train_reward(EXACT_GOLD, GOLD, easy_info)
    assert hard["difficulty_multiplier"] == pytest.approx(1.5)
    assert easy["difficulty_multiplier"] == pytest.approx(1.0)
    assert hard["reward_scalar"] == pytest.approx(easy["reward_scalar"])
    assert hard["composite_unweighted"] == pytest.approx(easy["composite_unweighted"])
    assert hard["composite_with_difficulty"] > easy["composite_with_difficulty"]


def test_ledger_reports_unweighted_components() -> None:
    result = _new(EXACT_GOLD)
    ledger = result["ledger"]
    for key in (
        "semantic_raw",
        "char_overlap_raw",
        "affix_raw",
        "special_char_raw",
        "length_penalty_raw",
        "composite_unweighted",
        "difficulty_multiplier",
        "composite_with_difficulty",
        "reward_scalar",
        "judge_correct",
    ):
        assert key in ledger, key
    assert ledger["reward_scalar"] == pytest.approx(ledger["composite_unweighted"])
    assert ledger["reward_scalar"] != pytest.approx(ledger["composite_with_difficulty"])
    assert ledger["judge_correct"] == pytest.approx(-1.0)


def test_length_penalty_punishes_gold_stuffing_even_with_trailing_gold() -> None:
    stuffed_with_trailing_gold = GOLD_STUFFED_COT + "\nDawid suŋkaku"
    result = _new(stuffed_with_trailing_gold)
    exact = _new(EXACT_GOLD)
    assert result["answer_span"] == GOLD
    assert result["semantic"] == pytest.approx(1.0)
    assert result["length_penalty"] < 1.0
    assert result["reward_scalar"] < exact["reward_scalar"]


def test_empty_completion_is_zero() -> None:
    result = _new(EMPTY)
    assert result["semantic"] == pytest.approx(0.0)
    assert result["length_penalty"] == pytest.approx(0.0)
    assert result["reward_scalar"] == pytest.approx(0.0)


def test_hack_probe_file_matches_expected_outcomes() -> None:
    probe_path = ROOT / "dakota_rl_training" / "datasets" / "hack_probes.jsonl"
    assert probe_path.exists()
    rows = [json.loads(line) for line in probe_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    names = {row["probe_id"] for row in rows}
    assert names >= {
        "gold_stuffed_cot",
        "char_sprinkle",
        "affix_wrong_stem",
        "empty",
        "exact_gold",
    }
    for row in rows:
        scored = score_train_reward(row["model_output"], row["answer"], row["info"])
        if row.get("expect_train_pass"):
            assert scored["passed"] is True, row["probe_id"]
        else:
            assert scored["passed"] is False, row["probe_id"]
            legacy = legacy_composite(row["model_output"], row["answer"], row["info"])
            if row["probe_id"] in {"gold_stuffed_cot", "affix_wrong_stem"}:
                assert legacy["reward_scalar"] >= PASS_THRESHOLD, row["probe_id"]


def test_eval_heldout_scores_probes_without_judge() -> None:
    sys.path.insert(0, str(ROOT / "dakota_rl_training"))
    from eval_heldout import score_row, summarize

    probe_path = ROOT / "dakota_rl_training" / "datasets" / "hack_probes.jsonl"
    rows = [json.loads(line) for line in probe_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scored = [score_row(row) for row in rows]
    by_id = {row["id"]: row for row in scored}
    assert by_id["exact_gold"]["passed"] is True
    assert by_id["gold_stuffed_cot"]["passed"] is False
    assert by_id["affix_wrong_stem"]["passed"] is False
    assert by_id["exact_gold"]["judge_correct"] is None
    summary = summarize(scored)
    assert summary["n"] == len(rows)
    assert "mean_semantic" in summary
