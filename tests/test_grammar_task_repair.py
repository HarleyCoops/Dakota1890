"""Repair helpers must drop dead affix labels, placeholders, and English-as-Dakota gold."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dakota_extraction.datasets.grammar_task_repair import (  # noqa: E402
    AttestedLexicon,
    is_placeholder_gold,
    repair_row,
    required_affixes_scorable,
)


WO_RULE_EXAMPLES = [
    {"dakota": "pidá", "english": "glad"},
    {"dakota": "wópida", "english": "gladness"},
    {"dakota": "waóŋšida", "english": "merciful"},
    {"dakota": "wówaoŋšida", "english": "mercy"},
]


def _lexicon() -> AttestedLexicon:
    return AttestedLexicon.from_example_pairs(
        WO_RULE_EXAMPLES
        + [
            {"dakota": "kaška", "english": "to bind"},
            {"dakota": "Dawid suŋkaku", "english": "David's younger brother"},
            {"dakota": "Oglala", "english": "one of the Teton bands"},
        ]
    )


def test_placeholder_gold_is_detected() -> None:
    assert is_placeholder_gold(
        "The incorrect example violates the rule because: [explanation]."
    )
    assert is_placeholder_gold("[TO BE GENERATED: violation of pattern]")
    assert is_placeholder_gold("Step 2: [apply first affix]\nFinal: pidá [with all affixes applied]")
    assert not is_placeholder_gold("wópida")
    assert not is_placeholder_gold("Dawid suŋkaku")


def test_required_affixes_scorable_requires_affix_token_in_gold() -> None:
    assert required_affixes_scorable("wópida", ["wó-"]) is True
    assert required_affixes_scorable("wópida", ["wo-"]) is False
    assert required_affixes_scorable("wópida", ["wo-", "wó-"]) is False
    assert required_affixes_scorable("glad", ["wo-", "wó-"]) is False
    assert required_affixes_scorable("Dawid suŋkaku", ["-ku"]) is True
    assert required_affixes_scorable("wicaštaku", ["-ku"]) is True
    assert required_affixes_scorable("", ["-ku"]) is False
    assert required_affixes_scorable("wópida", []) is False


def test_repair_drops_placeholder_exception_and_evidence_rows() -> None:
    lexicon = _lexicon()
    dropped = repair_row(
        {
            "prompt": "Explain the exception",
            "answer": "These words are exceptions because: [explanation].",
            "info": {"task_type": "exception_trigger", "rule_id": "grammar_p6_r2"},
        },
        lexicon,
    )
    assert dropped is None
    dropped_evidence = repair_row(
        {
            "prompt": "Study this rule",
            "answer": "The incorrect example violates the rule because: [explanation].",
            "info": {"task_type": "positive_negative_evidence", "rule_id": "grammar_p6_r2"},
        },
        lexicon,
    )
    assert dropped_evidence is None


def test_repair_nulls_dead_affix_labels_when_no_attested_form() -> None:
    lexicon = _lexicon()
    repaired = repair_row(
        {
            "prompt": "Analyze Black-boy",
            "answer": "Subject of mourning song published in Dakota Friend",
            "info": {
                "task_type": "morphology",
                "rule_id": "term_p40_Black_boy",
                "base_form": "Black-boy",
                "required_affixes": ["Black-boy"],
            },
        },
        lexicon,
    )
    assert repaired is not None
    assert repaired["info"]["required_affixes"] == []
    assert repaired["answer"] == "Subject of mourning song published in Dakota Friend"


def test_repair_rewrites_english_gold_to_attested_affixed_dakota() -> None:
    lexicon = _lexicon()
    repaired = repair_row(
        {
            "prompt": "Apply wo- to pidá",
            "answer": "glad",
            "info": {
                "task_type": "morphology",
                "rule_id": "grammar_p6_r2",
                "base_form": "pidá",
                "required_affixes": ["wo-", "wó-"],
            },
        },
        lexicon,
        rule_examples={"grammar_p6_r2": WO_RULE_EXAMPLES},
    )
    assert repaired is not None
    assert repaired["answer"] == "wópida"
    assert repaired["info"]["required_affixes"] == ["wó-"]
    assert required_affixes_scorable(repaired["answer"], repaired["info"]["required_affixes"])


def test_repair_keeps_affixes_already_present_in_dakota_gold() -> None:
    lexicon = _lexicon()
    repaired = repair_row(
        {
            "prompt": "Insert wo-",
            "answer": "wópida (gladness)",
            "info": {
                "task_type": "affix_insertion",
                "rule_id": "grammar_p6_r2",
                "target_form": "wópida",
                "required_affixes": ["wo-", "wó-"],
            },
        },
        lexicon,
        rule_examples={"grammar_p6_r2": WO_RULE_EXAMPLES},
    )
    assert repaired is not None
    assert repaired["answer"] == "wópida"
    assert required_affixes_scorable(repaired["answer"], repaired["info"]["required_affixes"])


def test_repair_drops_reverse_translation_without_attested_dakota() -> None:
    lexicon = _lexicon()
    dropped = repair_row(
        {
            "prompt": "Translate this English sentence to Dakota:\n\nA word that points to a noun",
            "answer": "demonstrative",
            "info": {
                "task_type": "reverse_translation",
                "rule_id": "term_p20_demonstrative",
                "english_text": "A word that points to a noun",
            },
        },
        lexicon,
    )
    assert dropped is None


def test_repair_keeps_reverse_translation_with_dakota_gold() -> None:
    lexicon = _lexicon()
    repaired = repair_row(
        {
            "prompt": "Translate this English sentence to Dakota:\n\nto bind",
            "answer": "kaška",
            "info": {
                "task_type": "reverse_translation",
                "rule_id": "term_p21_kaska",
                "english_text": "to bind",
            },
        },
        lexicon,
    )
    assert repaired is not None
    assert repaired["answer"] == "kaška"


def test_repair_does_not_invent_dakota_forms() -> None:
    lexicon = _lexicon()
    repaired = repair_row(
        {
            "prompt": "Apply wo- to an unknown stem",
            "answer": "unknown-english-gloss",
            "info": {
                "task_type": "morphology",
                "rule_id": "grammar_missing_r1",
                "base_form": "zzzznotindata",
                "required_affixes": ["wo-"],
            },
        },
        lexicon,
        rule_examples={"grammar_missing_r1": [{"dakota": "zzzznotindata", "english": "unknown-english-gloss"}]},
    )
    assert repaired is not None
    assert repaired["answer"] == "unknown-english-gloss"
    assert repaired["info"]["required_affixes"] == []
    assert "wowzzzz" not in repaired["answer"]
    assert "wozzzznotindata" not in repaired["answer"]


def test_repair_drops_morphology_whose_source_is_english_place_name() -> None:
    lexicon = _lexicon()
    dropped = repair_row(
        {
            "prompt": "Translate this Dakota sentence to English:\n\nEnd Village",
            "answer": "One of the bands of Ihaŋktoŋwaŋna, estimated at four hundred lodges",
            "info": {
                "task_type": "word_translation",
                "rule_id": "term_p24_End_Village",
                "dakota_text": "End Village",
            },
        },
        lexicon,
    )
    assert dropped is None


def test_repair_replaces_english_reverse_gold_from_paired_example() -> None:
    lexicon = _lexicon()
    repaired = repair_row(
        {
            "prompt": "Translate this English sentence to Dakota:\n\ngladness",
            "answer": "gladness",
            "info": {
                "task_type": "reverse_translation",
                "rule_id": "grammar_p6_r2",
                "english_text": "gladness",
            },
        },
        lexicon,
        rule_examples={"grammar_p6_r2": WO_RULE_EXAMPLES},
    )
    assert repaired is not None
    assert repaired["answer"] == "wópida"
