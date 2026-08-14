"""Historical Dakota reward heuristics that the published Tinker diagnosis targeted.

These functions reproduce ``dakota_rl_training/verifiers/rubrics.py`` as it
existed before the grant-clean rewrite:

- semantic 1.0 if the gold string is a substring of the full response
- affix 1.0 if the affix appears on *any* word
- character score is special-character *recall* (sprinkle-once)
- length penalty hardwired to 1.0
- difficulty multipliers inflate the composite

The Thinking Machines loop actually scored
``environments/.../environment.py`` (exact match on the full completion, char
F1, same affix-anywhere check, length=1.0, difficulty on the scalar). Both
paths paid affix-anywhere and disabled length; only this legacy copy paid
gold-substring semantics. Tests use this module to show which probes the old
train signal accepted.
"""

from __future__ import annotations

import re
from typing import Any

SPECIAL_CHARS = set("ćšŋḣṡáéíóúķśṅźėčžʼ")

DIFFICULTY_WEIGHTS = {
    "basic": 1.0,
    "easy": 1.0,
    "intermediate": 1.2,
    "medium": 1.2,
    "advanced": 1.5,
    "hard": 1.5,
    "expert": 2.0,
}


def legacy_character_preservation(response: str, expected_chars: list[str]) -> float:
    if not expected_chars:
        return 1.0
    response_chars = {char for char in response if char in SPECIAL_CHARS}
    expected_set = set(expected_chars)
    if not expected_set:
        return 1.0
    return len(response_chars & expected_set) / len(expected_set)


def legacy_affix_accuracy(response: str, required_affixes: list[str]) -> float:
    if not required_affixes:
        return 1.0
    correct_count = 0
    for affix in required_affixes:
        affix_clean = affix.strip("-")
        if affix.startswith("-") and not affix.endswith("-"):
            if re.search(rf"\w+{re.escape(affix_clean)}\b", response):
                correct_count += 1
        elif affix.endswith("-") and not affix.startswith("-"):
            if re.search(rf"\b{re.escape(affix_clean)}\w+", response):
                correct_count += 1
        elif affix_clean in response:
            correct_count += 1
    return correct_count / len(required_affixes)


def legacy_semantic_accuracy(response: str, expected: str, task_type: str = "morphology") -> float:
    response_norm = response.strip().lower()
    expected_norm = expected.strip().lower()
    if expected_norm and expected_norm in response_norm:
        return 1.0
    if task_type in {"word_translation", "sentence_translation"}:
        response_words = set(response_norm.split())
        expected_words = set(expected_norm.split())
        if not expected_words:
            return 0.0
        return len(response_words & expected_words) / len(expected_words)
    return 0.0


def legacy_length_penalty(response: str, expected: str) -> float:
    return 1.0


def legacy_composite(response: str, expected: str, task_info: dict[str, Any]) -> dict[str, float]:
    task_type = str(task_info.get("task_type", "morphology"))
    difficulty = str(task_info.get("difficulty", "intermediate"))
    char_reward = legacy_character_preservation(response, list(task_info.get("special_chars") or []))
    affix_reward = legacy_affix_accuracy(response, list(task_info.get("required_affixes") or []))
    semantic_reward = legacy_semantic_accuracy(response, expected, task_type)
    if task_type == "morphology":
        weights = {"char": 0.4, "affix": 0.4, "semantic": 0.2}
    elif task_type in {"word_translation", "sentence_translation"}:
        weights = {"char": 0.3, "affix": 0.0, "semantic": 0.7}
    elif task_type == "reverse_translation":
        weights = {"char": 0.5, "affix": 0.0, "semantic": 0.5}
    else:
        weights = {"char": 0.33, "affix": 0.33, "semantic": 0.34}
    base = (
        weights["char"] * char_reward
        + weights["affix"] * affix_reward
        + weights["semantic"] * semantic_reward
    )
    length_mult = legacy_length_penalty(response, expected)
    difficulty_mult = DIFFICULTY_WEIGHTS.get(difficulty.lower(), 1.0)
    reward = base * length_mult * difficulty_mult
    return {
        "semantic": semantic_reward,
        "affix": affix_reward,
        "char": char_reward,
        "length_penalty": length_mult,
        "difficulty_multiplier": difficulty_mult,
        "reward_scalar": reward,
    }
