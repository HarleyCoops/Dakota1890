"""Cheap, deterministic Dakota train reward for Tinker / GRPO.

Scores an extracted final-answer span, not the whole chain of thought.
Difficulty multipliers are logged and must not change ``reward_scalar``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

SPECIAL_CHARS = set("ćšŋḣṡáéíóúķśṅźėčžʼ")

PASS_THRESHOLD = 0.5

# Live Tinker environment weights (published 30B path).
# Length is a multiplier, not an additive term, so the weighted sum maxes at 0.85.
# semantic_accuracy_reward is unused on this path and is not a weight.
WEIGHTS = {
    "exact": 0.4,
    "char": 0.2,
    "pattern": 0.15,
    "affix": 0.1,
}

DIFFICULTY_WEIGHTS = {
    "basic": 1.0,
    "easy": 1.0,
    "intermediate": 1.2,
    "medium": 1.2,
    "advanced": 1.5,
    "hard": 1.5,
    "expert": 2.0,
}

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer\s*(?:is|:))\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def completion_text(completion: Any) -> str:
    """Extract the assistant string from a chat completion or raw text."""
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content") or "")
        if completion:
            last = completion[-1]
            if isinstance(last, dict):
                return str(last.get("content") or "")
            return str(last)
    return str(completion)


def extract_final_answer(text: str) -> str:
    """Return the span that should be scored as the model's answer.

    Priority: last ``\\boxed{...}``, then last ``final answer is/`` line,
    then the last non-empty line, then the stripped text.
    """
    if not text or not str(text).strip():
        return ""
    boxed = list(BOXED_RE.finditer(text))
    if boxed:
        return boxed[-1].group(1).strip()
    finals = list(FINAL_ANSWER_RE.finditer(text))
    if finals:
        return finals[-1].group(1).strip().splitlines()[0].strip()
    lines = [line.strip() for line in str(text).strip().splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) > 1:
        return lines[-1]
    return lines[0]


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _char_f1(prediction: str, target: str) -> float:
    pred_chars = Counter(normalize(prediction).replace(" ", ""))
    target_chars = Counter(normalize(target).replace(" ", ""))
    if not target_chars:
        return 0.0
    overlap = sum(min(pred_chars[char], target_chars[char]) for char in target_chars)
    precision = overlap / max(sum(pred_chars.values()), 1)
    recall = overlap / max(sum(target_chars.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _special_char_f1(span: str, gold: str, expected_chars: list[str] | None) -> float:
    if expected_chars:
        target = Counter(expected_chars)
    else:
        target = Counter(char for char in gold if char in SPECIAL_CHARS)
    if not target:
        return 1.0
    predicted = Counter(char for char in span if char in SPECIAL_CHARS)
    overlap = sum(min(predicted[char], target[char]) for char in target)
    precision = overlap / max(sum(predicted.values()), 1)
    recall = overlap / max(sum(target.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match_score(span: str, gold: str) -> float:
    """Exact normalized match of the extracted span. Buried gold is not 1.0."""
    if not span.strip() or not gold.strip():
        return 0.0
    if normalize(span) == normalize(gold):
        return 1.0
    return 0.0


# Leftover name from the unused 40/40/20 rubric. Not a Tinker train weight.
semantic_score = exact_match_score


def _affix_bearing_tokens(text: str, affix: str) -> list[str]:
    affix_clean = affix.strip("-")
    if not affix_clean:
        return []
    tokens = [re.sub(r"^[^\wŋćšžʼ]+|[^\wŋćšžʼ]+$", "", token, flags=re.UNICODE) for token in text.split()]
    matched: list[str] = []
    if affix.startswith("-") and not affix.endswith("-"):
        pattern = re.compile(rf"(?u)\w+{re.escape(affix_clean)}$", re.IGNORECASE)
        for token in tokens:
            if token and pattern.search(token):
                matched.append(token)
    elif affix.endswith("-") and not affix.startswith("-"):
        pattern = re.compile(rf"(?u)^{re.escape(affix_clean)}\w+", re.IGNORECASE)
        for token in tokens:
            if token and pattern.search(token):
                matched.append(token)
    else:
        for token in tokens:
            if affix_clean.lower() in token.lower():
                matched.append(token)
    return matched


def affix_score(span: str, gold: str, required_affixes: list[str]) -> float:
    """Require the gold's affixed token(s) in the span.

    An empty ``required_affixes`` list is not a free 1.0. Most Dakota JSONL
    rows have no affix metadata; paying 1.0 there was the live Tinker hack.
    """
    if not required_affixes:
        return 0.0
    span_tokens = {normalize(token) for token in span.split() if token.strip()}
    hits = 0
    for affix in required_affixes:
        gold_tokens = {normalize(token) for token in _affix_bearing_tokens(gold, affix)}
        if not gold_tokens:
            continue
        if span_tokens & gold_tokens:
            hits += 1
    return hits / len(required_affixes)


def character_score(span: str, gold: str) -> float:
    if not span.strip() or not gold.strip():
        return 0.0
    return _char_f1(span, gold)


def special_char_score(span: str, gold: str, expected_chars: list[str] | None) -> float:
    """Eval-only specials F1. ``-1.0`` means the gold has no special characters."""
    expected = list(expected_chars or [])
    if not expected and not any(char in SPECIAL_CHARS for char in gold):
        return -1.0
    if not span.strip():
        return 0.0
    return _special_char_f1(span, gold, expected or None)


def pattern_score(span: str, info: dict[str, Any]) -> float:
    """Span-only pattern match. Hints never pay. Missing pattern is 0.0, not 1.0."""
    pattern = (info or {}).get("verification_pattern")
    if not pattern:
        return 0.0
    pattern_text = str(pattern)
    if not pattern_text.strip():
        return 0.0
    literal_candidates = [pattern_text]
    if ":" in pattern_text:
        literal_candidates.append(pattern_text.split(":", 1)[1].strip())
    try:
        if re.search(pattern_text, span, flags=re.IGNORECASE):
            return 1.0
    except re.error:
        pass
    normalized_span = normalize(span)
    if any(
        normalize(candidate) and normalize(candidate) in normalized_span
        for candidate in literal_candidates
    ):
        return 1.0
    return 0.0


def length_penalty(completion: str, gold: str, max_length_ratio: float = 3.0) -> float:
    """Penalize empty outputs and completions much longer than the gold form."""
    pred_len = len(completion.strip())
    gold_len = max(len(gold.strip()), 1)
    if pred_len == 0:
        return 0.0
    ratio = pred_len / gold_len
    if ratio <= max_length_ratio:
        return 1.0
    return max(0.0, max_length_ratio / ratio)


def score_train_reward(
    completion: Any,
    gold: str,
    task_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return unweighted components plus the GRPO scalar (no difficulty)."""
    info = dict(task_info or {})
    raw = completion_text(completion)
    span = extract_final_answer(raw)
    exact = exact_match_score(span, gold)
    char = character_score(span, gold)
    special_char = special_char_score(span, gold, list(info.get("special_chars") or []))
    affix = affix_score(span, gold, list(info.get("required_affixes") or []))
    pattern = pattern_score(span, info)
    length = length_penalty(raw, gold)
    difficulty = str(info.get("difficulty", "intermediate"))
    difficulty_mult = DIFFICULTY_WEIGHTS.get(difficulty.lower(), 1.0)

    contrib_exact = WEIGHTS["exact"] * exact
    contrib_char = WEIGHTS["char"] * char
    contrib_pattern = WEIGHTS["pattern"] * pattern
    contrib_affix = WEIGHTS["affix"] * affix
    composite_pre = contrib_exact + contrib_char + contrib_pattern + contrib_affix
    composite_unweighted = composite_pre * length
    composite_with_difficulty = composite_unweighted * difficulty_mult
    # Competence gate is exact span match. Length still scales the GRPO scalar.
    passed = exact == 1.0

    ledger = {
        "answer_span_len": float(len(span)),
        "exact_match_raw": exact,
        "semantic_raw": exact,
        "char_overlap_raw": char,
        "special_char_raw": special_char,
        "pattern_raw": pattern,
        "affix_raw": affix,
        "length_penalty_raw": length,
        "exact_match_norm": exact,
        "char_overlap_norm": char,
        "pattern_norm": pattern,
        "affix_norm": affix,
        "length_penalty_norm": length,
        "w_exact": WEIGHTS["exact"],
        "w_char": WEIGHTS["char"],
        "w_pattern": WEIGHTS["pattern"],
        "w_affix": WEIGHTS["affix"],
        "w_length": 0.15,
        "difficulty_multiplier": difficulty_mult,
        "contrib_exact": contrib_exact,
        "contrib_char": contrib_char,
        "contrib_pattern": contrib_pattern,
        "contrib_affix": contrib_affix,
        "composite_pre": composite_pre,
        "composite_unweighted": composite_unweighted,
        "composite_with_length": composite_unweighted,
        "composite_with_difficulty": composite_with_difficulty,
        "composite_predicted": composite_unweighted,
        "reward_scalar": composite_unweighted,
        "composite_diff": 0.0,
        "judge_correct": -1.0,
        "judge_morphology_ok": -1.0,
        "judge_meaning_ok": -1.0,
        "judge_orthography_ok": -1.0,
        "passed": float(passed),
    }

    return {
        "answer_span": span,
        "exact_match": exact,
        "semantic": exact,
        "char": char,
        "special_char": special_char,
        "affix": affix,
        "pattern": pattern,
        "length_penalty": length,
        "difficulty_multiplier": difficulty_mult,
        "composite_unweighted": composite_unweighted,
        "composite_with_difficulty": composite_with_difficulty,
        "reward_scalar": composite_unweighted,
        "passed": passed,
        "ledger": ledger,
    }
