"""
Reward Functions (Rubrics) for Dakota Grammar RL Training

Delegates to the grant-clean train reward used by the Tinker environment:
span extraction, gold-token affix checks, restored length penalty, and
unweighted ledger components. Historical heuristics live in
``dakota_grammar_translation.legacy_reward``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any

from .base import Rubric

_ENV_PKG = Path(__file__).resolve().parents[2] / "environments" / "dakota_grammar_translation"
if str(_ENV_PKG) not in sys.path:
    sys.path.insert(0, str(_ENV_PKG))

from dakota_grammar_translation.train_reward import (  # noqa: E402
    affix_score,
    character_score,
    extract_final_answer,
    length_penalty as train_length_penalty,
    score_train_reward,
    semantic_score,
    special_char_score,
)


class DakotaGrammarRubric(Rubric):
    """Reward functions for Dakota grammar tasks"""

    def __init__(self):
        super().__init__()
        self.special_chars = set("ćšŋḣṡáéíóúķśṅźėčžʼ")
        self._last_ledger: Dict[str, float] | None = None

        # Logged only. Not applied to the GRPO scalar.
        self.difficulty_weights = {
            "basic": 1.0,
            "intermediate": 1.2,
            "advanced": 1.5,
            "expert": 2.0
        }

    def character_preservation_reward(
        self,
        response: str,
        expected_chars: List[str],
        **kwargs
    ) -> float:
        span = extract_final_answer(response)
        gold = str(kwargs.get("expected") or kwargs.get("gold") or "")
        if gold:
            return character_score(span, gold)
        return special_char_score(span, "".join(expected_chars or []), list(expected_chars or []))

    def affix_accuracy_reward(
        self,
        response: str,
        required_affixes: List[str],
        **kwargs
    ) -> float:
        span = extract_final_answer(response)
        gold = str(kwargs.get("expected") or kwargs.get("gold") or "")
        return affix_score(span, gold, list(required_affixes or []))

    def semantic_accuracy_reward(
        self,
        response: str,
        expected: str,
        task_type: str = "morphology",
        **kwargs
    ) -> float:
        return semantic_score(extract_final_answer(response), expected)

    def length_penalty(
        self,
        response: str,
        expected: str,
        max_length_ratio: float = 3.0,
        **kwargs
    ) -> float:
        return train_length_penalty(response, expected, max_length_ratio=max_length_ratio)

    def composite_reward(
        self,
        response: str,
        expected: str,
        task_info: Dict[str, Any],
        **kwargs
    ) -> float:
        result = score_train_reward(response, expected, task_info)
        self._last_ledger = result["ledger"]
        return float(result["reward_scalar"])

    def binary_reward(
        self,
        response: str,
        expected: str,
        task_info: Dict[str, Any],
        threshold: float = 0.95,
        **kwargs
    ) -> float:
        """
        Binary reward (1.0 or 0.0) based on threshold

        Useful for strict learning
        """
        composite = self.composite_reward(response, expected, task_info)

        return 1.0 if composite >= threshold else 0.0

    def progressive_reward(
        self,
        messages: List[Dict],
        state: Dict,
        task_info: Dict[str, Any],
        **kwargs
    ) -> float:
        """
        Reward improvement across turns (for multi-turn env)

        Encourages learning from feedback
        """
        if not messages or len(messages) < 2:
            return 0.0

        # Get last two responses
        current_response = messages[-1]["content"]
        expected = kwargs.get("answer", "")

        # Current accuracy
        current_score = self.composite_reward(
            current_response,
            expected,
            task_info
        )

        # Previous accuracy (if exists)
        if len(messages) >= 3:
            previous_response = messages[-3]["content"]
            previous_score = self.composite_reward(
                previous_response,
                expected,
                task_info
            )

            # Reward improvement
            improvement = current_score - previous_score
            return max(0.0, improvement)  # Only reward positive improvement

        return current_score

    def curriculum_bonus(
        self,
        response: str,
        expected: str,
        task_info: Dict[str, Any],
        student_level: str = "basic",
        **kwargs
    ) -> float:
        """
        Bonus reward for attempting harder tasks

        Encourages curriculum progression
        """
        task_difficulty = task_info.get("difficulty", "intermediate")
        base_reward = self.composite_reward(response, expected, task_info)

        # Student levels
        level_order = ["basic", "intermediate", "advanced", "expert"]
        student_idx = level_order.index(student_level) if student_level in level_order else 0
        task_idx = level_order.index(task_difficulty) if task_difficulty in level_order else 1

        # Bonus for attempting harder tasks
        if task_idx > student_idx:
            bonus = 0.1 * (task_idx - student_idx)
            return base_reward + bonus

        return base_reward

    def score(self, response: str, expected: str, **kwargs: Any) -> float:
        """
        Implementation of the abstract Rubric.score interface.

        Kwargs may include a `task_info` dictionary, which defaults to an empty
        mapping when not provided.
        """
        task_info = kwargs.get("task_info", {})
        return self.composite_reward(response, expected, task_info, **kwargs)

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Compute Levenshtein distance"""
        if len(s1) < len(s2):
            return DakotaGrammarRubric._levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


# Metrics for evaluation (not rewards, just tracking)
class DakotaMetrics:
    """Track learning metrics"""

    @staticmethod
    def char_accuracy_by_type(responses: List[Dict]) -> Dict[str, float]:
        """Track which special chars are hardest to learn"""
        char_counts = {}
        char_correct = {}

        for resp in responses:
            expected_chars = resp.get("expected_chars", [])
            response_text = resp.get("response", "")

            for char in expected_chars:
                char_counts[char] = char_counts.get(char, 0) + 1
                if char in response_text:
                    char_correct[char] = char_correct.get(char, 0) + 1

        accuracies = {}
        for char in char_counts:
            accuracies[char] = char_correct.get(char, 0) / char_counts[char]

        return accuracies

    @staticmethod
    def affix_accuracy_by_type(responses: List[Dict]) -> Dict[str, float]:
        """Track which affixes are hardest to learn"""
        affix_counts = {}
        affix_correct = {}

        for resp in responses:
            required_affixes = resp.get("required_affixes", [])
            response_text = resp.get("response", "")

            for affix in required_affixes:
                affix_counts[affix] = affix_counts.get(affix, 0) + 1
                if affix.strip("-") in response_text:
                    affix_correct[affix] = affix_correct.get(affix, 0) + 1

        accuracies = {}
        for affix in affix_counts:
            accuracies[affix] = affix_correct.get(affix, 0) / affix_counts[affix]

        return accuracies


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Test rubric
    rubric = DakotaGrammarRubric()

    # Test case from actual extraction
    task_info = {
        "task_type": "morphology",
        "base_form": "suŋka",
        "required_affixes": ["-ku"],
        "special_chars": ["ŋ"],
        "difficulty": "advanced"
    }

    response = "Dawid suŋkaku"
    expected = "Dawid suŋkaku"

    print("Test Case: Kinship suffix -ku")
    print(f"Response: {response}")
    print(f"Expected: {expected}")
    print()

    # Test individual rewards
    char_reward = rubric.character_preservation_reward(
        response, task_info["special_chars"]
    )
    print(f"Character reward: {char_reward:.2f}")

    affix_reward = rubric.affix_accuracy_reward(
        response, task_info["required_affixes"]
    )
    print(f"Affix reward: {affix_reward:.2f}")

    semantic_reward = rubric.semantic_accuracy_reward(
        response, expected, task_info["task_type"]
    )
    print(f"Semantic reward: {semantic_reward:.2f}")

    # Composite reward
    composite = rubric.composite_reward(response, expected, task_info)
    print(f"Composite reward: {composite:.2f}")

    # Test wrong answer (missing ŋ)
    print("\n" + "="*60)
    print("Test Case: Wrong answer (missing ŋ)")
    wrong_response = "Dawid sunkaku"
    print(f"Response: {wrong_response}")
    print(f"Expected: {expected}")

    char_reward_wrong = rubric.character_preservation_reward(
        wrong_response, task_info["special_chars"]
    )
    print(f"Character reward: {char_reward_wrong:.2f}")

    composite_wrong = rubric.composite_reward(wrong_response, expected, task_info)
    print(f"Composite reward: {composite_wrong:.2f}")
    
    # Test Verbose Chain of Thought (The Fix)
    print("\n" + "="*60)
    print("Test Case: Verbose Reasoning (CoT)")
    cot_response = "To form the kinship term for 'younger brother' using the base 'suŋka' and the suffix '-ku', we first note that 'suŋka' ends in a vowel. In Dakota morphology, '-ku' is added directly to kinship terms ending in 'a'. Therefore, we combine 'suŋka' + 'ku' to get 'suŋkaku'. The final answer is Dawid suŋkaku."
    
    print(f"Response: {cot_response}")
    print(f"Expected: {expected}")
    
    char_reward_cot = rubric.character_preservation_reward(
        cot_response, task_info["special_chars"]
    )
    print(f"Character reward: {char_reward_cot:.2f} (Should be 1.0)")
    
    affix_reward_cot = rubric.affix_accuracy_reward(
        cot_response, task_info["required_affixes"]
    )
    print(f"Affix reward: {affix_reward_cot:.2f} (Should be 1.0)")
    
    semantic_reward_cot = rubric.semantic_accuracy_reward(
        cot_response, expected, task_info["task_type"]
    )
    print(f"Semantic reward: {semantic_reward_cot:.2f} (Should be 1.0)")
    
    composite_cot = rubric.composite_reward(cot_response, expected, task_info)
    print(f"Composite reward: {composite_cot:.2f} (Should be ~1.2 with difficulty mult)")
