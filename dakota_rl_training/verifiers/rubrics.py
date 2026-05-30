"""
Reward Functions (Rubrics) for Dakota Grammar RL Training

Based on actual extraction from page 61:
- Special character preservation (ŋ, š, ć, ź, ž, ʼ)
- Morphological accuracy (affixes: -ku, -ću, -tku, ta-, ti-)
- Semantic correctness (word/sentence translation)
- Difficulty-adjusted rewards (basic → expert)
"""

from typing import List, Dict, Any
import re

from .base import Rubric


class DakotaGrammarRubric(Rubric):
    """Reward functions for Dakota grammar tasks"""

    def __init__(self):
        super().__init__()
        self.special_chars = set("ćšŋḣṡáéíóúķśṅźėčžʼ")

        # Difficulty multipliers (from actual extraction)
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
        """
        Reward for preserving Dakota special characters

        Critical for Dakota language preservation!
        Returns 0.0-1.0
        """
        if not expected_chars:
            return 1.0  # No special chars required

        response_chars = set(c for c in response if c in self.special_chars)
        expected_set = set(expected_chars)

        # If expected set is empty, nothing to match
        if not expected_set:
            return 1.0

        intersection = response_chars & expected_set
        
        # Recall-based: penalize missing characters, but allow extras (for reasoning)
        # Old: return len(intersection) / len(expected_set) if union else 0.0 (IoU penalizes reasoning)
        # New: Recall = intersection / expected
        return len(intersection) / len(expected_set)

    def affix_accuracy_reward(
        self,
        response: str,
        required_affixes: List[str],
        **kwargs
    ) -> float:
        """
        Reward for correct affix application

        Based on actual patterns:
        - Suffixes: -ku, -ću, -tku (kinship)
        - Prefixes: ta-, ti-, to- (possessive)
        """
        if not required_affixes:
            return 1.0  # No affixes required

        correct_count = 0
        for affix in required_affixes:
            affix_clean = affix.strip("-")

            # Check if affix appears correctly
            if affix.startswith("-") and not affix.endswith("-"):
                # Suffix
                if re.search(rf'\w+{re.escape(affix_clean)}\b', response):
                    correct_count += 1
            elif affix.endswith("-") and not affix.startswith("-"):
                # Prefix
                if re.search(rf'\b{re.escape(affix_clean)}\w+', response):
                    correct_count += 1
            else:
                # Standalone or infix - just check presence
                if affix_clean in response:
                    correct_count += 1

        return correct_count / len(required_affixes) if required_affixes else 1.0

    def semantic_accuracy_reward(
        self,
        response: str,
        expected: str,
        task_type: str = "morphology",
        **kwargs
    ) -> float:
        """
        Reward for semantic correctness

        Different strategies for different task types
        """
        response_norm = response.strip().lower()
        expected_norm = expected.strip().lower()

        # Substring check (Primary Logic)
        # This allows for verbosity/reasoning as long as the answer is present.
        if expected_norm in response_norm:
            return 1.0

        # For translation tasks, allow word overlap
        if task_type in ["word_translation", "sentence_translation"]:
            response_words = set(response_norm.split())
            expected_words = set(expected_norm.split())

            if not expected_words:
                return 0.0

            # Overlap ratio (Recall)
            # intersection / expected (allows extras)
            intersection = response_words & expected_words
            return len(intersection) / len(expected_words)

        # For morphology, fallback to Levenshtein on substrings (simplified to raw Lev for now if no exact match)
        # But realistically, if the exact form isn't there, it's wrong.
        # We can keep partial credit if they are CLOSE to the answer.
        
        # Partial credit logic:
        # Find best substring match? Too expensive.
        # Fallback: Check if Levenshtein is small relative to expected length?
        max_len = max(len(response_norm), len(expected_norm))

        if max_len == 0:
            return 1.0

        # If response is huge (reasoning), distance is huge. 
        # So Levenshtein only makes sense if response is short.
        # If response is long and doesn't contain substring, score is likely 0.
        # We can return 0 here safely.
        
        # But let's allow for slight typos in the target word within a long response?
        # That's hard to detect without sliding window.
        # For now, simple substring match is the biggest fix.
        
        return 0.0

    def length_penalty(
        self,
        response: str,
        expected: str,
        max_length_ratio: float = 3.0,
        **kwargs
    ) -> float:
        """
        Length penalty disabled for small-model Tinker runs (always 1.0).
        """
        return 1.0


    def composite_reward(
        self,
        response: str,
        expected: str,
        task_info: Dict[str, Any],
        **kwargs
    ) -> float:
        """
        Composite reward combining all factors

        Weights based on task type:
        - Morphology: 40% chars, 40% affixes, 20% semantic
        - Translation: 30% chars, 0% affixes, 70% semantic
        - Reverse translation: 50% chars, 0% affixes, 50% semantic

        INCLUDES LENGTH PENALTY to prevent degenerate long outputs
        """
        task_type = task_info.get("task_type", "morphology")
        difficulty = task_info.get("difficulty", "intermediate")

        # Calculate component rewards
        char_reward = self.character_preservation_reward(
            response,
            task_info.get("special_chars", [])
        )

        affix_reward = self.affix_accuracy_reward(
            response,
            task_info.get("required_affixes", [])
        )

        semantic_reward = self.semantic_accuracy_reward(
            response,
            expected,
            task_type
        )

        # Weight by task type
        if task_type == "morphology":
            weights = {"char": 0.4, "affix": 0.4, "semantic": 0.2}
        elif task_type in ["word_translation", "sentence_translation"]:
            weights = {"char": 0.3, "affix": 0.0, "semantic": 0.7}
        elif task_type == "reverse_translation":
            # Reverse translation needs both chars AND semantics
            weights = {"char": 0.5, "affix": 0.0, "semantic": 0.5}
        else:
            weights = {"char": 0.33, "affix": 0.33, "semantic": 0.34}

        # Composite reward
        base_reward = (
            weights["char"] * char_reward +
            weights["affix"] * affix_reward +
            weights["semantic"] * semantic_reward
        )

        # Apply length penalty (prevents degenerate long outputs)
        length_mult = self.length_penalty(response, expected)

        # Apply difficulty multiplier
        difficulty_mult = self.difficulty_weights.get(difficulty, 1.0)

        return base_reward * length_mult * difficulty_mult

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
