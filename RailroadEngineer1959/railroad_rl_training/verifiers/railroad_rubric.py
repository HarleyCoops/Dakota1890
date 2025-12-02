import json
from typing import Dict, Any

class RailroadRubric:
    def __init__(self):
        pass

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().strip().split())

    @staticmethod
    def _token_f1(pred: str, ref: str) -> float:
        pred_tokens = RailroadRubric._normalize(pred).split()
        ref_tokens = RailroadRubric._normalize(ref).split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        ref_counts: Dict[str, int] = {}
        for tok in ref_tokens:
            ref_counts[tok] = ref_counts.get(tok, 0) + 1
        common = 0
        for tok in pred_tokens:
            if ref_counts.get(tok, 0) > 0:
                common += 1
                ref_counts[tok] -= 1
        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def evaluate(self, task_description: str, expected_outcome: str, agent_response: str) -> Dict[str, float]:
        """
        Evaluate the agent's response against the task and expected outcome.
        Deterministic scoring: exact match + token F1 (no external API).
        """
        exact = 1.0 if self._normalize(agent_response) == self._normalize(expected_outcome) else 0.0
        f1 = self._token_f1(agent_response, expected_outcome)

        safety = max(exact, f1)
        procedure = f1
        terminology = f1

        return {
            "safety": safety,
            "procedure": procedure,
            "terminology": terminology,
            "reasoning": "Deterministic rubric: exact match + token F1",
            "exact_match": exact,
            "token_f1": f1,
        }
