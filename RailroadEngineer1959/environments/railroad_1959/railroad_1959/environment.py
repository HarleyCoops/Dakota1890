import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import verifiers as vf
from datasets import Dataset
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a railroad safety student. "
    "Respond to the scenario based on the 1959 Consolidated Code of Operating Rules. "
    "Ensure your response is safe, follows correct procedure, and uses precise terminology."
)

class RailroadRubric(Rubric):
    """
    Deterministic rubric for evaluating railroad safety tasks.
    Scores on Safety, Procedure, and Terminology using string similarity.
    """

    def __init__(self):
        super().__init__()
        self._last_ledger: Optional[Dict[str, float]] = None

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

    def score(
        self,
        completion: Messages,
        answer: str,
        info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute reward using LLM evaluation.
        """
        # Extract student response
        # verifiers passes a list of messages, we need the last assistant message
        student_response = ""
        for msg in reversed(completion):
            role = getattr(msg, "role", None) if not isinstance(msg, dict) else msg.get("role")
            content = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
            if role == "assistant" and content:
                student_response = content
                break
        
        if not student_response:
            return 0.0

        expected_outcome = answer

        # Deterministic similarity-based scoring (no external API)
        exact = 1.0 if self._normalize(student_response) == self._normalize(expected_outcome) else 0.0
        f1 = self._token_f1(student_response, expected_outcome)

        safety = max(exact, f1)  # prioritize exact match; otherwise token overlap
        procedure = f1
        terminology = f1

        # Calculate composite reward
        # Weights: Safety 0.5, Procedure 0.3, Terminology 0.2
        reward = (
            0.5 * safety +
            0.3 * procedure +
            0.2 * terminology
        )
        
        self._last_ledger = {
            "safety": safety,
            "procedure": procedure,
            "terminology": terminology,
            "exact_match": exact,
            "token_f1": f1,
            "reasoning": "Deterministic rubric: exact match + token F1",
            "reward_scalar": reward
        }
        
        return reward

    def get_last_ledger(self) -> Optional[Dict[str, float]]:
        return self._last_ledger


class RailroadEnv(SingleTurnEnv):
    def __init__(
        self,
        dataset: Dataset,
        system_prompt: str,
        rubric: RailroadRubric,
        **kwargs: Any,
    ):
        super().__init__(
            dataset=dataset,
            system_prompt=system_prompt,
            rubric=rubric,
            message_type="chat",
            **kwargs,
        )
        self.rubric = rubric

    def get_reward_ledger(self) -> Optional[Dict[str, float]]:
        return self.rubric.get_last_ledger()


def load_environment(
    dataset_path: str | Path | None = None,
    system_prompt: Optional[str] = None,
    max_examples: int = -1,
) -> vf.Environment:
    """
    Load the Railroad 1959 environment.
    """
    default_path = Path(__file__).resolve().parent / "data" / "safety_tasks_complete.json"
    download_url = (
        "https://raw.githubusercontent.com/HarleyCooper/Dakota1890/main/"
        "RailroadEngineer1959/data/railroad_extracted/safety_tasks_complete.json"
    )

    if dataset_path is None:
        dataset_path = default_path
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        logger.warning("Dataset not found at %s; attempting download from %s", dataset_path, download_url)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(download_url, timeout=60)
        resp.raise_for_status()
        dataset_path.write_text(resp.text, encoding="utf-8")
        logger.info("Downloaded dataset to %s", dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to list of dicts with 'question' and 'answer'
    records = []
    for item in data:
        records.append({
            "question": f"Task ID: {item.get('task_id')}\nScenario: {item.get('description')}",
            "answer": item.get("expected_outcome", ""),
            "info": {
                "task_id": item.get("task_id"),
                "description": item.get("description"),
                "applicable_rules": item.get("applicable_rules")
            }
        })

    if max_examples > 0:
        records = records[:max_examples]

    dataset = Dataset.from_list(records)
    rubric = RailroadRubric()
    
    return RailroadEnv(
        dataset=dataset,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        rubric=rubric
    )
