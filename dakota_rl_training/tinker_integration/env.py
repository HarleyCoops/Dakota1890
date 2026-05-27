from __future__ import annotations

import logging
from typing import Any, Dict

import tinker
from dakota_grammar_translation.environment import DakotaGrammarRubric, DEFAULT_SYSTEM_PROMPT
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import Action, ActionExtra, Env, Observation, StepResult

from .types import DakotaGrammarExample

logger = logging.getLogger(__name__)


class DakotaTinkerEnv(Env):
    """Tinker-compatible environment that reuses the Dakota grammar rubric."""

    def __init__(
        self,
        example: DakotaGrammarExample,
        renderer: renderers.Renderer,
        system_prompt: str | None = None,
    ):
        self.example = example
        self.renderer = renderer
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._rubric = DakotaGrammarRubric()
        self._base_messages = self._build_base_messages()
        self._stop_condition: StopCondition = renderer.get_stop_sequences()

    def _build_base_messages(self) -> list[Dict[str, str]]:
        messages: list[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.example.prompt})
        return messages

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        prompt = self.renderer.build_generation_prompt(self._base_messages)
        return prompt, self._stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, termination = self.renderer.parse_response(action)

        completion: list[Dict[str, Any]] = [*self._base_messages, message]
        reward = float(
            self._rubric.score(
                completion,
                self.example.answer,
                info=self.example.info,
            )
        )

        ledger = self._rubric.get_last_ledger() or {}
        metrics = self._format_metrics(ledger, termination)

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self._stop_condition,
            metrics=metrics,
        )

    def _parse_success_value(self, termination: Any) -> float:
        if isinstance(termination, bool):
            return float(termination)
        value = getattr(termination, "value", termination)
        return 0.0 if str(value).lower() == "malformed" else 1.0

    def _format_metrics(self, ledger: Dict[str, Any], termination: Any) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for key, value in ledger.items():
            if isinstance(value, (int, float)):
                metrics[f"ledger/{key}"] = float(value)
        metrics["ledger/parse_success"] = self._parse_success_value(termination)
        metrics["ledger/difficulty_multiplier"] = float(
            ledger.get("difficulty_multiplier", 1.0)
        )
        metrics["reward/scalar"] = float(ledger.get("reward_scalar", 0.0))
        return metrics
