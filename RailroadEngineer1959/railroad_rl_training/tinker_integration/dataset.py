from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Sequence

import chz
from railroad_1959.environment import DEFAULT_SYSTEM_PROMPT
from tinker_cookbook import model_info, renderers
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

from .env import RailroadTinkerEnv
from .types import RailroadExample

PROMPT_TEMPLATE = """Task ID: {task_id}
Scenario: {description}

Instructions: Provide the correct response or action according to the 1959 Consolidated Code of Operating Rules."""


def _load_examples(
    dataset_path: str | Path | None,
    max_examples: int,
    seed: int,
    shuffle: bool,
) -> list[RailroadExample]:
    if dataset_path is None:
        return []

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    if shuffle:
        random.Random(seed).shuffle(raw)
    if max_examples > 0:
        raw = raw[:max_examples]

    examples: list[RailroadExample] = []
    for idx, row in enumerate(raw):
        task_id = row.get("task_id") or f"railroad_{idx:05d}"
        description = row.get("description") or ""
        answer = row.get("expected_outcome") or ""
        prompt = PROMPT_TEMPLATE.format(task_id=task_id, description=description)
        examples.append(
            RailroadExample(
                example_id=task_id,
                prompt=prompt,
                answer=answer,
                info={
                    "task_id": task_id,
                    "description": description,
                    "applicable_rules": row.get("applicable_rules", []),
                    "source": "railroad_1959",
                },
            )
        )
    return examples


def _split_eval(
    train_examples: list[RailroadExample],
    eval_fraction: float,
    eval_examples: int,
) -> tuple[list[RailroadExample], list[RailroadExample]]:
    if not train_examples or eval_fraction <= 0 and eval_examples <= 0:
        return train_examples, []

    total = len(train_examples)
    if eval_examples > 0:
        split_size = min(eval_examples, total)
    else:
        split_size = max(1, int(total * eval_fraction))

    eval_set = train_examples[:split_size]
    train_set = train_examples[split_size:]
    return train_set, eval_set


def _get_renderer(model_name: str, renderer_name: str | None = None) -> renderers.Renderer:
    tokenizer: Tokenizer = get_tokenizer(model_name)
    if renderer_name is None:
        renderer_name = model_info.get_recommended_renderer_name(model_name)
    return renderers.get_renderer(renderer_name, tokenizer=tokenizer)


class RailroadEnvGroupBuilder(EnvGroupBuilder):
    """Builds a group of Railroad 1959 environments for a single task."""

    def __init__(
        self,
        example: RailroadExample,
        renderer: renderers.Renderer,
        system_prompt: str,
        group_size: int,
    ):
        self.example = example
        self.renderer = renderer
        self.system_prompt = system_prompt
        self.group_size = group_size

    async def make_envs(self):
        return [
            RailroadTinkerEnv(
                example=self.example,
                renderer=self.renderer,
                system_prompt=self.system_prompt,
            )
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(self, trajectory_group, env_group):
        # Per-step rewards already captured; group reward unused.
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return [
            "railroad_1959",
            f"task/{self.example.example_id}",
        ]


class RailroadDataset(RLDataset):
    def __init__(
        self,
        examples: Sequence[RailroadExample],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        system_prompt: str,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if not examples:
            raise ValueError("RailroadDataset requires at least one example.")
        self.examples = list(examples)
        if shuffle:
            random.Random(seed).shuffle(self.examples)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return math.ceil(len(self.examples) / self.batch_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.examples))
        batch = self.examples[start:end]
        return [
            RailroadEnvGroupBuilder(
                example=example,
                renderer=self.renderer,
                system_prompt=self.system_prompt,
                group_size=self.group_size,
            )
            for example in batch
        ]


@chz.chz
class RailroadDatasetBuilder(RLDatasetBuilder):
    """Construct train/eval datasets for Railroad 1959 on Tinker."""

    model_name: str
    batch_size: int
    group_size: int
    dataset_path: str | None = None
    eval_path: str | None = None
    renderer_name: str | None = None
    max_examples: int = -1
    eval_examples: int = -1
    eval_fraction: float = 0.1
    system_prompt: str | None = None
    shuffle: bool = True
    seed: int = 0
    eval_group_size: int = 1

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        renderer = _get_renderer(self.model_name, self.renderer_name)
        system_prompt = self.system_prompt or DEFAULT_SYSTEM_PROMPT

        train_examples = _load_examples(self.dataset_path, self.max_examples, self.seed, self.shuffle)

        if self.eval_path:
            eval_examples = _load_examples(self.eval_path, self.eval_examples, self.seed, False)
        else:
            train_examples, eval_examples = _split_eval(train_examples, self.eval_fraction, self.eval_examples)

        train_dataset = RailroadDataset(
            examples=train_examples,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            system_prompt=system_prompt,
            shuffle=self.shuffle,
            seed=self.seed,
        )

        eval_dataset: RailroadDataset | None = None
        if eval_examples:
            eval_dataset = RailroadDataset(
                examples=eval_examples,
                batch_size=max(1, self.batch_size),
                group_size=max(1, self.eval_group_size),
                renderer=renderer,
                system_prompt=system_prompt,
                shuffle=False,
                seed=self.seed,
            )

        return train_dataset, eval_dataset
