from __future__ import annotations

import math
import random
from typing import Sequence

import chz
from datasets import Dataset
from dakota_grammar_translation.environment import (
    DEFAULT_SYSTEM_PROMPT,
    DatasetBundle,
    build_dataset_bundle,
)
from tinker_cookbook import model_info, renderers
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

from .types import DakotaGrammarExample
from .env import DakotaTinkerEnv


class DakotaGrammarEnvGroupBuilder(EnvGroupBuilder):
    """EnvGroupBuilder that produces Dakota grammar environments."""

    def __init__(
        self,
        example: DakotaGrammarExample,
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
            DakotaTinkerEnv(
                example=self.example,
                renderer=self.renderer,
                system_prompt=self.system_prompt,
            )
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(self, trajectory_group, env_group):
        # Use per-step rewards only; ledger already captures decomposition.
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return [
            "dakota",
            f"difficulty/{self.example.difficulty}",
            f"task/{self.example.task}",
        ]


class DakotaGrammarDataset(RLDataset):
    def __init__(
        self,
        examples: Sequence[DakotaGrammarExample],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        system_prompt: str,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if not examples:
            raise ValueError("DakotaGrammarDataset requires at least one example.")
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
            DakotaGrammarEnvGroupBuilder(
                example=example,
                renderer=self.renderer,
                system_prompt=self.system_prompt,
                group_size=self.group_size,
            )
            for example in batch
        ]


def _dataset_to_examples(dataset: Dataset | None) -> list[DakotaGrammarExample]:
    if dataset is None:
        return []
    rows = dataset.to_list()
    examples: list[DakotaGrammarExample] = []
    for idx, row in enumerate(rows):
        info = dict(row.get("info") or {})
        prompt = row.get("question") or row.get("prompt") or ""
        answer = row.get("answer") or ""
        task = row.get("task") or info.get("task_type") or "default"
        example_id = row.get("id") or row.get("task_id") or f"dakota_{idx:05d}"
        examples.append(
            DakotaGrammarExample(
                example_id=example_id,
                prompt=prompt,
                answer=answer,
                info=info,
                task=task,
            )
        )
    return examples


def _get_renderer(model_name: str, renderer_name: str | None = None) -> renderers.Renderer:
    tokenizer: Tokenizer = get_tokenizer(model_name)
    if renderer_name is None:
        renderer_name = model_info.get_recommended_renderer_name(model_name)
    return renderers.get_renderer(renderer_name, tokenizer=tokenizer)


@chz.chz
class DakotaGrammarDatasetBuilder(RLDatasetBuilder):
    """Build train/eval datasets for Dakota RL on Tinker."""

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
    difficulty_filter: Sequence[str] | None = None
    task_filter: Sequence[str] | None = None
    include_hints: bool = True
    eval_group_size: int = 1

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        renderer = _get_renderer(self.model_name, self.renderer_name)
        system_prompt = self.system_prompt or DEFAULT_SYSTEM_PROMPT

        bundle: DatasetBundle = build_dataset_bundle(
            dataset_path=self.dataset_path,
            eval_path=self.eval_path,
            max_examples=self.max_examples,
            eval_examples=self.eval_examples,
            eval_fraction=self.eval_fraction,
            seed=self.seed,
            difficulty_filter=self.difficulty_filter,
            task_filter=self.task_filter,
            include_hints=self.include_hints,
        )

        train_examples = _dataset_to_examples(bundle.train)
        eval_examples = _dataset_to_examples(bundle.eval)

        train_dataset = DakotaGrammarDataset(
            examples=train_examples,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            system_prompt=system_prompt,
            shuffle=self.shuffle,
            seed=self.seed,
        )

        eval_dataset: DakotaGrammarDataset | None = None
        if eval_examples:
            eval_dataset = DakotaGrammarDataset(
                examples=eval_examples,
                batch_size=max(1, self.batch_size),
                group_size=max(1, self.eval_group_size),
                renderer=renderer,
                system_prompt=system_prompt,
                shuffle=False,
                seed=self.seed,
            )

        return train_dataset, eval_dataset
