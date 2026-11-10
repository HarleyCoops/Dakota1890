"""Tinker (tinker-cookbook) environments for lexgym tasks.

Single-turn tasks end after one answer. Lookup tasks are multi-turn: the model
may send `LOOKUP: <word>` up to LOOKUP_BUDGET times and gets the dictionary
entry back as the next user message. Reward is paid only on the final answer.
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any, Sequence

import chz
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.rl.types import Action, ActionExtra, Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .orthography import loose, plain
from .scoring import score
from .tasks import LOOKUP_BUDGET, LOOKUP_SYSTEM_PROMPT, SYSTEM_PROMPT

_LOOKUP_RE = re.compile(r"^\s*LOOKUP\s*:\s*(.+?)\s*$", re.I | re.M)


def _text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def pick_renderer_name(model_name: str, renderer_name: str | None = None) -> str:
    """Prefer the renderer without a thinking block: answers are one short line."""
    if renderer_name:
        return renderer_name
    names = model_info.get_recommended_renderer_names(model_name)
    return next((n for n in names if n.endswith("disable_thinking")), names[0])


class LexiconIndex:
    """Read-only dictionary used by the lookup tool."""

    def __init__(self, path: Path):
        self.entries: dict[str, list[str]] = json.loads(Path(path).read_text(encoding="utf-8"))
        self._loose: dict[str, list[str]] = {}
        for key in self.entries:
            self._loose.setdefault(loose(key), []).append(key)

    def lookup(self, word: str) -> tuple[str, str | None]:
        """Return (text for the model, plain key that was found or None)."""
        key = plain(word).strip(" .").replace(" ", "")
        if key in self.entries:
            return "ENTRY: " + " | ".join(self.entries[key]), key
        near = self._loose.get(loose(key), [])
        if near:
            return f"NO ENTRY for '{key}'. Similar spelling: {', '.join(near[:3])}", None
        return f"NO ENTRY for '{key}'.", None


class LexgymEnv(Env):
    def __init__(self, task: dict, renderer: renderers.Renderer, index: LexiconIndex | None):
        self.task, self.renderer, self.index = task, renderer, index
        self.is_lookup = task["task"].startswith("lookup")
        system = LOOKUP_SYSTEM_PROMPT.format(budget=LOOKUP_BUDGET) if self.is_lookup else SYSTEM_PROMPT
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system},
                                               {"role": "user", "content": task["prompt"]}]
        self.lookups = 0
        self.target_seen = 0.0
        self.stop = renderer.get_stop_sequences()

    def _observation(self):
        out = self.renderer.build_generation_prompt(self.messages)
        return out[0] if isinstance(out, tuple) else out

    async def initial_observation(self):
        return self._observation(), self.stop

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, _ = self.renderer.parse_response(action)
        text = _text(message)
        visible = text.split("</think>")[-1]
        call = _LOOKUP_RE.search(visible) if self.is_lookup and "ANSWER" not in visible.upper() else None
        if call and self.lookups < LOOKUP_BUDGET and self.index is not None:
            self.lookups += 1
            result, found = self.index.lookup(call.group(1))
            if found and found == self.task["info"].get("target_plain"):
                self.target_seen = 1.0
            self.messages += [{"role": "assistant", "content": visible.strip()}, {"role": "user", "content": result}]
            return StepResult(reward=0.0, episode_done=False, next_observation=self._observation(),
                              next_stop_condition=self.stop, metrics={})
        reward, ledger = (0.0, {"exact": 0.0, "raw": 0.0, "parse_ok": 0.0, "reward": 0.0}) if call else score(self.task, text)
        metrics = {f"ledger/{k}": float(v) for k, v in ledger.items()}
        if self.is_lookup:
            metrics.update({"ledger/lookups_used": float(self.lookups), "ledger/target_entry_seen": self.target_seen,
                            "ledger/budget_exhausted": float(bool(call))})
        return StepResult(reward=float(reward), episode_done=True, next_observation=tinker.ModelInput.empty(),
                          next_stop_condition=self.stop, metrics=metrics)


class LexgymGroupBuilder(EnvGroupBuilder):
    def __init__(self, task: dict, renderer, index, group_size: int):
        self.task, self.renderer, self.index, self.group_size = task, renderer, index, group_size

    async def make_envs(self) -> Sequence[Env]:
        return [LexgymEnv(self.task, self.renderer, self.index) for _ in range(self.group_size)]

    async def compute_group_rewards(self, trajectory_group, env_group):
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return [f"task/{self.task['task']}", f"split/{self.task['split']}"]


class LexgymDataset(RLDataset):
    def __init__(self, tasks: list[dict], batch_size: int, group_size: int, renderer, index):
        self.tasks, self.batch_size, self.group_size = tasks, batch_size, group_size
        self.renderer, self.index = renderer, index

    def __len__(self) -> int:
        return math.ceil(len(self.tasks) / self.batch_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        chunk = self.tasks[index * self.batch_size:(index + 1) * self.batch_size]
        return [LexgymGroupBuilder(t, self.renderer, self.index, self.group_size) for t in chunk]


def load_tasks(path: Path, tasks: Sequence[str] | None = None, limit_per_task: int = -1, seed: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8")]
    if tasks:
        rows = [r for r in rows if r["task"] in set(tasks)]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit_per_task > 0:
        kept, count = [], {}
        for r in rows:
            if count.get(r["task"], 0) < limit_per_task:
                kept.append(r)
                count[r["task"]] = count.get(r["task"], 0) + 1
        rows = kept
    return rows


@chz.chz
class LexgymDatasetBuilder(RLDatasetBuilder):
    model_name: str
    task_dir: str = "data/lexgym_tasks"
    batch_size: int = 32
    group_size: int = 8
    train_tasks: Sequence[str] | None = None
    train_limit_per_task: int = -1
    eval_limit_per_task: int = 50
    renderer_name: str | None = None
    seed: int = 42

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        name = pick_renderer_name(self.model_name, self.renderer_name)
        renderer = renderers.get_renderer(name, tokenizer=get_tokenizer(self.model_name))
        d = Path(self.task_dir)
        index = LexiconIndex(d / "lookup_index.json")
        train = load_tasks(d / "rl_train.jsonl", self.train_tasks, self.train_limit_per_task, self.seed)
        evals = (load_tasks(d / "eval_heldout_rule.jsonl", None, self.eval_limit_per_task, self.seed)
                 + load_tasks(d / "eval_recall.jsonl", None, self.eval_limit_per_task, self.seed))
        return (LexgymDataset(train, self.batch_size, self.group_size, renderer, index),
                LexgymDataset(evals, self.batch_size, 1, renderer, index))
