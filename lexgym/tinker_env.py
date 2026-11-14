"""Tinker (tinker-cookbook) environments for lexgym tasks.

Single-turn tasks end after one answer. Lookup tasks are multi-turn: the model
may send `LOOKUP: <word>` up to LOOKUP_BUDGET times and gets the dictionary
entry back as the next user message. Reward is paid only on the final answer.

Reward arms: `LexgymDatasetBuilder.arm` (default "baseline") is injected into
every task row as `task["arm"]`, and scoring.py selects the reward composition
from it. The trace metadata records the arm and its scorer version.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import random
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

import chz
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.rl.types import Action, ActionExtra, Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .orthography import loose, plain
from .scoring import extract_answer, score, scorer_version
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


TRACE_VERSION = 1


class TraceWriter:
    """Append one JSON line per rollout. See docs/TRACE_SCHEMA.md."""

    def __init__(self, trace_dir: str | Path, run: dict):
        self.dir = Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run = run

    def write(self, role: str, records: list[dict]) -> None:
        with (self.dir / f"{role}.jsonl").open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps({"trace_version": TRACE_VERSION, "run": self.run, **record}, ensure_ascii=False) + "\n")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance field only
        return ""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(model_input) -> list[int]:
    try:
        return list(model_input.to_ints())
    except Exception:  # noqa: BLE001 - a non-text chunk; the messages still hold the text
        return []


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
        self.ledger: dict[str, float] = {}
        self.answer_span = ""
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
        ledger = {k: float(v) for k, v in ledger.items()}
        if self.is_lookup:
            ledger.update(lookups_used=float(self.lookups), target_entry_seen=self.target_seen,
                          budget_exhausted=float(bool(call)))
        self.ledger = ledger
        self.answer_span = "" if call else extract_answer(text)[0]
        self.messages.append({"role": "assistant", "content": text})
        metrics = {f"ledger/{k}": v for k, v in ledger.items()}
        return StepResult(reward=float(reward), episode_done=True, next_observation=tinker.ModelInput.empty(),
                          next_stop_condition=self.stop, metrics=metrics)


class LexgymGroupBuilder(EnvGroupBuilder):
    def __init__(self, task: dict, renderer, index, group_size: int, trace: TraceWriter | None = None,
                 position: dict | None = None):
        self.task, self.renderer, self.index, self.group_size = task, renderer, index, group_size
        self.trace, self.position = trace, position or {}
        self.uses = 0  # the cookbook builds the evaluation builders once and reuses them

    async def make_envs(self) -> Sequence[Env]:
        self.uses += 1
        return [LexgymEnv(self.task, self.renderer, self.index) for _ in range(self.group_size)]

    async def compute_group_rewards(self, trajectory_group, env_group):
        if self.trace is not None:
            self._write_traces(trajectory_group, env_group)
        return [(0.0, {}) for _ in trajectory_group]

    def _write_traces(self, trajectories, envs) -> None:
        rewards = [sum(t.reward for t in traj.transitions) for traj in trajectories]
        mean = sum(rewards) / len(rewards) if rewards else 0.0
        now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        records = []
        for i, (traj, env, reward) in enumerate(zip(trajectories, envs, rewards)):
            turns = [{
                "ob_tokens": _tokens(t.ob),
                "ac_tokens": list(t.ac.tokens),
                "ac_logprobs": list(t.ac.maybe_logprobs) if t.ac.maybe_logprobs is not None else None,
                "stop_reason": str(getattr(t.ac, "stop_reason", "")),
                "sequence_id": getattr(t.ac, "sequence_id", None),
                "reward": float(t.reward),
            } for t in traj.transitions]
            records.append({
                "time": now,
                "position": {**self.position, "rollout_index": i, "group_size": len(trajectories),
                             "pass_index": max(self.position.get("pass_index", 0), self.uses - 1)},
                "task": {k: self.task[k] for k in ("id", "task", "split", "prompt", "answer", "accepted", "info")},
                "messages": env.messages,
                "turns": turns,
                "scoring": {"answer_span": env.answer_span, "ledger": env.ledger, "reward": reward,
                            "group_mean_reward": mean, "advantage": reward - mean},
            })
        self.trace.write(self.position.get("role", "train"), records)

    def logging_tags(self) -> list[str]:
        return [f"task/{self.task['task']}", f"split/{self.task['split']}"]


class LexgymDataset(RLDataset):
    def __init__(self, tasks: list[dict], batch_size: int, group_size: int, renderer, index,
                 trace: TraceWriter | None = None, role: str = "train"):
        self.tasks, self.batch_size, self.group_size = tasks, batch_size, group_size
        self.renderer, self.index, self.trace, self.role = renderer, index, trace, role
        self.passes = -1  # counts full passes: epochs for train, evaluation passes for eval

    def __len__(self) -> int:
        return math.ceil(len(self.tasks) / self.batch_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        if index == 0:
            self.passes += 1
        chunk = self.tasks[index * self.batch_size:(index + 1) * self.batch_size]
        return [LexgymGroupBuilder(
            t, self.renderer, self.index, self.group_size, self.trace,
            {"role": self.role, "pass_index": self.passes, "batch_index": index, "group_index": g},
        ) for g, t in enumerate(chunk)]


def load_tasks(path: Path, tasks: Sequence[str] | None = None, limit_per_task: int = -1, seed: int = 0,
               arm: str = "baseline") -> list[dict]:
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
    # every task row carries its reward arm; scoring.py reads it
    return [{**r, "arm": arm} for r in rows]


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
    # reward arm: "baseline" or "gated" (see scoring.py); recorded in the traces
    arm: str = "baseline"
    # traces: one JSON line per rollout, with text, token ids, log-probabilities and the ledger
    trace_dir: str | None = None
    run_id: str = ""
    stage: str = ""
    load_checkpoint_path: str | None = None
    temperature: float = 1.0
    max_tokens: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        name = pick_renderer_name(self.model_name, self.renderer_name)
        renderer = renderers.get_renderer(name, tokenizer=get_tokenizer(self.model_name))
        d = Path(self.task_dir)
        index = LexiconIndex(d / "lookup_index.json")
        train = load_tasks(d / "rl_train.jsonl", self.train_tasks, self.train_limit_per_task, self.seed, self.arm)
        evals = (load_tasks(d / "eval_heldout_rule.jsonl", None, self.eval_limit_per_task, self.seed, self.arm)
                 + load_tasks(d / "eval_recall.jsonl", None, self.eval_limit_per_task, self.seed, self.arm))
        trace = None
        if self.trace_dir:
            trace = TraceWriter(self.trace_dir, {
                "run_id": self.run_id, "stage": self.stage, "base_model": self.model_name,
                "load_checkpoint_path": self.load_checkpoint_path, "renderer": name,
                "temperature": self.temperature, "max_tokens": self.max_tokens,
                "train_group_size": self.group_size, "train_batch_size": self.batch_size, "seed": self.seed,
                "arm": self.arm, "scorer_version": scorer_version(self.arm),
                "lexgym_commit": _git_commit(),
                "task_files_sha256": {f: _sha256(d / f) for f in
                                      ("rl_train.jsonl", "eval_heldout_rule.jsonl", "eval_recall.jsonl", "lookup_index.json")},
            })
        return (LexgymDataset(train, self.batch_size, self.group_size, renderer, index, trace, "train"),
                LexgymDataset(evals, self.batch_size, 1, renderer, index, trace, "eval"))
