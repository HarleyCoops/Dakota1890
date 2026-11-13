"""Build a tool-safe SFT file: the single-turn SFT rows plus solved lookup conversations.

    python build_mixed_sft.py --task-dir data/lexgym_tasks --out data/lexgym_tasks/sft_mixed.jsonl

A lookup conversation is built from the task and the lookup index, with no model:
the assistant sends `LOOKUP: <word>`, the user turn is the text the lookup tool
returns, and the last assistant turn is the gold answer. Each conversation is
replayed with the rules of `lexgym.tinker_env.LexgymEnv` and kept only when the
final answer scores 1.0 and the target entry was read. Only tasks from
rl_train.jsonl (training word families) are used.

Each conversation is written as one row per assistant turn (the conversation up to
and including that turn). The SFT loss is on the last assistant message of a row,
so every turn is trained with the same token prefix that the model sees at
generation time. The Qwen3.5 renderer needs this: it does not have the sequence
extension property.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from lexgym.orthography import loose, plain
from lexgym.scoring import score
from lexgym.tasks import LOOKUP_BUDGET, LOOKUP_SYSTEM_PROMPT

LOOKUP_RE = re.compile(r"^\s*LOOKUP\s*:\s*(.+?)\s*$", re.I | re.M)  # same as lexgym.tinker_env
WORD_RE = re.compile(r"Dakota word (\S+?)[?]?(?: is derived|$)")


class Index:
    """Same lookup behaviour as lexgym.tinker_env.LexiconIndex (kept here so the build needs no Tinker import)."""

    def __init__(self, path: Path):
        self.entries = json.loads(Path(path).read_text(encoding="utf-8"))
        self._loose = collections.defaultdict(list)
        for key in self.entries:
            self._loose[loose(key)].append(key)

    def lookup(self, word: str):
        key = plain(word).strip(" .").replace(" ", "")
        if key in self.entries:
            return "ENTRY: " + " | ".join(self.entries[key]), key
        near = self._loose.get(loose(key), [])
        if near:
            return f"NO ENTRY for '{key}'. Similar spelling: {', '.join(near[:3])}", None
        return f"NO ENTRY for '{key}'.", None


def build(task: dict, index: Index):
    """Return the messages of one solved lookup conversation, or None."""
    m = WORD_RE.search(task["prompt"])
    if not m:
        return None
    word, target = m.group(1), task["info"]["target_plain"]
    hops = [word] if task["task"] == "lookup_gloss" else [word, target]
    messages = [{"role": "system", "content": LOOKUP_SYSTEM_PROMPT.format(budget=LOOKUP_BUDGET)},
                {"role": "user", "content": task["prompt"]}]
    for w in hops:
        text, found = index.lookup(w)
        if not found:
            return None
        messages += [{"role": "assistant", "content": f"LOOKUP: {w}"}, {"role": "user", "content": text}]
    messages.append({"role": "assistant", "content": f"ANSWER: {task['answer']}"})
    return messages


def replay(task: dict, messages: list[dict], index: Index) -> bool:
    """Replay the assistant turns with the environment's rules. True when reward is 1.0 and the target was read."""
    lookups, seen, turn = 0, False, 2
    while turn < len(messages):
        text = messages[turn]["content"]
        visible = text.split("</think>")[-1]
        call = LOOKUP_RE.search(visible) if "ANSWER" not in visible.upper() else None
        if call and lookups < LOOKUP_BUDGET:
            lookups += 1
            result, found = index.lookup(call.group(1))
            seen = seen or (found is not None and found == task["info"].get("target_plain"))
            if turn + 1 >= len(messages) or messages[turn + 1]["content"] != result:
                return False
            turn += 2
            continue
        if call or turn != len(messages) - 1:
            return False
        reward, _ = score(task, text)
        return float(reward) == 1.0 and seen
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-dir", default="data/lexgym_tasks")
    ap.add_argument("--out", default="data/lexgym_tasks/sft_mixed.jsonl")
    ap.add_argument("--one-hop", type=int, default=3000, help="lookup_gloss conversations to add")
    ap.add_argument("--two-hop", type=int, default=-1, help="lookup_root_gloss conversations to add (-1: all valid)")
    ap.add_argument("--max-chars", type=int, default=2400, help="drop conversations longer than this")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.task_dir)
    index = Index(d / "lookup_index.json")
    rng = random.Random(a.seed)
    tasks = [json.loads(line) for line in (d / "rl_train.jsonl").open(encoding="utf-8")]
    tasks = [t for t in tasks if t["task"].startswith("lookup") and t["split"] == "train"]
    rng.shuffle(tasks)
    limits = {"lookup_gloss": a.one_hop, "lookup_root_gloss": a.two_hop}
    kept, stats = [], collections.Counter()
    for t in tasks:
        name = t["task"]
        stats[f"{name}/seen"] += 1
        if 0 <= limits[name] <= stats[f"{name}/kept"]:
            continue
        messages = build(t, index)
        if messages is None:
            stats[f"{name}/no_entry"] += 1
        elif sum(len(m["content"]) for m in messages) > a.max_chars:
            stats[f"{name}/too_long"] += 1
        elif not replay(t, messages, index):
            stats[f"{name}/replay_failed"] += 1
        else:
            for end in range(3, len(messages) + 1, 2):  # one row per assistant turn
                kept.append({"id": f"{t['id']}#turn{end // 2}", "task": name, "messages": messages[:end]})
                stats[f"{name}/rows"] += 1
            stats[f"{name}/kept"] += 1
    base = (d / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()
    with Path(a.out).open("w", encoding="utf-8") as h:
        for line in base:
            h.write(line + "\n")
        for row in kept:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"single_turn_rows": len(base), "lookup_rows": len(kept), "total": len(base) + len(kept), **dict(sorted(stats.items()))}
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
