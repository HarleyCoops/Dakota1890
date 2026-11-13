"""Prompts, the dictionary lookup tool, and the model loop for the lexgym demo.

The prompts and the lookup rules are the same as in lexgym/tasks.py and lexgym/tinker_env.py
(repository HarleyCoops/Dakota1890), so the demo asks the model the questions it was trained on.
"""
from __future__ import annotations

import collections
import json
import re
import unicodedata
from pathlib import Path

BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
SYSTEM_PROMPT = (
    "You answer from Stephen R. Riggs, A Dakota-English Dictionary (1890). "
    "Use the spelling of that book. Give only the answer on one line, starting with 'ANSWER:'."
)
LOOKUP_SYSTEM_PROMPT = (
    "You answer from Stephen R. Riggs, A Dakota-English Dictionary (1890). "
    "You can read the dictionary. To read the entry for a Dakota word, reply with exactly one line: "
    "'LOOKUP: <word>'. You get the entry in the next message. You have {budget} lookups. "
    "When you know the answer, reply with one line that starts with 'ANSWER:'."
)
LOOKUP_BUDGET = 3
PROMPTS = {
    "gloss": "Give the English definition of the Dakota word: {w}",
    "syllabify": "Write the Dakota word with Riggs' syllable division and accent mark (example: a-hi'-yo-ka-sin): {w}",
    "pos": "Give Riggs' part-of-speech abbreviation for the Dakota word: {w}",
    "inflect": "Give the inflected form or forms that Riggs lists under the Dakota word: {w} ({pos})",
    "root": "Give the Dakota word from which this word is derived: {w}",
    "en_to_dak": "Give the Dakota word for the English word: {w} ({pos})",
    "lookup_gloss": "Read the dictionary. What is the English definition of the Dakota word {w}?",
    "lookup_root_gloss": "Read the dictionary. The Dakota word {w} is derived from another word. "
                         "What is the English definition of that other word?",
}
NEEDS_POS = {"inflect", "en_to_dak"}
LOOKUP_RE = re.compile(r"^\s*LOOKUP\s*:\s*(.+?)\s*$", re.I | re.M)
ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.*)", re.I | re.S)

VARIANT_MAP = {"š": "ś", "ṡ": "ś", "ș": "ś", "ž": "ź", "ż": "ź", "η": "ŋ", "ƞ": "ŋ", "ṅ": "ŋ", "ń": "ŋ", "ņ": "ŋ",
               "ǧ": "ġ", "ğ": "ġ", "ǵ": "ġ", "ȟ": "ḣ", "ħ": "ḣ", "č": "ć", "ċ": "ć", "é": "ć", "ţ": "ṭ", "ț": "ṭ",
               "ṫ": "ṭ", "ķ": "ḳ"}
_TRANS = str.maketrans(VARIANT_MAP)
_ACCENT_RE = re.compile("[" + re.escape("'’ʹ´ˊ`′") + "]")


def canon(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "").lower().translate(_TRANS)
    return re.sub(r"\s+", " ", _ACCENT_RE.sub("'", t)).strip()


def plain(text: str) -> str:
    return canon(text).replace("-", "").replace("'", "")


def loose(text: str) -> str:
    return plain(text).replace("ŋ", "n").replace(" ", "")


class Index:
    """The lookup tool: one call returns the printed entry or entries for one Dakota word."""

    def __init__(self, path: Path):
        self.entries: dict[str, list[str]] = json.loads(Path(path).read_text(encoding="utf-8"))
        self._loose = collections.defaultdict(list)
        for key in self.entries:
            self._loose[loose(key)].append(key)

    def lookup(self, word: str) -> tuple[str, str | None]:
        key = plain(word).strip(" .").replace(" ", "")
        if key in self.entries:
            return "ENTRY: " + " | ".join(self.entries[key]), key
        near = self._loose.get(loose(key), [])
        if near:
            return f"NO ENTRY for '{key}'. Similar spelling: {', '.join(near[:3])}", None
        return f"NO ENTRY for '{key}'.", None


def build_prompt(task: str, word: str, pos: str = "") -> tuple[str, str]:
    system = LOOKUP_SYSTEM_PROMPT.format(budget=LOOKUP_BUDGET) if task.startswith("lookup") else SYSTEM_PROMPT
    return system, PROMPTS[task].format(w=word, pos=pos or "?")


def clean_answer(text: str) -> tuple[str, bool]:
    """Return (answer text, True when the reply had an ANSWER line)."""
    visible = text.split("</think>")[-1].split("<|content_text|>")[-1]
    m = ANSWER_RE.search(visible)
    if m:
        return m.group(1).strip().splitlines()[0].strip() if m.group(1).strip() else "", True
    visible = re.sub(r"<\|[^|]*\|>", " ", visible).strip()
    return (visible[:280] + " …") if len(visible) > 280 else visible, False


async def run_model(client, model: str, task: str, word: str, pos: str, index: Index, max_tokens: int = 96) -> dict:
    """Ask one model one question. For lookup tasks, run the tool loop. Returns the steps and token use."""
    system, prompt = build_prompt(task, word, pos)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    steps, lookups, tokens = [], 0, {"prompt": 0, "completion": 0}
    for _ in range(LOOKUP_BUDGET + 1):
        out = await client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                                   temperature=0.0, reasoning_effort="none")
        text = out.choices[0].message.content or ""
        if out.usage:
            tokens["prompt"] += out.usage.prompt_tokens or 0
            tokens["completion"] += out.usage.completion_tokens or 0
        visible = text.split("</think>")[-1]
        call = LOOKUP_RE.search(visible) if task.startswith("lookup") and "ANSWER" not in visible.upper() else None
        if call and lookups < LOOKUP_BUDGET:
            lookups += 1
            result, _ = index.lookup(call.group(1))
            steps.append({"type": "lookup", "word": call.group(1), "result": result})
            messages += [{"role": "assistant", "content": f"LOOKUP: {call.group(1)}"}, {"role": "user", "content": result}]
            continue
        answer, ok = clean_answer(text)
        steps.append({"type": "answer", "text": answer, "format_ok": ok})
        break
    return {"steps": steps, "lookups": lookups, "tokens": tokens}


DERIVED_RE = re.compile(r"\[derived from: ([^\]]+)\]")


def book_entries(index: Index, task: str, word: str) -> list[dict]:
    """The printed entries for the word, for display beside the model answers. Empty for English input.

    For the two-hop task the entries of the word it is derived from are added, because the answer is there.
    """
    if task == "en_to_dak":
        return []
    own = index.entries.get(plain(word).strip(" .").replace(" ", ""), [])
    out = [{"text": e, "role": "word"} for e in own]
    if task == "lookup_root_gloss":
        for e in own:
            for root in (r.strip() for m in DERIVED_RE.finditer(e) for r in m.group(1).split(",")):
                out += [{"text": x, "role": "root"} for x in index.entries.get(plain(root), [])]
    return out
