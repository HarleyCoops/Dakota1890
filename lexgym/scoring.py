"""Deterministic verifiers. One function per task type, one ledger per rollout.

Rules that hold for every task:
- Only the final answer span is scored (text after the last `ANSWER:`; if there
  is none, the last non-empty line). Gold text inside reasoning earns nothing.
- `reward` is in [0, 1]. Nothing multiplies it except the length guard.
- The ledger holds every component, so a reward can be reconstructed by hand.
"""
from __future__ import annotations

import re

from .orthography import canon, loose, plain

_THINK = re.compile(r"<think>.*?</think>", re.S)
_STOP = set("a an the to of as or and in on for with by be is it one any which that at from".split())
MAX_SPAN_RATIO = 3.0


def extract_answer(text: str) -> tuple[str, bool]:
    """Return (answer span, parse_ok)."""
    t = _THINK.sub("", text or "")
    if "</think>" in t:
        t = t.split("</think>")[-1]
    # a thinking block that never closes (truncated output) has no final answer
    t = t.split("<think>")[0].strip()
    m = list(re.finditer(r"ANSWER\s*:", t, flags=re.I))
    if m:
        rest = t[m[-1].end():].strip()
        span = rest.splitlines()[0].strip() if rest else ""
        return span.strip(" `\"*"), bool(span)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return (lines[-1].strip(" `\"*") if lines else ""), False


def _forms(span: str) -> set[str]:
    return {p for p in (plain(x).strip(" .()") for x in re.split(r"[,;/\n]|\band\b", span)) if p}


def _set_f1(pred: set[str], gold: set[str]) -> float:
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if not tp:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def _tokens(text: str) -> set[str]:
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    return (words - _STOP) or words


def _dice(a: str, b: str) -> float:
    ga = {a[i:i + 2] for i in range(len(a) - 1)}
    gb = {b[i:i + 2] for i in range(len(b) - 1)}
    return 2 * len(ga & gb) / (len(ga) + len(gb)) if ga and gb else 0.0


def _cuts(syllables: list[str]) -> set[int]:
    out, pos = set(), 0
    for s in syllables[:-1]:
        pos += len(s)
        out.add(pos)
    return out


def _score_syllabify(span: str, task: dict) -> dict:
    cand = canon(span).replace(" ", "")
    gold = task["answer"]
    letters_ok = float(plain(cand) == plain(gold))
    exact = float(cand == gold)
    led = {"exact": exact, "letters_ok": letters_ok, "boundary_f1": 0.0, "stress_ok": 0.0}
    if not letters_ok:
        return {**led, "raw": 0.0}
    parts = cand.split("-")
    syl = [p.replace("'", "") for p in parts]
    stressed = [i for i, p in enumerate(parts) if "'" in p]
    gold_parts = gold.split("-")
    gold_syl = [p.replace("'", "") for p in gold_parts]
    gold_stress = next((i for i, p in enumerate(gold_parts) if "'" in p), 0)
    pc, gc = _cuts(syl), _cuts(gold_syl)
    led["boundary_f1"] = 1.0 if not pc and not gc else _set_f1({str(c) for c in pc}, {str(c) for c in gc})
    # stress is compared by character position, so a wrong split cannot hide it
    def pos(sy, i):
        return sum(len(s) for s in sy[:i + 1])
    led["stress_ok"] = float(len(stressed) == 1 and pos(syl, stressed[0]) == pos(gold_syl, gold_stress))
    return {**led, "raw": 1.0 if exact else 0.6 * led["boundary_f1"] + 0.2 * led["stress_ok"]}


def _score_set(span: str, task: dict) -> dict:
    pred, gold = _forms(span), {plain(a) for a in task["accepted"]}
    f1 = _set_f1(pred, gold)
    lf1 = _set_f1({loose(p) for p in pred}, {loose(g) for g in gold})
    return {"exact": float(pred == gold), "set_f1": f1, "loose_f1": lf1, "raw": f1}


def _score_choice(span: str, task: dict) -> dict:
    def norm(s):
        return re.sub(r"[^a-z]", "", s.lower())
    exact = float(norm(span) in {norm(a) for a in task["accepted"]})
    return {"exact": exact, "raw": exact}


def _score_en_to_dak(span: str, task: dict) -> dict:
    cand = plain(span).strip(" .").replace(" ", "")
    gold = [plain(a).replace(" ", "") for a in task["accepted"]]
    exact = float(cand in gold)
    lmatch = float(loose(cand) in {loose(g) for g in gold})
    near = max((_dice(cand, g) for g in gold), default=0.0)
    raw = 1.0 if exact else 0.7 if lmatch else 0.25 * near
    return {"exact": exact, "loose_match": lmatch, "char_dice": near, "raw": raw}


def _score_gloss(span: str, task: dict) -> dict:
    pred = _tokens(span)
    best = max((_set_f1(pred, _tokens(a)) for a in task["accepted"]), default=0.0)
    return {"exact": float(best == 1.0), "token_f1": best, "raw": best}


_SCORERS = {
    "syllabify": _score_syllabify,
    "inflect": _score_set,
    "root": _score_set,
    "pos": _score_choice,
    "en_to_dak": _score_en_to_dak,
    "gloss": _score_gloss,
    "lookup_gloss": _score_gloss,
    "lookup_root_gloss": _score_gloss,
}


def score(task: dict, completion: str) -> tuple[float, dict]:
    """Score one completion. Returns (reward, ledger)."""
    span, parse_ok = extract_answer(completion)
    led = _SCORERS[task["task"]](span, task) if span else {"exact": 0.0, "raw": 0.0}
    longest = max([len(task.get("answer") or "")] + [len(a) for a in task["accepted"]])
    ratio = len(span) / max(1, longest)
    length_mult = 1.0 if ratio <= MAX_SPAN_RATIO or len(span) <= 24 else MAX_SPAN_RATIO / ratio
    reward = led["raw"] * length_mult
    ledger = {k: float(v) for k, v in led.items()}
    ledger.update(parse_ok=float(parse_ok), answer_len=float(len(span)), length_mult=length_mult, reward=reward)
    return reward, ledger
