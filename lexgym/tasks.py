"""Generate SFT rows and RL tasks from the lexicon graph.

Task row: {id, task, split, prompt, answer, accepted, info{entry_id, page, source_image, ...}}

Splits
- train          : families in the train split.
- heldout_rule   : families never seen in training. Only tasks that a rule can
                   answer (syllabify, pos, inflect, root) and the lookup tasks.
                   This measures generalization.
- recall         : a fixed sample of TRAIN vocabulary (gloss, en_to_dak). A new
                   word cannot be guessed, so these measure memory of the
                   volume, not generalization. They stay in training.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from .orthography import plain

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
POS_CLOSED = ["n.", "v. a.", "v. n.", "v.", "adv.", "adj.", "cont.", "v. red.", "part.", "v. pos.",
              "v. reflex.", "pron.", "prep.", "intj.", "conj.", "num. adj."]
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


def _row(task, key, split, w, answer, accepted, info, **fmt):
    return {"id": f"{task}:{key}", "task": task, "split": split,
            "prompt": PROMPTS[task].format(w=w, **fmt), "answer": answer, "accepted": accepted, "info": info}


def format_entry(e: dict) -> str:
    """Text the lookup tool returns for one entry."""
    parts = [f"{e['headword'] or e['headword_raw']}", f"{e['pos'] or ''}".strip(), e["gloss"] or "(no definition printed)"]
    text = ", ".join(p for p in parts if p)
    if e["derived_from"]:
        text += f" [derived from: {', '.join(e['derived_from'])}]"
    if e["inflected_forms"]:
        text += f" [forms: {', '.join(e['inflected_forms'])}]"
    return text + f" (scan page {e['page']})"


def generate(graph_dir: Path, out_dir: Path, seed: int = 42, recall_n: int = 300) -> dict:
    load = lambda n: [json.loads(l) for l in (graph_dir / f"{n}.jsonl").open(encoding="utf-8")]
    dakota, english = load("dakota_entries"), load("english_entries")
    by_plain = collections.defaultdict(list)
    for e in dakota:
        if e["headword_ok"]:
            by_plain[e["plain"]].append(e)
    rows = []
    for e in dakota:
        if not e["headword_ok"]:
            continue
        w, sp = e["plain"], e["split"]
        info = {"entry_id": e["entry_id"], "page": e["page"], "source_image": e["source_image"],
                "family": e["family"], "confidence": e["confidence"]}
        homographs = by_plain[w]
        if len(e["syllables"]) > 1 and len({h["headword"] for h in homographs}) == 1:
            rows.append(_row("syllabify", e["entry_id"], sp, w, e["headword"], [e["headword"]], info))
        if e["pos"] in POS_CLOSED:
            rows.append(_row("pos", e["entry_id"], sp, w, e["pos"], sorted({h["pos"] for h in homographs if h["pos"]}), info))
        if e["inflected_forms"]:
            rows.append(_row("inflect", e["entry_id"], sp, w, ", ".join(e["inflected_forms"]), e["inflected_forms"], info,
                             pos=e["pos"] or "?"))
        roots = [r for r in e["derived_from"] if r in by_plain]
        if roots and len(roots) == len(e["derived_from"]):
            rows.append(_row("root", e["entry_id"], sp, w, ", ".join(roots), roots, info))
            target = by_plain[roots[0]][0]
            if len(roots) == 1 and target["gloss"] and len(by_plain[roots[0]]) == 1:
                rows.append(_row("lookup_root_gloss", e["entry_id"], sp, w, target["gloss"], [target["gloss"]],
                                 {**info, "hops": 2, "target_plain": roots[0], "target_entry_id": target["entry_id"]}))
        if e["gloss"]:
            glosses = sorted({h["gloss"] for h in homographs if h["gloss"]})
            if sp == "train":  # a held-out word has no guessable definition
                rows.append(_row("gloss", e["entry_id"], sp, w, e["gloss"], glosses, info))
            rows.append(_row("lookup_gloss", e["entry_id"], sp, w, e["gloss"], glosses,
                             {**info, "hops": 1, "target_plain": w, "target_entry_id": e["entry_id"]}))
    dropped_english = 0
    for e in english:
        if not e["dakota_forms"]:
            continue
        if e["split"] != "train":
            dropped_english += 1
            continue
        rows.append(_row("en_to_dak", e["entry_id"], "train", e["english"], e["dakota_forms"][0], e["dakota_forms"],
                         {"entry_id": e["entry_id"], "page": e["page"], "source_image": e["source_image"],
                          "gold_tier": "two_witnesses" if e["attested_forms"] else "one_witness",
                          "confidence": e["confidence"]}, pos=e["pos"] or "?"))

    rng = random.Random(seed)
    train = [r for r in rows if r["split"] == "train"]
    heldout = [dict(r, split="heldout_rule") for r in rows if r["split"] == "heldout"]
    recall = []
    for task in ("gloss", "en_to_dak"):
        pool = [r for r in train if r["task"] == task and (task != "en_to_dak" or r["info"]["gold_tier"] == "two_witnesses")]
        recall += [dict(r, split="recall") for r in rng.sample(pool, min(recall_n, len(pool)))]
    # RL trains only on gold with two witnesses for en_to_dak; SFT sees all single-turn train rows
    rl_train = [r for r in train if r["task"] != "en_to_dak" or r["info"]["gold_tier"] == "two_witnesses"]
    sft = [{"id": r["id"], "task": r["task"], "messages": [
        {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["prompt"]},
        {"role": "assistant", "content": f"ANSWER: {r['answer']}"}]} for r in train if not r["task"].startswith("lookup")]

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"rl_train": rl_train, "eval_heldout_rule": heldout, "eval_recall": recall, "sft_train": sft}
    for name, data in files.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as h:
            for r in data:
                h.write(json.dumps(r, ensure_ascii=False) + "\n")
    lookup_index = {p: [format_entry(e) for e in es] for p, es in by_plain.items()}
    (out_dir / "lookup_index.json").write_text(json.dumps(lookup_index, ensure_ascii=False), encoding="utf-8")
    report = {name: dict(sorted(collections.Counter(r["task"] for r in data).items()), total=len(data))
              for name, data in files.items()}
    report["english_rows_dropped_for_heldout_forms"] = dropped_english
    train_fams = {r["info"].get("family") for r in rl_train} - {None}
    report["family_overlap_train_vs_heldout"] = len(train_fams & {r["info"]["family"] for r in heldout})
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default="data/lexicon_graph")
    ap.add_argument("--out", default="data/lexgym_tasks")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(json.dumps(generate(Path(a.graph), Path(a.out), a.seed), indent=2))


if __name__ == "__main__":
    main()
