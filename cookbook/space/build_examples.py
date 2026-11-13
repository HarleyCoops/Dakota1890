"""Build examples.json for the demo from the public evaluation traces of the run.

    python build_examples.py            # downloads the two trace files from the Hub
    python build_examples.py e0.jsonl e3.jsonl
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

REPO = "HarleyCooper/riggs1890-lexgym-traces"
RUN = "sft_mixed_thinking_off_2026_11_12/inkling-2-small"
PER_TASK = 6
WORD_RE = {
    "lookup_gloss": r"Dakota word (\S+?)\?$", "lookup_root_gloss": r"Dakota word (\S+) is derived",
    "en_to_dak": r"English word: (.+?) \((.*)\)$", "inflect": r"Dakota word: (\S+) \((.*)\)$",
}


def load(path):
    out = {}
    for line in Path(path).open(encoding="utf-8"):
        r = json.loads(line)
        out[r["task"]["id"]] = r
    return out


def steps(row):
    from engine import clean_answer
    out, msgs = [], row["messages"][2:]
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        text = m["content"]
        if text.strip().upper().startswith("LOOKUP:") and i + 1 < len(msgs):
            out.append({"type": "lookup", "word": text.split(":", 1)[1].strip(), "result": msgs[i + 1]["content"]})
        else:
            answer, ok = clean_answer(text)
            out.append({"type": "answer", "text": answer, "format_ok": ok})
    return out


def main():
    if len(sys.argv) == 3:
        p0, p3 = sys.argv[1], sys.argv[2]
    else:
        from huggingface_hub import hf_hub_download
        p0 = hf_hub_download(REPO, f"{RUN}/eval_epoch0_base/traces/eval.jsonl", repo_type="dataset")
        p3 = hf_hub_download(REPO, f"{RUN}/eval_epoch3/traces/eval.jsonl", repo_type="dataset")
    e0, e3 = load(p0), load(p3)
    rng, out = random.Random(7), []
    by_task = {}
    for i, r in e3.items():
        by_task.setdefault(r["task"]["task"], []).append(i)
    for task, ids in sorted(by_task.items()):
        ids = sorted(ids)
        right = [i for i in ids if e3[i]["scoring"]["ledger"]["exact"] == 1.0]
        wrong = [i for i in ids if e3[i]["scoring"]["ledger"]["exact"] != 1.0]
        n_wrong = min(len(wrong), max(1, round(PER_TASK * len(wrong) / len(ids)))) if wrong else 0
        for i in rng.sample(right, PER_TASK - n_wrong) + rng.sample(wrong, n_wrong):
            t = e3[i]["task"]
            m = re.search(WORD_RE.get(task, r": (\S+)$"), t["prompt"])
            out.append({
                "id": i, "task": task, "split": t["split"], "word": m.group(1) if m else "",
                "pos": m.group(2) if m and m.lastindex == 2 else "", "prompt": t["prompt"], "gold": t["answer"],
                "page": t["info"].get("page"),
                "base": {"steps": steps(e0[i]), "exact": e0[i]["scoring"]["ledger"]["exact"] == 1.0},
                "tuned": {"steps": steps(e3[i]), "exact": e3[i]["scoring"]["ledger"]["exact"] == 1.0},
            })
    Path(__file__).with_name("examples.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(out), "examples;", sum(e["tuned"]["exact"] for e in out), "tuned exact;", sum(e["base"]["exact"] for e in out), "base exact")


if __name__ == "__main__":
    main()
