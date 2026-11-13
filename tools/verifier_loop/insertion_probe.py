"""Probe for verifier-loop milestone 2: do affix insertion and short stem changes explain Riggs' inflected forms?

    python tools/verifier_loop/insertion_probe.py            # downloads the published gold
    python tools/verifier_loop/insertion_probe.py dakota_entries.parquet

Method
- A pair is (headword, inflected form) from one entry, both in plain form (no hyphens, no accent mark).
- Each pair is reduced to ONE edit by removing the longest common prefix and then the longest common suffix:
  (old, new). A pure insert has old == "". The plural suffix "pi" is removed from the form when the form
  without it gives an edit.
- The edit inventory is LEARNED from the train families only. An edit is admitted when it occurs at least
  --min-count times. Nothing is taken from outside the book, and no affix is typed in by hand.
- Coverage is then measured on the held-out families, which were not used to build the inventory.
- Specificity: the number of different forms that the admitted edits can build from one held-out headword.
  A rule set with high coverage and a large candidate count can license a form, but it cannot choose one.

This is a feasibility probe. It is not the compiled inventory of milestone 2, which must cite Riggs' grammar.
"""
from __future__ import annotations

import argparse
import collections
import json

import pandas as pd

GOLD = "hf://datasets/HarleyCooper/riggs1890-lexicon-graph/graph/dakota_entries.parquet"
LOOSE = str.maketrans("ŋġḣśźćṭḳṗ", "nghszctkp")


def strip_pi(form: str) -> str | None:
    return form[:-2] if form.endswith("pi") and len(form) > 3 else None


def edit(head: str, form: str) -> tuple[str, str, int]:
    """The single edit (old, new, position) left after the common prefix and the common suffix are removed."""
    p = 0
    while p < min(len(head), len(form)) and head[p] == form[p]:
        p += 1
    s = 0
    while s < min(len(head), len(form)) - p and head[len(head) - 1 - s] == form[len(form) - 1 - s]:
        s += 1
    return head[p:len(head) - s], form[p:len(form) - s], p


def candidates(head: str, admitted: set) -> set[str]:
    out = set()
    for old, new in admitted:
        if not old:
            out.update(head[:i] + new + head[i:] for i in range(len(head) + 1))
        else:
            start = head.find(old)
            while start != -1:
                out.add(head[:start] + new + head[start + len(old):])
                start = head.find(old, start + 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=GOLD)
    ap.add_argument("--min-count", type=int, default=25)
    a = ap.parse_args()
    d = pd.read_parquet(a.path)
    d = d[d.headword_ok & (d.inflected_forms.map(len) > 0)]
    pairs = [(r.plain, f.replace("-", "").replace("'", "").replace(" ", ""), r.split) for r in d.itertuples() for f in r.inflected_forms]
    pairs = [p for p in pairs if p[1] and p[1] != p[0]]
    train = [p for p in pairs if p[2] == "train"]
    held = [p for p in pairs if p[2] == "heldout"]

    counts = collections.Counter()
    for head, form, _ in train:
        for g in (form, strip_pi(form)):
            if g and g != head:
                old, new, _i = edit(head, g)
                if len(new) <= 5 and len(old) <= 2:
                    counts[(old, new)] += 1
                    break
    admitted = {k for k, c in counts.items() if c >= a.min_count}
    admitted_loose = {(o.translate(LOOSE), n.translate(LOOSE)) for o, n in admitted}

    def explained(head, form, loose=False):
        for g in (form, strip_pi(form)):
            if not g:
                continue
            if g == head:
                return True
            if loose:
                old, new, _i = edit(head.translate(LOOSE), g.translate(LOOSE))
                if (old, new) in admitted_loose:
                    return True
            else:
                old, new, _i = edit(head, g)
                if (old, new) in admitted:
                    return True
        return False

    def rate(rows, loose=False):
        return round(sum(explained(h, f, loose) for h, f, _ in rows) / len(rows), 4)

    sizes = sorted(len(candidates(h, admitted)) for h in sorted({h for h, _, _ in held}))
    position = collections.Counter()
    for head, form, _ in pairs:
        for g in (form, strip_pi(form)):
            if g:
                old, new, i = edit(head, g)
                if not old and new == "wa":
                    position["prefix" if i == 0 else "inside the word"] += 1
                    break
    e_as_c = [(h, f) for h, f, _ in pairs if f.startswith("ć") and h.startswith("e")]
    report = {
        "pairs": {"train": len(train), "heldout": len(held)}, "min_count": a.min_count,
        "admitted": {"inserts": sorted(((n, c) for (o, n), c in counts.items() if not o and c >= a.min_count), key=lambda x: -x[1]),
                     "substitutions": sorted(((f"{o}->{n}", c) for (o, n), c in counts.items() if o and c >= a.min_count), key=lambda x: -x[1])},
        "coverage": {"train": rate(train), "heldout": rate(held), "heldout_ignoring_diacritics": rate(held, True)},
        "candidate_forms_per_heldout_headword": {"median": sizes[len(sizes) // 2], "max": sizes[-1]},
        "position_of_inserted_wa": dict(position),
        "suspected_extraction_error_e_read_as_c_acute": {"count": len(e_as_c), "examples": e_as_c[:8]},
        "heldout_unexplained_examples": [(h, f) for h, f, _ in held if not explained(h, f, True)][:20],
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
