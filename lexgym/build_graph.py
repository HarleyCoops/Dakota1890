"""Build the lexicon graph from per-page extraction JSON.

Input : data/extracted/page_NNN.json (Riggs 1890, both halves of the dictionary)
Output: data/lexicon_graph/{dakota_entries,english_entries,edges}.jsonl + report.json

Every row keeps entry_id, page and source_image, so each fact can be checked
against the scan.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import re
from pathlib import Path

from .orthography import canon, is_suspect, loose, parse_headword, plain

ENGLISH_HINT = re.compile(
    r"\b(the|to|of|and|or|as|is|he|she|it|they|i|we|you|with|for|not|one|that|this|see|from|in|on|a|an|be|by|at)\b"
)
POS_MAP = {"v.a.": "v. a.", "v.n.": "v. n.", "inj.": "intj.", "interj.": "intj."}
HELDOUT_BUCKETS = 10  # one family in ten is held out


def _clean_dakota(form: str) -> str | None:
    """Canonical Dakota form, or None if it looks like English or is suspect."""
    c = canon(form).strip(" .,;:()[]")
    if not c or len(c) > 60 or is_suspect(c) or ENGLISH_HINT.search(c):
        return None
    return c


def _split_refs(value) -> list[str]:
    if not value:
        return []
    items = value if isinstance(value, list) else re.split(r"\band\b|[,;]", str(value))
    return [p for p in (plain(x).strip(" .()") for x in items) if p and not is_suspect(p) and " " not in p]


def _english_equivalents(definition: str) -> list[str]:
    """Dakota equivalents from an English-Dakota entry.

    Riggs writes `apa ; kastaka : beat out as corn, kapa`. Only the part before
    the first colon is unqualified, so only that part is used.
    """
    head = (definition or "").split(":")[0]
    out = []
    for seg in re.split(r"[;,]", head):
        c = _clean_dakota(seg)
        if c and c not in out:
            out.append(c)
    return out


class _UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build(extracted_dir: Path, out_dir: Path, english_start_page: int) -> dict:
    raw = []
    for f in sorted(glob.glob(str(extracted_dir / "page_*.json"))):
        page = int(re.search(r"page_(\d+)", f).group(1))
        data = json.loads(Path(f).read_text(encoding="utf-8-sig"))
        raw += [(page, e) for e in data.get("entries", []) if isinstance(e, dict) and e.get("headword")]

    stats = collections.Counter(raw_entries=len(raw))
    dakota, english = [], []
    for page, e in raw:
        base = {
            "entry_id": e.get("entry_id"),
            "page": page,
            "source_image": e.get("source_image"),
            "confidence": e.get("confidence"),
            "pos": POS_MAP.get((e.get("part_of_speech") or "").strip(), (e.get("part_of_speech") or "").strip()) or None,
        }
        if page >= english_start_page:
            forms = _english_equivalents(e.get("definition_primary") or "")
            english.append({**base, "english": canon(e["headword"]).replace("'", ""), "headword_raw": e["headword"],
                            "definition_raw": e.get("definition_primary"), "dakota_forms": forms})
            stats["english_entries"] += 1
            stats["english_entries_with_forms"] += bool(forms)
            continue
        parsed = parse_headword(e["headword"])
        row = {
            **base,
            "headword_raw": e["headword"],
            "headword": parsed["form"] if parsed else None,
            "plain": parsed["plain"] if parsed else plain(e["headword"]),
            "syllables": parsed["syllables"] if parsed else None,
            "stress": parsed["stress"] if parsed else None,
            "gloss": (e.get("definition_primary") or "").strip() or None,
            "gloss_secondary": (e.get("definition_secondary") or "").strip() or None,
            "inflected_forms": [c for c in (_clean_dakota(x) for x in e.get("inflected_forms") or []) if c and " " not in c],
            "derived_from": _split_refs(e.get("derived_from")),
            "see_also": _split_refs(e.get("see_also")),
            "notes": e.get("grammatical_notes"),
            "headword_ok": parsed is not None,
        }
        dakota.append(row)
        stats["dakota_entries"] += 1
        stats["dakota_headword_ok"] += parsed is not None

    index = collections.defaultdict(list)
    for r in dakota:
        if r["headword_ok"]:
            index[r["plain"]].append(r["entry_id"])

    # families: a headword and everything derived from it stay in one split
    uf = _UnionFind()
    edges = []
    for r in dakota:
        uf.find(r["plain"])
        for kind in ("derived_from", "see_also"):
            for ref in r[kind]:
                resolved = ref in index
                edges.append({"src": r["entry_id"], "src_plain": r["plain"], "kind": kind, "dst_plain": ref,
                              "dst_ids": index.get(ref, []), "resolved": resolved})
                stats[f"edge_{kind}"] += 1
                stats[f"edge_{kind}_resolved"] += resolved
                if kind == "derived_from" and resolved:
                    uf.union(r["plain"], ref)
    family_size = collections.Counter(uf.find(r["plain"]) for r in dakota)
    for r in dakota:
        fam = uf.find(r["plain"])
        r["family"] = fam
        r["family_size"] = family_size[fam]
        bucket = int(hashlib.sha256(fam.encode("utf-8")).hexdigest(), 16) % HELDOUT_BUCKETS
        r["split"] = "heldout" if bucket == 0 else "train"
    split_of = {r["plain"]: r["split"] for r in dakota}

    # the English-Dakota half is a second witness for each Dakota form
    attested = collections.Counter()
    loose_index = collections.defaultdict(list)
    for key in index:
        loose_index[loose(key)].append(key)
    for r in english:
        r["attested_forms"] = [f for f in r["dakota_forms"] if plain(f) in index]
        r["attested_loose"] = [f for f in r["dakota_forms"] if plain(f) not in index and loose(f) in loose_index]
        stats["english_forms_attested_loose_only"] += len(r["attested_loose"])
        r["split"] = "heldout" if any(split_of.get(plain(f)) == "heldout" for f in r["dakota_forms"]) else "train"
        stats["english_forms"] += len(r["dakota_forms"])
        stats["english_forms_attested_in_dakota_half"] += len(r["attested_forms"])
        for f in r["attested_forms"]:
            attested[plain(f)] += 1
            for dst in index[plain(f)]:
                edges.append({"src": r["entry_id"], "src_plain": r["english"], "kind": "english_equivalent",
                              "dst_plain": plain(f), "dst_ids": [dst], "resolved": True})
    for r in dakota:
        r["attested_in_english_half"] = attested.get(r["plain"], 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("dakota_entries", dakota), ("english_entries", english), ("edges", edges)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as h:
            for r in rows:
                h.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats.update(
        families=len(family_size),
        families_multi=sum(1 for v in family_size.values() if v > 1),
        dakota_train=sum(r["split"] == "train" for r in dakota),
        dakota_heldout=sum(r["split"] == "heldout" for r in dakota),
        dakota_with_gloss=sum(bool(r["gloss"]) for r in dakota),
        dakota_with_inflected=sum(bool(r["inflected_forms"]) for r in dakota),
        dakota_attested_twice=sum(r["attested_in_english_half"] > 0 for r in dakota),
        english_start_page=english_start_page,
    )
    report = dict(sorted(stats.items()))
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extracted", default="data/extracted")
    ap.add_argument("--out", default="data/lexicon_graph")
    ap.add_argument("--english-start-page", type=int, default=371,
                    help="First scan page of the English-Dakota half (Riggs 1890: 371).")
    a = ap.parse_args()
    print(json.dumps(build(Path(a.extracted), Path(a.out), a.english_start_page), indent=2))


if __name__ == "__main__":
    main()
