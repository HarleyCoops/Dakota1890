"""Executable rule checker v0 for Riggs (1890) Dakota forms.

Grounded on lexgym/orthography.py from HarleyCoops/Dakota1890 — the alphabet,
variant-glyph map and canonicalization below are that module's, so this checker
agrees with the graph builder by construction.

v0 checks (no compiled morphology yet — the affix inventory is Milestone 2):
  1. alphabet   — every character is in the Riggs alphabet
  2. shape      — a hyphenated form parses as syllables + one accent mark
  3. attested   — the form is a headword, an entry's own inflected form,
                  or another attested form in the same family
  4. ng-variant — the form matches an attested form except for n/ŋ
                  (1,494 known extraction suspects; reported, never correct)
  5. relation   — how an unattested form relates to its headword stem:
                  prefix / suffix / contained / other. "other" is the triage
                  list; the sampled misses there were mostly extraction errors.

Run as an audit:
    python checker.py --audit --push
loads HarleyCooper/riggs1890-lexicon-graph (config dakota_entries), prints the
counts, and pushes triage_unexplained_v0.parquet to
HarleyCooper/riggs1890-verifier-loop (needs HF_TOKEN in the environment).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Orthography constants — copied from lexgym/orthography.py (Dakota1890).
# Each mapping is a hypothesis about the extraction, not about the language.
# ---------------------------------------------------------------------------
VARIANT_MAP = {
    "š": "ś", "ṡ": "ś", "ș": "ś",
    "ž": "ź", "ż": "ź",
    "η": "ŋ", "ƞ": "ŋ", "ṅ": "ŋ", "ń": "ŋ", "ņ": "ŋ",
    "ǧ": "ġ", "ğ": "ġ", "ǵ": "ġ",
    "ȟ": "ḣ", "ħ": "ḣ",
    "č": "ć", "ċ": "ć",
    "é": "ć",
    "ţ": "ṭ", "ț": "ṭ", "ṫ": "ṭ",
    "ķ": "ḳ",
}
ALPHABET = set("abcdeghikmnopstuwyz" + "ćġḣŋśźṭḳṗ")
ACCENT_MARKS = """'’ʹ´ˊ`′"""
_ACCENT_RE = re.compile("[" + re.escape(ACCENT_MARKS) + "]")
_TRANS = str.maketrans(VARIANT_MAP)


def canon(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "").lower().translate(_TRANS)
    t = _ACCENT_RE.sub("'", t)
    return re.sub(r"\s+", " ", t).strip()


def plain(text: str) -> str:
    return canon(text).replace("-", "").replace("'", "")


def is_suspect(text: str) -> bool:
    return any(ch not in ALPHABET and ch != " " for ch in plain(text))


def bad_chars(text: str) -> list[str]:
    return sorted({ch for ch in plain(text) if ch not in ALPHABET and ch != " "})


def parse_syllables(headword: str) -> dict | None:
    """Split a hyphenated form like a-hi'-yo-ka-sin. None if untrustworthy."""
    c = canon(headword)
    if " " in c or not c or is_suspect(c):
        return None
    parts = c.split("-")
    syllables, stress = [], None
    for i, part in enumerate(parts):
        if part.count("'") > 1:
            return None
        if "'" in part:
            if not part.endswith("'") or stress is not None:
                return None
            stress = i
            part = part[:-1]
        if not part:
            return None
        syllables.append(part)
    if stress is None:
        if len(syllables) > 1:
            return None
        stress = 0
    return {"syllables": syllables, "stress": stress}


def loose(text: str) -> str:
    """Plain form with the n/ŋ distinction removed (reporting only)."""
    return plain(text).replace("ŋ", "n").replace(" ", "")


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------
@dataclass
class Lexicon:
    """Indexes built from the dakota_entries config."""

    headword_to_entry: dict = field(default_factory=dict)   # plain -> entry dict
    attested_forms: set = field(default_factory=set)        # plain forms
    form_to_entries: dict = field(default_factory=dict)     # plain -> [entry_id]
    loose_index: dict = field(default_factory=dict)         # loose -> set of plain
    family_members: dict = field(default_factory=dict)      # family -> set of plain


def build_lexicon(rows) -> Lexicon:
    lex = Lexicon()
    for r in rows:
        hw = plain(r["headword"])
        if hw and not is_suspect(r["headword"]):
            lex.headword_to_entry[hw] = r
            lex.attested_forms.add(hw)
            lex.form_to_entries.setdefault(hw, []).append(r["entry_id"])
            lex.loose_index.setdefault(loose(hw), set()).add(hw)
        fam = lex.family_members.setdefault(r["family"], set())
        for f in [r["headword"], *r["inflected_forms"]]:
            p = plain(f)
            if p:
                fam.add(p)
                if not is_suspect(f):
                    lex.attested_forms.add(p)
                    lex.form_to_entries.setdefault(p, []).append(r["entry_id"])
                    lex.loose_index.setdefault(loose(p), set()).add(p)
    return lex


def relation(form_plain: str, head_plain: str) -> str:
    """How an unattested form relates to its headword stem."""
    if not head_plain:
        return "no-stem"
    if form_plain == head_plain:
        return "identity"
    if form_plain.startswith(head_plain):
        return "prefix"
    if form_plain.endswith(head_plain):
        return "suffix"
    if head_plain in form_plain:
        return "contained"
    return "other"


def check_form(form: str, lex: Lexicon, family: str | None = None) -> dict:
    """Label one candidate form. Returns verdict + reasons. No soft scores."""
    reasons = {}
    p = plain(form)
    reasons["plain"] = p
    if is_suspect(form):
        reasons["alphabet_violation"] = bad_chars(form)
    if "-" in canon(form):
        parsed = parse_syllables(form)
        if parsed is None:
            reasons["shape_violation"] = "hyphenated form does not parse as syllables + one accent"
    if p in lex.attested_forms:
        reasons["verdict"] = "attested"
        reasons["entries"] = lex.form_to_entries.get(p, [])[:5]
        return reasons
    if family and p in lex.family_members.get(family, set()):
        reasons["verdict"] = "attested"
        reasons["note"] = "attested within the same word family"
        return reasons
    if is_suspect(form):
        reasons["verdict"] = "rejected"
        return reasons
    hit = lex.loose_index.get(loose(p))
    if hit and p not in hit:
        reasons["ng_variant_of"] = sorted(hit)[:5]
        reasons["verdict"] = "ng-variant"
        return reasons
    stem = None
    # find the headword this form was proposed against, if caller supplied family
    if family:
        for member in lex.family_members.get(family, set()):
            if member in lex.headword_to_entry:
                stem = member
                break
    reasons["relation"] = relation(p, stem) if stem else "no-stem"
    reasons["verdict"] = "licensed-pending-rules" if reasons["relation"] in (
        "prefix", "suffix", "contained") else "rejected"
    if reasons["verdict"] == "rejected":
        reasons["triage"] = "unexplained relative to every known headword stem"
    return reasons


# ---------------------------------------------------------------------------
# Audit: run the checker against the published gold
# ---------------------------------------------------------------------------
def run_audit(push: bool) -> dict:
    from datasets import load_dataset

    ds = load_dataset("HarleyCooper/riggs1890-lexicon-graph", "dakota_entries", split="train")
    rows = list(ds)
    # Headwords-only index: attested = the plain form is a headword anywhere.
    headwords: set = set()
    loose_headwords: dict = {}   # loose(plain headword) -> set of plain headwords
    for r in rows:
        hw = plain(r["headword"])
        if hw and not is_suspect(r["headword"]):
            headwords.add(hw)
            loose_headwords.setdefault(loose(hw), set()).add(hw)

    counts = collections.Counter()
    triage = []
    examples = collections.defaultdict(list)
    rel_dist_unexplained = collections.Counter()

    for r in rows:
        head = plain(r["headword"])
        for f in r["inflected_forms"]:
            p = plain(f)
            if not p:
                counts["empty_form"] += 1
                continue
            counts["total_forms"] += 1
            # One verdict per instance, in precedence order:
            # attested-headword > ng-variant > alphabet-violation > relation.
            if p in headwords:
                verdict = "attested-headword"
            elif loose(p) in loose_headwords:
                verdict = "ng-variant"
            elif bad_chars(f):
                verdict = "alphabet-violation"
            else:
                rel = relation(p, head)
                verdict = ("licensed-pending" if rel in ("prefix", "suffix",
                                                          "contained")
                           else "unexplained")
            counts[verdict] += 1
            if verdict == "unexplained":
                rel_dist_unexplained[rel] += 1
            if len(examples[verdict]) < 5:
                examples[verdict].append(
                    {"form": f, "headword": r["headword"], "gloss": r["gloss"],
                     "relation": rel if verdict in ("licensed-pending",
                                                    "unexplained") else ""})
            if verdict in ("unexplained", "ng-variant", "alphabet-violation"):
                triage.append({
                    "entry_id": r["entry_id"], "page": r["page"],
                    "source_image": r["source_image"],
                    "form_raw": f, "form_plain": p,
                    "headword_raw": r["headword"], "headword_plain": head,
                    "gloss": r["gloss"], "family": r["family"],
                    "verdict": verdict,
                    "relation": relation(p, head),
                    "confidence": r["confidence"],
                })

    summary = {
        "entries": len(rows),
        "entries_with_inflected_forms": sum(1 for r in rows if r["inflected_forms"]),
        "inflected_form_instances": counts["total_forms"],
        "verdicts": {k: v for k, v in counts.items()
                     if k in ("attested-headword", "ng-variant",
                              "licensed-pending", "unexplained",
                              "alphabet-violation")},
        "relation_distribution_unexplained": dict(rel_dist_unexplained),
        "triage_rows": len(triage),
        "examples": dict(examples),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    if push:
        import pandas as pd
        from huggingface_hub import HfApi
        df = pd.DataFrame(triage, columns=[
            "entry_id", "page", "source_image", "form_raw", "form_plain",
            "headword_raw", "headword_plain", "gloss", "family", "verdict",
            "relation", "confidence"])
        path = "triage_unexplained_v0.parquet"
        df.to_parquet(path)
        repo_id = "HarleyCooper/riggs1890-verifier-loop"
        HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
            path_or_fileobj=path, path_in_repo=path,
            repo_id=repo_id, repo_type="dataset",
            commit_message=f"checker v0 triage: {len(df)} unexplained forms")
        print(f"pushed {len(df)} triage rows to {repo_id}/{path}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--form", help="check a single form interactively")
    args = ap.parse_args()
    if args.audit or args.push or (not args.form and not args.audit and not args.push):
        # default action when run with no flags (job runners pass no args):
        # full audit against the published gold, pushing the triage parquet.
        run_audit(push=True if (args.push or not args.audit) else False)
    elif args.form:
        from datasets import load_dataset
        rows = load_dataset("HarleyCooper/riggs1890-lexicon-graph",
                            "dakota_entries", split="train")
        lex = build_lexicon(rows)
        print(json.dumps(check_form(args.form, lex), indent=2, ensure_ascii=False))
