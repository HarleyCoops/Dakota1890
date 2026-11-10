"""Build an Adaptive Data instruction_dataset upload from Dakota1890 v2 JSONL.

Does not call Adaption. Does not invent Dakota forms.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\chris\Dakota1890")
DATASETS = ROOT / "dakota_rl_training" / "datasets"
V2 = DATASETS / "grammar_tasks_complete_v2.jsonl"
HELD = DATASETS / "grammar_tasks_heldout.jsonl"
OUT_PARQUET = DATASETS / "adaption_upload_v2.parquet"
OUT_REPORT = DATASETS / "adaption_upload_v2_report.json"
OUT_SPEC = DATASETS / "adaption_spec_dakota1890.md"

TARGET_MIN = 3000
TARGET_MAX = 4000
SAMPLE_SEED = 42

PLACEHOLDER_RE = re.compile(
    r"\[TO BE GENERATED|\[explanation\]|\[incorrect form\]|"
    r"\[apply first affix\]|\[with all affixes applied\]",
    re.IGNORECASE,
)
ENGLISH_FUNCTION = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "and",
    "or",
    "in",
    "on",
    "for",
    "with",
    "that",
    "this",
    "is",
    "are",
    "was",
    "be",
    "as",
    "by",
    "from",
    "it",
    "word",
    "meaning",
    "used",
    "when",
    "which",
    "their",
    "they",
}
LINGUISTIC_SINGLE = {
    "agreement",
    "antecedent",
    "article",
    "circumfix",
    "conjunction",
    "demonstrative",
    "infix",
    "interjection",
    "morphology",
    "orthography",
    "particle",
    "phonology",
    "prefix",
    "preposition",
    "pronoun",
    "suffix",
    "syntax",
    "tense",
}
ENGLISH_PLACE_PHRASES = {
    "end village",
    "marsh village",
    "village at the end",
    "village of the prairie",
    "second variety",
    "active transitive verbs",
    "aorist tense",
}
GRAMMAR_GLOSS_RE = re.compile(
    r"\b(word|verb|noun|tense|prefix|suffix|particle|grammatical|sentence|"
    r"pronoun|adjective|adverb|conjunction|syntax|morphology|phonology|"
    r"plural|singular|animate|inanimate|orthograph)\b",
    re.IGNORECASE,
)

SPEC_TEXT = """# Dakota1890 Adaption recipe / specification notes

Use this file as recipe notes for an Adaptive Data `instruction_dataset`.
Source gold is repaired Riggs grammar tasks (`grammar_tasks_complete_v2.jsonl` from PR 110).
Do not call paid APIs from this recipe.

## Orthography and gold

- 1890 Riggs Dakota orthography (Dakota-English Dictionary grammar; preserve ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú and other attested marks exactly).
- For Dakota-production tasks, gold must be Dakota.
- Every required_affix must appear in gold.
- Do not replace attested Dakota with an English glossary.
- Do not invent stems.

Dakota-production task types: `reverse_translation` (English → Dakota), `affix_insertion`, and `morphology` when the completion is a Dakota surface form.
Comprehension tasks (`word_translation`, Dakota → English) have English gold by design; do not rewrite those completions into unattested Dakota.

## Columns

- `prompt`: the user / task prompt
- `completion`: gold
- Extra columns Adaption can ignore or we can rejoin later: `task_type`, `difficulty`, `required_affixes` (JSON string), `source_id` (`info.rule_id`; v2 has no native source_id)

## Constraints

- Preserve attested Dakota surface forms from the 1890 grammar exactly.
- Nonempty `required_affixes` are supervision: each affix must occur in the gold completion.
- Empty `required_affixes` is allowed and still scores 0.0 on the live grant-clean rubric.
- Frozen held-out v1 (`grammar_tasks_heldout.jsonl`) is the eval set; do not overwrite it and do not treat it as train gold.
"""


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_placeholder_gold(text: str | None) -> bool:
    return bool(PLACEHOLDER_RE.search(text or ""))


def has_dakota_letter(text: str) -> bool:
    for char in text or "":
        if not char.isalpha():
            continue
        if ord(char) > 127:
            return True
        if unicodedata.category(char) == "Mn":
            return True
    return bool(re.search(r"[\u0300-\u036f]", text or ""))


def looks_like_english_metalanguage(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    if is_placeholder_gold(value):
        return True
    if has_dakota_letter(value):
        return False
    lowered = value.lower()
    if lowered in LINGUISTIC_SINGLE or lowered in ENGLISH_PLACE_PHRASES:
        return True
    tokens = re.findall(r"[A-Za-z']+", value)
    if len(tokens) >= 2 and all(token.isascii() for token in tokens):
        if any(token.lower() in ENGLISH_FUNCTION for token in tokens):
            return True
        if any(token.lower() in {"village", "tense", "verbs", "variety"} for token in tokens):
            return True
        if GRAMMAR_GLOSS_RE.search(value):
            return True
    if len(tokens) == 1 and tokens[0].lower() in LINGUISTIC_SINGLE:
        return True
    return False


def gold_language(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return "empty"
    if is_placeholder_gold(value):
        return "placeholder"
    if has_dakota_letter(value):
        return "dakota_markers"
    if looks_like_english_metalanguage(value):
        return "english_like"
    return "ascii_or_unknown"


def info_of(row: dict) -> dict:
    info = row.get("info")
    return info if isinstance(info, dict) else {}


def task_type_of(row: dict) -> str:
    return str(row.get("task_type") or info_of(row).get("task_type") or "")


def difficulty_of(row: dict) -> str:
    return str(row.get("difficulty") or info_of(row).get("difficulty") or "")


def required_affixes_of(row: dict) -> list[str]:
    raw = info_of(row).get("required_affixes") or row.get("required_affixes") or []
    return [str(affix).strip() for affix in raw if str(affix).strip()]


def source_id_of(row: dict) -> str:
    return str(
        row.get("source_id")
        or row.get("task_id")
        or info_of(row).get("rule_id")
        or ""
    )


def gold_of(row: dict) -> str:
    return str(row.get("answer") or row.get("completion") or "").strip()


def prompt_of(row: dict) -> str:
    return str(row.get("prompt") or "").strip()


def row_key(row: dict) -> tuple[str, str]:
    return (prompt_of(row), gold_of(row))


def affixes_appear_in_gold(gold: str, affixes: list[str]) -> bool:
    lowered = gold.lower()
    for affix in affixes:
        token = affix.lower()
        body = token.strip("-")
        if token and token in lowered:
            continue
        if body and body in lowered:
            continue
        return False
    return True


def sample_tier(row: dict) -> int:
    tt = task_type_of(row)
    gold = gold_of(row)
    dakota_src = str(info_of(row).get("dakota_text") or info_of(row).get("base_form") or "")
    gold_dak = has_dakota_letter(gold)
    gold_eng = looks_like_english_metalanguage(gold)
    src_dak = has_dakota_letter(dakota_src)
    if tt == "morphology" and gold_dak and not gold_eng:
        return 0
    if tt == "morphology" and not gold_eng:
        return 1
    if tt == "word_translation" and src_dak:
        return 2
    if tt == "word_translation":
        return 3
    return 4


def to_record(row: dict) -> dict:
    affixes = required_affixes_of(row)
    gold = gold_of(row)
    return {
        "prompt": prompt_of(row),
        "completion": gold,
        "task_type": task_type_of(row),
        "difficulty": difficulty_of(row),
        "required_affixes": json.dumps(affixes, ensure_ascii=False),
        "source_id": source_id_of(row),
        "gold_language": gold_language(gold),
    }


def main() -> None:
    rows = load_jsonl(V2)
    heldout = load_jsonl(HELD)
    held_prompts = {prompt_of(row) for row in heldout if prompt_of(row)}

    source_counts = Counter(task_type_of(row) for row in rows)
    source_affix = sum(1 for row in rows if required_affixes_of(row))
    source_placeholder = sum(1 for row in rows if is_placeholder_gold(gold_of(row)))

    candidates: list[dict] = []
    excluded_heldout = 0
    excluded_placeholder = 0
    excluded_empty = 0
    for row in rows:
        if not prompt_of(row) or not gold_of(row):
            excluded_empty += 1
            continue
        if is_placeholder_gold(gold_of(row)) or is_placeholder_gold(prompt_of(row)):
            excluded_placeholder += 1
            continue
        if prompt_of(row) in held_prompts:
            excluded_heldout += 1
            continue
        candidates.append(row)

    must: list[dict] = []
    sample_pool: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in candidates:
        tt = task_type_of(row)
        key = row_key(row)
        if key in seen:
            continue
        if tt == "reverse_translation" or required_affixes_of(row) or tt == "affix_insertion":
            must.append(row)
            seen.add(key)
        elif tt in {"morphology", "word_translation"}:
            sample_pool.append(row)

    sample_pool = [row for row in sample_pool if row_key(row) not in seen]
    rng = random.Random(SAMPLE_SEED)
    by_tier: dict[int, list[dict]] = {}
    for row in sample_pool:
        by_tier.setdefault(sample_tier(row), []).append(row)

    # Reach a few thousand without dumping the whole 9k.
    extra_budget = max(0, TARGET_MIN - len(must))
    if extra_budget == 0:
        extra_budget = min(800, max(0, TARGET_MAX - len(must)))
    extra_budget = min(extra_budget, max(0, TARGET_MAX - len(must)))

    extra: list[dict] = []
    for tier in sorted(by_tier):
        bucket = list(by_tier[tier])
        rng.shuffle(bucket)
        take = min(len(bucket), extra_budget - len(extra))
        extra.extend(bucket[:take])
        if len(extra) >= extra_budget:
            break

    selected = must + extra
    records = [to_record(row) for row in selected]

    frame = pd.DataFrame(
        [
            {
                "prompt": rec["prompt"],
                "completion": rec["completion"],
                "task_type": rec["task_type"],
                "difficulty": rec["difficulty"],
                "required_affixes": rec["required_affixes"],
                "source_id": rec["source_id"],
            }
            for rec in records
        ]
    )
    frame.to_parquet(OUT_PARQUET, index=False)

    upload_types = Counter(rec["task_type"] for rec in records)
    upload_gold = Counter(rec["gold_language"] for rec in records)
    gold_by_type: dict[str, Counter[str]] = {}
    for rec in records:
        gold_by_type.setdefault(rec["task_type"], Counter())[rec["gold_language"]] += 1

    affix_rows = 0
    affix_in_gold = 0
    for row in selected:
        affixes = required_affixes_of(row)
        if not affixes:
            continue
        affix_rows += 1
        if affixes_appear_in_gold(gold_of(row), affixes):
            affix_in_gold += 1

    OUT_SPEC.write_text(SPEC_TEXT, encoding="utf-8")

    report = {
        "output_path": str(OUT_PARQUET),
        "output_size_bytes": OUT_PARQUET.stat().st_size,
        "spec_path": str(OUT_SPEC),
        "report_path": str(OUT_REPORT),
        "local_v2_present": False,
        "v2_source": "fetched",
        "v2_fetch": {
            "method": "raw GitHub",
            "url": "https://raw.githubusercontent.com/HarleyCoops/Dakota1890/main/dakota_rl_training/datasets/grammar_tasks_complete_v2.jsonl",
            "sha": "8d481baae0594109810f7321cb57c03028558db8",
            "bytes": 4051839,
            "pr": 110,
            "merge_commit": "5e14924c3d4233cc0bc59d661ecf0cd410facd6c",
        },
        "heldout_v1_local": True,
        "heldout_v1_path": str(HELD),
        "heldout_v1_rows": len(heldout),
        "source_v2_rows": len(rows),
        "source_v2_task_type_counts": dict(source_counts),
        "source_v2_nonempty_required_affixes": source_affix,
        "source_v2_placeholder_rows": source_placeholder,
        "excluded_empty": excluded_empty,
        "excluded_placeholder": excluded_placeholder,
        "excluded_heldout_v1_prompt_overlap": excluded_heldout,
        "selection": {
            "all_reverse_translation": sum(1 for row in must if task_type_of(row) == "reverse_translation"),
            "all_nonempty_required_affixes": sum(1 for row in must if required_affixes_of(row)),
            "all_affix_insertion": sum(1 for row in must if task_type_of(row) == "affix_insertion"),
            "must_include_rows": len(must),
            "sampled_morphology_word_translation": len(extra),
            "sampled_by_task_type": dict(Counter(task_type_of(row) for row in extra)),
            "sample_seed": SAMPLE_SEED,
            "target_min": TARGET_MIN,
            "target_max": TARGET_MAX,
        },
        "upload_rows": len(records),
        "upload_task_type_counts": dict(upload_types),
        "upload_gold_language_heuristic": dict(upload_gold),
        "upload_gold_language_by_task_type": {key: dict(val) for key, val in gold_by_type.items()},
        "upload_nonempty_required_affixes": affix_rows,
        "upload_required_affix_appears_in_gold_heuristic": affix_in_gold,
        "columns": [
            "prompt",
            "completion",
            "task_type",
            "difficulty",
            "required_affixes",
            "source_id",
        ],
        "notes": [
            "source_id is info.rule_id; v2 has no source_id/task_id field.",
            "Frozen heldout v1 prompts were excluded to protect the cebp9acs eval set.",
            "No Dakota forms were invented. No Adaption API call was made.",
        ],
    }
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "upload_rows": len(records),
        "task_types": dict(upload_types),
        "gold": dict(upload_gold),
        "must": len(must),
        "extra": len(extra),
        "excluded_heldout": excluded_heldout,
        "path": str(OUT_PARQUET),
        "size": OUT_PARQUET.stat().st_size,
        "report": str(OUT_REPORT),
        "spec": str(OUT_SPEC),
    }, indent=2))


if __name__ == "__main__":
    main()
