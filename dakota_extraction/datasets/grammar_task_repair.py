"""Repair Dakota grammar-task JSONL so grant-clean affix/exact rewards can fire.

Uses only attested in-repo Dakota (Riggs rule examples, morphology tables,
existing gold). Does not invent forms. Does not change the live rubric.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_ENV_PKG = Path(__file__).resolve().parents[2] / "environments" / "dakota_grammar_translation"
if str(_ENV_PKG) not in sys.path:
    sys.path.insert(0, str(_ENV_PKG))

from dakota_grammar_translation.train_reward import _affix_bearing_tokens  # noqa: E402

PLACEHOLDER_RE = re.compile(
    r"\[TO BE GENERATED|\[explanation\]|\[incorrect form\]|"
    r"\[apply first affix\]|\[with all affixes applied\]",
    re.IGNORECASE,
)
ENGLISH_PAREN_RE = re.compile(r"^(?P<dakota>.+?)\s+\((?P<gloss>[^)]+)\)\s*$")
GRAMMAR_GLOSS_RE = re.compile(
    r"\b(word|verb|noun|tense|prefix|suffix|particle|grammatical|sentence|"
    r"pronoun|adjective|adverb|conjunction|syntax|morphology|phonology|"
    r"plural|singular|animate|inanimate|orthograph)\b",
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

COMPLETE_V1_REL = Path("dakota_rl_training/datasets/grammar_tasks_complete.jsonl")
COMPLETE_V2_REL = Path("dakota_rl_training/datasets/grammar_tasks_complete_v2.jsonl")
HELD_OUT_V1_REL = Path("dakota_rl_training/datasets/grammar_tasks_heldout.jsonl")
HELD_OUT_V2_REL = Path("dakota_rl_training/datasets/grammar_tasks_heldout_v2.jsonl")
REPORT_REL = Path("dakota_rl_training/datasets/splits/REPAIR_REPORT.json")
RULES_REL = Path("data/rl_training_rules/all_rl_rules.json")
PAGE061_REL = Path("data/grammar_test/rl_tasks_page_061.jsonl")
GRAMMAR_COMBINED_REL = Path("data/grammar_extracted/grammar_combined_1-88.json")


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


def is_plausible_affix(affix: str) -> bool:
    token = (affix or "").strip()
    if not token or token == "-" or "/" in token or " " in token:
        return False
    body = token.strip("-")
    if not body or len(body) > 8:
        return False
    if token.startswith("-") and not token.endswith("-"):
        return True
    if token.endswith("-") and not token.startswith("-"):
        return True
    return False


def required_affixes_scorable(gold: str, affixes: Sequence[str]) -> bool:
    if not affixes:
        return False
    return all(bool(_affix_bearing_tokens(gold, affix)) for affix in affixes)


def _fold_letters(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text or "")
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z]+", "", stripped.lower())


def _shares_stem(left: str, right: str) -> bool:
    folded_left = _fold_letters(left)
    folded_right = _fold_letters(right)
    if not folded_left or not folded_right:
        return False
    shorter, longer = sorted((folded_left, folded_right), key=len)
    if len(shorter) >= 3 and shorter in longer:
        return True
    for prefix in ("wo", "ta", "ti", "to", "wa", "wi"):
        if folded_right.startswith(prefix) and len(folded_right) > len(prefix) + 2:
            remainder = folded_right[len(prefix) :]
            if len(folded_left) >= 3 and (
                folded_left in remainder or remainder in folded_left or remainder[:4] in folded_left
            ):
                return True
    overlap = 0
    max_n = min(len(folded_left), len(folded_right))
    for size in range(max_n, 2, -1):
        for start in range(0, len(folded_left) - size + 1):
            chunk = folded_left[start : start + size]
            if chunk in folded_right:
                overlap = size
                break
        if overlap:
            break
    return overlap >= 4


@dataclass
class AttestedLexicon:
    """Dakota surface forms attested in-repo. Never synthesize new ones."""

    forms: set[str] = field(default_factory=set)
    english_to_dakota: dict[str, list[str]] = field(default_factory=dict)

    def add_form(self, dakota: str, english: str | None = None) -> None:
        form = (dakota or "").strip()
        if not form or is_placeholder_gold(form) or looks_like_english_metalanguage(form):
            return
        self.forms.add(form)
        gloss = (english or "").strip()
        if gloss and not is_placeholder_gold(gloss):
            key = gloss.lower()
            bucket = self.english_to_dakota.setdefault(key, [])
            if form not in bucket:
                bucket.append(form)

    def contains(self, text: str) -> bool:
        value = (text or "").strip()
        if not value:
            return False
        if value in self.forms:
            return True
        lowered = value.lower()
        return any(form.lower() == lowered for form in self.forms)

    def dakota_for_english(self, english: str) -> list[str]:
        return list(self.english_to_dakota.get((english or "").strip().lower(), []))

    @classmethod
    def from_example_pairs(cls, pairs: Iterable[Mapping[str, Any]]) -> "AttestedLexicon":
        lexicon = cls()
        for pair in pairs:
            lexicon.add_form(str(pair.get("dakota") or ""), str(pair.get("english") or "") or None)
        return lexicon


def is_dakota_gold(text: str | None, lexicon: AttestedLexicon) -> bool:
    value = (text or "").strip()
    if not value or is_placeholder_gold(value):
        return False
    if looks_like_english_metalanguage(value):
        return False
    if has_dakota_letter(value):
        return True
    stripped = _strip_english_parenthetical(value, lexicon)
    if stripped != value:
        return is_dakota_gold(stripped, lexicon)
    return lexicon.contains(value)


def _strip_english_parenthetical(gold: str, lexicon: AttestedLexicon) -> str:
    match = ENGLISH_PAREN_RE.match((gold or "").strip())
    if not match:
        return gold
    dakota = match.group("dakota").strip()
    gloss = match.group("gloss").strip()
    if not dakota:
        return gold
    if has_dakota_letter(dakota) or lexicon.contains(dakota):
        if gloss.isascii() and (looks_like_english_metalanguage(gloss) or " " in gloss or gloss.isalpha()):
            return dakota
    return gold


def _example_dakota_forms(examples: Sequence[Mapping[str, Any]], lexicon: AttestedLexicon) -> list[str]:
    forms: list[str] = []
    for example in examples:
        dakota = str(example.get("dakota") or "").strip()
        if dakota and is_dakota_gold(dakota, lexicon):
            forms.append(dakota)
    return forms


def _choose_attested_affixed_gold(
    gold: str,
    affixes: Sequence[str],
    info: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
    lexicon: AttestedLexicon,
) -> str | None:
    stripped = _strip_english_parenthetical(gold, lexicon)
    if is_dakota_gold(stripped, lexicon) and required_affixes_scorable(stripped, affixes):
        return stripped

    candidates: list[str] = []
    for field_name in ("target_form", "base_form", "dakota_text"):
        value = str(info.get(field_name) or "").strip()
        if value:
            candidates.append(value)
    candidates.extend(_example_dakota_forms(examples, lexicon))

    affixed: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        form = candidate.strip()
        if not form or form in seen:
            continue
        if not is_dakota_gold(form, lexicon):
            continue
        if any(_affix_bearing_tokens(form, affix) for affix in affixes):
            affixed.append(form)
            seen.add(form)
    if not affixed:
        return None

    paired_base = ""
    gold_key = gold.strip().lower()
    for example in examples:
        if str(example.get("english") or "").strip().lower() == gold_key:
            paired_base = str(example.get("dakota") or "").strip()
            break
    stem_src = paired_base or str(info.get("base_form") or "").strip()

    if stem_src and any(_affix_bearing_tokens(stem_src, affix) for affix in affixes) and stem_src in affixed:
        return stem_src

    if stem_src:
        stemmed = [form for form in affixed if form != stem_src and _shares_stem(stem_src, form)]
        if len(stemmed) == 1:
            return stemmed[0]
        if stemmed:
            return min(stemmed, key=len)

    if len(affixed) == 1:
        return affixed[0]
    return None


def _dakota_from_examples(
    english: str,
    examples: Sequence[Mapping[str, Any]],
    lexicon: AttestedLexicon,
) -> str | None:
    key = (english or "").strip().lower()
    if not key:
        return None
    matches: list[str] = []
    for example in examples:
        if str(example.get("english") or "").strip().lower() != key:
            continue
        dakota = str(example.get("dakota") or "").strip()
        if dakota and is_dakota_gold(dakota, lexicon):
            matches.append(dakota)
    if len(matches) == 1:
        return matches[0]
    lexicon_hits = [form for form in lexicon.dakota_for_english(english) if is_dakota_gold(form, lexicon)]
    unique = list(dict.fromkeys(matches or lexicon_hits))
    if len(unique) == 1:
        return unique[0]
    return None


def repair_row(
    row: Mapping[str, Any],
    lexicon: AttestedLexicon,
    rule_examples: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any] | None:
    """Return a repaired row, or None to drop it. Never invents Dakota."""
    repaired = {
        "prompt": row.get("prompt"),
        "answer": row.get("answer"),
        "info": dict(row.get("info") or {}) if isinstance(row.get("info"), dict) else {},
    }
    for key, value in row.items():
        if key not in repaired:
            repaired[key] = value

    gold = str(repaired.get("answer") or "")
    if is_placeholder_gold(gold) or not gold.strip():
        return None

    info = repaired["info"]
    task = str(info.get("task_type") or "")
    rule_id = str(info.get("rule_id") or "")
    examples = list((rule_examples or {}).get(rule_id) or [])

    affixes = [str(affix) for affix in (info.get("required_affixes") or []) if str(affix).strip()]
    affixes = [affix for affix in affixes if is_plausible_affix(affix)]
    if affixes:
        replacement = _choose_attested_affixed_gold(gold, affixes, info, examples, lexicon)
        if replacement:
            gold = replacement
        kept = [affix for affix in affixes if _affix_bearing_tokens(gold, affix)]
        if not is_dakota_gold(gold, lexicon):
            kept = []
        info["required_affixes"] = kept
        repaired["answer"] = gold
    elif "required_affixes" in info:
        info["required_affixes"] = []

    gold = _strip_english_parenthetical(str(repaired.get("answer") or ""), lexicon)
    repaired["answer"] = gold
    if info.get("required_affixes"):
        info["required_affixes"] = [
            affix for affix in info["required_affixes"] if _affix_bearing_tokens(gold, affix)
        ]

    if task == "reverse_translation":
        if is_dakota_gold(gold, lexicon):
            return repaired
        for hint in (info.get("english_text"), gold):
            replacement = _dakota_from_examples(str(hint or ""), examples, lexicon)
            if replacement:
                repaired["answer"] = replacement
                return repaired
        return None

    source = str(info.get("dakota_text") or info.get("base_form") or "")
    if task in {"word_translation", "morphology"} and source and looks_like_english_metalanguage(source):
        return None

    return repaired


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rule_examples(rules_path: Path) -> dict[str, list[dict[str, str]]]:
    if not rules_path.is_file() or rules_path.read_text(encoding="utf-8").startswith("version https://git-lfs"):
        return {}
    payload = json.loads(rules_path.read_text(encoding="utf-8"))
    rules = payload.get("rules") if isinstance(payload, dict) else payload
    examples: dict[str, list[dict[str, str]]] = {}
    for rule in rules or []:
        rule_id = str(rule.get("rule_id") or "")
        if not rule_id:
            continue
        pairs = []
        for example in rule.get("positive_examples") or []:
            dakota = str(example.get("dakota") or "").strip()
            english = str(example.get("english") or "").strip()
            if dakota:
                pairs.append({"dakota": dakota, "english": english})
        if pairs:
            examples[rule_id] = pairs
    return examples


def _collect_grammar_combined_forms(path: Path, lexicon: AttestedLexicon) -> None:
    if not path.is_file() or path.read_text(encoding="utf-8").startswith("version https://git-lfs"):
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    for page in payload.get("pages") or []:
        for rule in page.get("grammar_rules") or []:
            for example in rule.get("examples") or []:
                lexicon.add_form(str(example.get("dakota") or ""), str(example.get("english") or "") or None)
        for term in page.get("linguistic_terms") or []:
            for dakota in term.get("dakota_examples") or []:
                lexicon.add_form(str(dakota or ""))
        for interlinear in page.get("interlinear_texts") or []:
            lines = interlinear.get("dakota_lines") or []
            if lines:
                lexicon.add_form(" ".join(str(line) for line in lines if line))


def load_attested_lexicon(root: Path) -> AttestedLexicon:
    lexicon = AttestedLexicon()
    rule_examples = load_rule_examples(root / RULES_REL)
    for pairs in rule_examples.values():
        for pair in pairs:
            lexicon.add_form(pair["dakota"], pair.get("english"))

    page061 = root / PAGE061_REL
    if page061.is_file() and not page061.read_text(encoding="utf-8").startswith("version https://git-lfs"):
        for row in load_jsonl(page061):
            info = row.get("info") or {}
            answer = str(row.get("answer") or "")
            task = str(info.get("task_type") or "")
            if has_dakota_letter(answer) or (
                task == "morphology" and answer and not looks_like_english_metalanguage(answer)
            ):
                lexicon.add_form(answer, str(info.get("expected_gloss") or "") or None)
            lexicon.add_form(str(info.get("base_form") or ""))

    _collect_grammar_combined_forms(root / GRAMMAR_COMBINED_REL, lexicon)

    complete = root / COMPLETE_V1_REL
    if complete.is_file():
        for row in load_jsonl(complete):
            info = row.get("info") or {}
            for field_name in ("target_form", "dakota_text", "base_form"):
                lexicon.add_form(str(info.get(field_name) or ""))
            gold = str(row.get("answer") or "")
            if has_dakota_letter(gold) and not is_placeholder_gold(gold):
                lexicon.add_form(_strip_english_parenthetical(gold, lexicon))
    return lexicon


def quality_counts(rows: Sequence[Mapping[str, Any]], lexicon: AttestedLexicon | None = None) -> dict[str, int]:
    nonempty_affix = 0
    any_affix_in_gold = 0
    fully_scorable_affix = 0
    placeholder = 0
    reverse_total = 0
    reverse_dakota = 0
    reverse_marker = 0
    for row in rows:
        info = row.get("info") or {}
        gold = str(row.get("answer") or "")
        affixes = [str(affix) for affix in (info.get("required_affixes") or []) if str(affix).strip()]
        if affixes:
            nonempty_affix += 1
            hits = sum(1 for affix in affixes if _affix_bearing_tokens(gold, affix))
            if hits:
                any_affix_in_gold += 1
            if required_affixes_scorable(gold, affixes):
                fully_scorable_affix += 1
        if is_placeholder_gold(gold):
            placeholder += 1
        if info.get("task_type") == "reverse_translation":
            reverse_total += 1
            if has_dakota_letter(gold):
                reverse_marker += 1
            if lexicon is not None and is_dakota_gold(gold, lexicon):
                reverse_dakota += 1
            elif lexicon is None and has_dakota_letter(gold):
                reverse_dakota += 1
    return {
        "rows": len(rows),
        "nonempty_required_affixes": nonempty_affix,
        "scorable_affix_rows": any_affix_in_gold,
        "fully_scorable_affix_rows": fully_scorable_affix,
        "placeholder_rows": placeholder,
        "reverse_translation_rows": reverse_total,
        "reverse_translation_dakota_gold": reverse_dakota,
        "reverse_translation_marker_gold": reverse_marker,
    }


def repair_rows(
    rows: Sequence[Mapping[str, Any]],
    lexicon: AttestedLexicon,
    rule_examples: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    for row in rows:
        item = repair_row(row, lexicon, rule_examples=rule_examples)
        if item is not None:
            repaired.append(item)
    return repaired


def _append_page061(rows: list[dict[str, Any]], page061_path: Path, lexicon: AttestedLexicon) -> int:
    if not page061_path.is_file() or page061_path.read_text(encoding="utf-8").startswith("version https://git-lfs"):
        return 0
    seen = {(str(row.get("prompt") or "").strip(), str(row.get("answer") or "").strip()) for row in rows}
    added = 0
    for row in load_jsonl(page061_path):
        repaired = repair_row(row, lexicon)
        if repaired is None:
            continue
        key = (str(repaired.get("prompt") or "").strip(), str(repaired.get("answer") or "").strip())
        if key in seen:
            continue
        rows.append(repaired)
        seen.add(key)
        added += 1
    return added


def repair_dataset(root: Path) -> dict[str, Any]:
    lexicon = load_attested_lexicon(root)
    rule_examples = load_rule_examples(root / RULES_REL)
    complete = load_jsonl(root / COMPLETE_V1_REL)
    heldout = load_jsonl(root / HELD_OUT_V1_REL)
    before = quality_counts(complete, lexicon)

    repaired_complete = repair_rows(complete, lexicon, rule_examples=rule_examples)
    page061_added = _append_page061(repaired_complete, root / PAGE061_REL, lexicon)
    repaired_heldout = repair_rows(heldout, lexicon, rule_examples=rule_examples)

    write_jsonl(root / COMPLETE_V2_REL, repaired_complete)
    write_jsonl(root / HELD_OUT_V2_REL, repaired_heldout)

    after = quality_counts(repaired_complete, lexicon)
    after_heldout = quality_counts(repaired_heldout, lexicon)
    report = {
        "source_complete": str(COMPLETE_V1_REL),
        "repaired_complete": str(COMPLETE_V2_REL),
        "frozen_heldout_v1": str(HELD_OUT_V1_REL),
        "repaired_heldout_v2": str(HELD_OUT_V2_REL),
        "heldout_v1_untouched": True,
        "page061_rows_appended": page061_added,
        "before": before,
        "after": after,
        "heldout_v2": after_heldout,
        "notes": [
            "Empty required_affixes still score 0.0 on the live rubric.",
            "Holdout v1 is the cebp9acs eval set and was not overwritten.",
            "Holdout v2 applies the same repair to the v1 rows (seed 42 split identity).",
            "No Dakota forms were invented; gold rewrites use attested in-repo strings only.",
        ],
    }
    report_path = root / REPORT_REL
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
