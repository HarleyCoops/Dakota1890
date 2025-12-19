from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .ingest import load_source_documents


class RuleInductionError(RuntimeError):
    pass


@dataclass(frozen=True)
class GrammarRule:
    rule_id: str
    rule_type: str
    pattern: str
    constraints: str
    positive_examples: List[Dict[str, str]]
    negative_examples: List[Dict[str, str]]
    difficulty: str
    confidence: float
    source_oare_ids: List[str]


@dataclass
class InductionConfig:
    model: str = "gpt-5.2-codex"
    temperature: float = 0.2
    max_output_tokens: int = 1200
    reasoning_effort: str = "high"
    chunk_size: int = 25
    max_docs: int = -1
    seed: int = 42
    dry_run: bool = False
    output_dir: Path = Path("Akkadian/data/grammar_rules")
    save_raw: bool = True


SYSTEM_PROMPT = (
    "You are an Assyriologist and computational linguist. "
    "Induce explicit Old Assyrian grammar rules from transliteration samples. "
    "Focus on morphology, determinatives, logograms, and formulaic syntax."
)

USER_PROMPT_TEMPLATE = """
You are given Old Assyrian transliteration samples. Induce grammar rules that are
explicit enough to be used as verification or reward functions for RL training.

Return ONLY valid JSON (no markdown) as a JSON array. Each element must follow
this schema:

{
  "rule_type": "morphology|determinative|logogram|syntax|formula",
  "pattern": "short description or regex-like pattern",
  "constraints": "clear constraints or exceptions",
  "positive_examples": [
    {"transliteration": "...", "note": "..."}
  ],
  "negative_examples": [
    {"transliteration": "...", "note": "..."}
  ],
  "difficulty": "easy|medium|hard",
  "confidence": 0.0
}

Guidelines:
- Use hyphenation to identify affixes and morpheme boundaries.
- Use determinatives in {curly braces} as grammatical signals.
- Treat ALL CAPS tokens as logograms.
- Prefer rules that can be checked mechanically (regex or substring checks).
- Provide at most 12 rules for this batch.

Transliteration samples:
{samples}
""".strip()


def _ensure_model_allowed(model: str) -> None:
    if not model.startswith("gpt-5.2"):
        raise RuleInductionError("Only gpt-5.2 models are allowed for rule induction.")


def _load_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuleInductionError(
            "openai package not installed. Add it to requirements or pip install openai."
        ) from exc
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuleInductionError("OPENAI_API_KEY is not set in the environment.")
    return OpenAI(api_key=api_key)


def _iter_document_index(path: Path, max_docs: int) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_docs > 0 and len(records) >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(record)
    return records


def _sample_records(records: List[Dict[str, str]], max_docs: int, seed: int) -> List[Dict[str, str]]:
    if max_docs <= 0 or max_docs >= len(records):
        return records
    random.Random(seed).shuffle(records)
    return records[:max_docs]


def _format_samples(records: Sequence[Dict[str, str]]) -> str:
    lines: List[str] = []
    for idx, record in enumerate(records, start=1):
        transliteration = record.get("transliteration") or record.get("transliteration_orig") or ""
        if not transliteration:
            continue
        ore = record.get("oare_id", "")
        cdli = record.get("cdli_id", "")
        genre = record.get("genre_label", "")
        meta = " | ".join(x for x in [ore, cdli, genre] if x)
        lines.append(f"[{idx}] {meta}\n{transliteration}")
    return "\n\n".join(lines)


def _parse_json_array(text: str) -> List[Dict[str, object]]:
    text = text.strip()
    if not text:
        return []
    if not text.startswith("["):
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuleInductionError(f"Failed to parse JSON from model output: {exc}") from exc
    if not isinstance(data, list):
        raise RuleInductionError("Model output is not a JSON array.")
    return data


def induce_rules_from_documents(
    document_index: Path,
    config: Optional[InductionConfig] = None,
) -> List[GrammarRule]:
    config = config or InductionConfig()
    _ensure_model_allowed(config.model)

    records = _iter_document_index(document_index, max_docs=-1)
    records = _sample_records(records, config.max_docs, config.seed)

    chunks: List[List[Dict[str, str]]] = []
    chunk: List[Dict[str, str]] = []
    for record in records:
        if len(chunk) >= config.chunk_size:
            chunks.append(chunk)
            chunk = []
        chunk.append(record)
    if chunk:
        chunks.append(chunk)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = config.output_dir / "raw"
    if config.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    client = None if config.dry_run else _load_openai_client()

    all_rules: List[GrammarRule] = []
    for idx, chunk_records in enumerate(chunks, start=1):
        sample_text = _format_samples(chunk_records)
        prompt = USER_PROMPT_TEMPLATE.format(samples=sample_text)

        if config.dry_run:
            prompt_path = config.output_dir / f"prompt_chunk_{idx:04d}.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            continue

        request_kwargs = {
            "model": config.model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens,
        }
        if config.reasoning_effort:
            request_kwargs["reasoning"] = {"effort": config.reasoning_effort}

        response = client.responses.create(**request_kwargs)
        output_text = getattr(response, "output_text", None)
        if output_text is None:
            output_text = "".join(
                part.text
                for item in response.output
                for part in item.content
                if hasattr(part, "text")
            )

        if config.save_raw:
            raw_path = raw_dir / f"response_chunk_{idx:04d}.txt"
            raw_path.write_text(output_text or "", encoding="utf-8")

        rules_payload = _parse_json_array(output_text or "")
        for jdx, rule in enumerate(rules_payload, start=1):
            rule_id = f"akk_rule_{idx:04d}_{jdx:03d}"
            source_ids = [r.get("oare_id", "") for r in chunk_records if r.get("oare_id")]
            grammar_rule = GrammarRule(
                rule_id=rule_id,
                rule_type=str(rule.get("rule_type", "")).strip(),
                pattern=str(rule.get("pattern", "")).strip(),
                constraints=str(rule.get("constraints", "")).strip(),
                positive_examples=list(rule.get("positive_examples", []) or []),
                negative_examples=list(rule.get("negative_examples", []) or []),
                difficulty=str(rule.get("difficulty", "medium")).strip(),
                confidence=float(rule.get("confidence", 0.0)),
                source_oare_ids=source_ids,
            )
            all_rules.append(grammar_rule)

        time.sleep(0.5)

    return all_rules


def save_rules(rules: Sequence[GrammarRule], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for rule in rules:
            handle.write(json.dumps(rule.__dict__, ensure_ascii=False) + "\n")


def run_induction(
    source_root: Path,
    out_path: Path,
    config: Optional[InductionConfig] = None,
) -> int:
    paths = load_source_documents(source_root)
    document_index = Path("Akkadian/data/document_index.jsonl")
    if not document_index.exists():
        raise RuleInductionError(
            "document_index.jsonl not found. Run build_document_index.py first."
        )

    rules = induce_rules_from_documents(document_index, config)
    if config and config.dry_run:
        return 0

    save_rules(rules, out_path)
    return len(rules)
