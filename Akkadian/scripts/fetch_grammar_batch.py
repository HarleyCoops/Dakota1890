#!/usr/bin/env python3
"""Fetch batch results and write rules.jsonl."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SCRIPT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from akkadian.rule_induction import GrammarRule, _extract_response_text, _parse_json_array


def _load_status(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Batch status not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_rules(rules, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for rule in rules:
            handle.write(json.dumps(rule.__dict__, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch OpenAI Batch results")
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("Akkadian/batch_status.json"),
        help="Batch status JSON",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("Akkadian/data/grammar_rules/batch_output.jsonl"),
        help="Raw batch output JSONL path",
    )
    parser.add_argument(
        "--rules-out",
        type=Path,
        default=Path("Akkadian/data/grammar_rules/rules.jsonl"),
        help="Final rules JSONL path",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow parsing even if batch is not completed",
    )

    args = parser.parse_args()

    status = _load_status(args.status)
    batch_id = status.get("batch_id")
    if not batch_id:
        print("batch_id not found in status file")
        return 1

    load_dotenv(dotenv_path=Path(".env"))
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found in .env")
        return 1

    client = OpenAI(api_key=api_key)
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed" and not args.allow_incomplete:
        print(f"Batch status: {batch.status}. Use --allow-incomplete to parse anyway.")
        return 0

    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        print("Batch has no output_file_id yet.")
        return 1

    content = client.files.content(output_file_id)
    if hasattr(content, "read"):
        content = content.read()
    elif not isinstance(content, (bytes, bytearray)):
        content = bytes(content)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_bytes(content)

    rules = []
    with args.output_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            response = payload.get("response")
            if not response:
                continue
            output_text = _extract_response_text(response)
            if not output_text:
                continue
            try:
                rules_payload = _parse_json_array(output_text)
            except Exception:
                continue
            custom_id = payload.get("custom_id", "chunk")
            for idx, rule in enumerate(rules_payload, start=1):
                rule_id = f"{custom_id}_{idx:03d}"
                rules.append(
                    GrammarRule(
                        rule_id=rule_id,
                        rule_type=str(rule.get("rule_type", "")).strip(),
                        pattern=str(rule.get("pattern", "")).strip(),
                        constraints=str(rule.get("constraints", "")).strip(),
                        positive_examples=list(rule.get("positive_examples", []) or []),
                        negative_examples=list(rule.get("negative_examples", []) or []),
                        difficulty=str(rule.get("difficulty", "medium")).strip(),
                        confidence=float(rule.get("confidence", 0.0)),
                        source_oare_ids=[],
                    )
                )

    _write_rules(rules, args.rules_out)
    print(f"Wrote {len(rules)} rules to {args.rules_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
