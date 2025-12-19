from __future__ import annotations

import argparse
import json
from pathlib import Path

from dakota_extraction.schemas.dictionary_schema import DictionaryEntry, validate_entry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate extracted page JSONs against DictionaryEntry schema.")
    parser.add_argument(
        "--extracted",
        type=str,
        default="data/extracted_qwen3vl_tinker",
        help="Directory containing page_*.json files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    extracted_dir = Path(args.extracted)
    paths = sorted(extracted_dir.glob("page_*.json"))
    if not paths:
        print(f"No page_*.json files found in {extracted_dir}")
        return 1

    page_count = 0
    entry_count = 0
    invalid_entries = 0
    hard_errors = 0

    for path in paths:
        page_count += 1
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            hard_errors += 1
            print(f"ERROR {path}: invalid JSON ({exc})")
            continue

        entries = data.get("entries", [])
        for entry_dict in entries:
            entry_count += 1
            try:
                entry = DictionaryEntry(**entry_dict)
                ok, issues = validate_entry(entry)
                if not ok:
                    invalid_entries += 1
                    print(f"WARNING {path.name} {entry.entry_id}: " + "; ".join(issues))
            except Exception as exc:
                hard_errors += 1
                print(f"ERROR {path.name}: could not parse entry ({exc})")

    print(f"Validated {page_count} pages, {entry_count} entries")
    print(f"Invalid entries: {invalid_entries}")
    print(f"Hard errors: {hard_errors}")
    return 0 if hard_errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

