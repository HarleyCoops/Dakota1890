#!/usr/bin/env python3
"""Submit an OpenAI Batch job for grammar rule induction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from openai import OpenAI


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit OpenAI Batch job")
    parser.add_argument(
        "--batch-file",
        type=Path,
        default=Path("Akkadian/data/grammar_rules/batch_requests.jsonl"),
        help="Input batch JSONL file",
    )
    parser.add_argument(
        "--status-out",
        type=Path,
        default=Path("Akkadian/batch_status.json"),
        help="Where to write batch metadata",
    )
    parser.add_argument(
        "--completion-window",
        type=str,
        default="24h",
        help="Batch completion window",
    )

    args = parser.parse_args()

    load_dotenv(dotenv_path=Path(".env"))
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found in .env")
        return 1

    if not args.batch_file.exists():
        print(f"Batch file not found: {args.batch_file}")
        return 1

    client = OpenAI(api_key=api_key)

    with args.batch_file.open("rb") as handle:
        file_obj = client.files.create(file=handle, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window=args.completion_window,
    )

    total_requests = _count_lines(args.batch_file)
    payload = {
        "batch_id": batch.id,
        "input_file_id": file_obj.id,
        "status": batch.status,
        "completion_window": args.completion_window,
        "total_requests": total_requests,
        "batch_file": str(args.batch_file),
    }

    args.status_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Submitted batch {batch.id} with {total_requests} requests")
    print(f"Wrote status to {args.status_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
