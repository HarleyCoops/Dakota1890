#!/usr/bin/env python3
"""Build OpenAI Batch JSONL requests for grammar rule induction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SCRIPT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from akkadian.rule_induction import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    SAMPLES_TOKEN,
    _format_samples,
    _iter_document_index,
    _sample_records,
)


def _chunk_records(records, chunk_size):
    chunk = []
    for record in records:
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
        chunk.append(record)
    if chunk:
        yield chunk


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OpenAI Batch JSONL requests")
    parser.add_argument(
        "--document-index",
        type=Path,
        default=Path("Akkadian/data/document_index.jsonl"),
        help="Path to document_index.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Akkadian/data/grammar_rules/batch_requests.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--max-docs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="gpt-5.2")
    parser.add_argument("--max-output-tokens", type=int, default=15000)
    parser.add_argument("--reasoning-effort", type=str, default="high")

    args = parser.parse_args()

    records = _iter_document_index(args.document_index, max_docs=-1)
    records = _sample_records(records, args.max_docs, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for idx, chunk in enumerate(_chunk_records(records, args.chunk_size), start=1):
            sample_text = _format_samples(chunk)
            prompt = USER_PROMPT_TEMPLATE.replace(SAMPLES_TOKEN, sample_text)
            body = {
                "model": args.model,
                "input": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_output_tokens": args.max_output_tokens,
            }
            if args.reasoning_effort:
                body["reasoning"] = {"effort": args.reasoning_effort}
            request = {
                "custom_id": f"akk_chunk_{idx:04d}",
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} batch requests to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
