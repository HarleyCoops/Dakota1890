#!/usr/bin/env python3
"""Induce Akkadian grammar rules using GPT-5.2 only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SCRIPT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from akkadian.rule_induction import InductionConfig, run_induction


def main() -> int:
    parser = argparse.ArgumentParser(description="Induce grammar rules with GPT-5.2")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("Akkadian/SourceDocuments"),
        help="Path to SourceDocuments folder",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Akkadian/data/grammar_rules/rules.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="GPT-5.2 model name (enforced)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25,
        help="Number of documents per prompt",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=-1,
        help="Limit number of documents to sample",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=15000,
        help="Maximum tokens per response",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="high",
        help="Reasoning effort setting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write prompts only, no API calls",
    )
    parser.add_argument(
        "--no-save-raw",
        action="store_true",
        help="Disable saving raw model outputs",
    )

    args = parser.parse_args()

    config = InductionConfig(
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        chunk_size=args.chunk_size,
        max_docs=args.max_docs,
        seed=args.seed,
        dry_run=args.dry_run,
        save_raw=not args.no_save_raw,
    )

    count = run_induction(
        source_root=args.source_root,
        out_path=args.out,
        config=config,
    )

    if args.dry_run:
        print("Dry run complete. Prompts written to grammar_rules directory.")
    else:
        print(f"Wrote {count} rules to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
