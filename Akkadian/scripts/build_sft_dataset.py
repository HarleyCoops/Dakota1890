#!/usr/bin/env python3
"""Build SFT dataset JSONL from SourceDocuments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SCRIPT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from akkadian.sft import SFTBuildOptions, build_sft_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SFT dataset JSONL")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("Akkadian/SourceDocuments"),
        help="Path to SourceDocuments folder",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Akkadian/data/sft.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Do not include transliteration normalization tasks",
    )
    parser.add_argument(
        "--skip-lexicon",
        action="store_true",
        help="Do not include lexicon tasks",
    )
    parser.add_argument(
        "--include-lexeme",
        action="store_true",
        help="Include lexeme lookup tasks",
    )
    parser.add_argument(
        "--no-normalize-translation",
        action="store_true",
        help="Disable English translation normalization",
    )
    parser.add_argument(
        "--normalize-transliteration",
        action="store_true",
        help="Normalize transliteration before prompts",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=-1,
        help="Limit number of train.csv rows",
    )
    parser.add_argument(
        "--max-published",
        type=int,
        default=-1,
        help="Limit number of published_texts rows",
    )
    parser.add_argument(
        "--max-lexicon",
        type=int,
        default=-1,
        help="Limit number of lexicon rows",
    )

    args = parser.parse_args()

    options = SFTBuildOptions(
        include_normalization=not args.skip_normalization,
        include_lexicon=not args.skip_lexicon,
        include_lexeme=args.include_lexeme,
        normalize_translation=not args.no_normalize_translation,
        normalize_transliteration=args.normalize_transliteration,
        max_train=args.max_train,
        max_published=args.max_published,
        max_lexicon=args.max_lexicon,
    )

    count = build_sft_dataset(
        source_root=args.source_root,
        out_path=args.out,
        options=options,
    )
    print(f"Wrote {count} tasks to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
