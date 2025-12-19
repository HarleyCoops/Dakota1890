#!/usr/bin/env python3
"""Build a unified document index from SourceDocuments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SCRIPT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from akkadian.ingest import build_document_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build document index JSONL")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("Akkadian/SourceDocuments"),
        help="Path to SourceDocuments folder",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Akkadian/data/document_index.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable transliteration/translation normalization",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=-1,
        help="Limit number of published_texts rows",
    )

    args = parser.parse_args()
    count = build_document_index(
        source_root=args.source_root,
        out_path=args.out,
        normalize=not args.no_normalize,
        max_rows=args.max_rows,
    )
    print(f"Wrote {count} records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
