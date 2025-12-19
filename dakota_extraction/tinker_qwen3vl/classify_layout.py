from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image


PAGE_RE = re.compile(r".*?_(\d{4})\.(jpg|jpeg|png)$", re.IGNORECASE)


@dataclass(frozen=True)
class LayoutResult:
    page_number: int
    path: str
    layout: str  # "two_column" | "single_column" | "unknown"
    split_x_norm: Optional[float]
    valley_ratio: Optional[float]
    confidence: float
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "path": self.path,
            "layout": self.layout,
            "split_x_norm": self.split_x_norm,
            "valley_ratio": self.valley_ratio,
            "confidence": self.confidence,
            "notes": self.notes,
        }


def _parse_page_number(path: Path) -> Optional[int]:
    match = PAGE_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def _smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    window = min(window, x.shape[0])
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="same")


def classify_two_column(
    image_path: Path,
    *,
    downsample_width: int = 900,
    binarize_threshold: int = 200,
    mid_band: tuple[float, float] = (0.35, 0.65),
    min_valley_ratio: float = 0.45,
    min_text_density: float = 0.01,
) -> LayoutResult:
    page_number = _parse_page_number(image_path) or 0

    try:
        with Image.open(image_path) as img:
            img = img.convert("L")
            w, h = img.size
            if w > downsample_width:
                scale = downsample_width / float(w)
                img = img.resize((downsample_width, max(1, int(round(h * scale)))), Image.Resampling.BILINEAR)

            arr = np.asarray(img, dtype=np.uint8)
    except Exception as exc:
        return LayoutResult(
            page_number=page_number,
            path=str(image_path),
            layout="unknown",
            split_x_norm=None,
            valley_ratio=None,
            confidence=0.0,
            notes=f"image_load_error: {exc}",
        )

    # Binarize: treat dark pixels as text/ink.
    ink = (arr < binarize_threshold).astype(np.float32)
    text_density = float(ink.mean())
    if text_density < min_text_density:
        return LayoutResult(
            page_number=page_number,
            path=str(image_path),
            layout="unknown",
            split_x_norm=None,
            valley_ratio=None,
            confidence=0.1,
            notes=f"low_text_density={text_density:.4f}",
        )

    col_ink = ink.mean(axis=0)
    col_ink = _smooth_1d(col_ink, window=max(3, col_ink.shape[0] // 150))

    n = col_ink.shape[0]
    lo = int(math.floor(mid_band[0] * n))
    hi = int(math.ceil(mid_band[1] * n))
    lo = max(0, min(n - 1, lo))
    hi = max(lo + 1, min(n, hi))

    mid = col_ink[lo:hi]
    valley_idx = int(lo + np.argmin(mid))
    valley = float(col_ink[valley_idx])

    # Compare valley to the average ink in left/right regions.
    left_mean = float(col_ink[: max(1, valley_idx)].mean())
    right_mean = float(col_ink[min(n - 1, valley_idx) :].mean())
    side_mean = (left_mean + right_mean) / 2.0

    if side_mean <= 1e-6:
        return LayoutResult(
            page_number=page_number,
            path=str(image_path),
            layout="unknown",
            split_x_norm=None,
            valley_ratio=None,
            confidence=0.0,
            notes="side_mean_zero",
        )

    valley_ratio = valley / side_mean
    split_x_norm = valley_idx / float(n)

    if valley_ratio < min_valley_ratio:
        # Lower ratio => stronger gutter => more confident two-column.
        confidence = min(1.0, max(0.0, (min_valley_ratio - valley_ratio) / min_valley_ratio))
        return LayoutResult(
            page_number=page_number,
            path=str(image_path),
            layout="two_column",
            split_x_norm=split_x_norm,
            valley_ratio=valley_ratio,
            confidence=confidence,
            notes=f"text_density={text_density:.4f}",
        )

    confidence = min(1.0, max(0.0, (valley_ratio - min_valley_ratio) / (1.0 - min_valley_ratio)))
    return LayoutResult(
        page_number=page_number,
        path=str(image_path),
        layout="single_column",
        split_x_norm=split_x_norm,
        valley_ratio=valley_ratio,
        confidence=confidence,
        notes=f"text_density={text_density:.4f}",
    )


def summarize_ranges(results: list[LayoutResult], *, layout: str) -> list[tuple[int, int]]:
    pages = sorted(r.page_number for r in results if r.layout == layout and r.page_number > 0)
    if not pages:
        return []
    ranges: list[tuple[int, int]] = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
            continue
        ranges.append((start, prev))
        start = prev = p
    ranges.append((start, prev))
    return ranges


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify Dakota pages as two-column dictionary vs single-column.")
    parser.add_argument(
        "--images",
        default="data/processed_images",
        help="Directory containing grammardictionar00riggrich_####.jpg images.",
    )
    parser.add_argument(
        "--out",
        default="data/page_layout_manifest.json",
        help="Output manifest JSON path.",
    )
    parser.add_argument("--downsample-width", type=int, default=900)
    parser.add_argument("--binarize-threshold", type=int, default=200)
    parser.add_argument("--min-valley-ratio", type=float, default=0.45)
    parser.add_argument("--mid-band-lo", type=float, default=0.35)
    parser.add_argument("--mid-band-hi", type=float, default=0.65)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image_dir = Path(args.images)
    paths = sorted(
        [p for p in image_dir.glob("*.jpg")] + [p for p in image_dir.glob("*.jpeg")] + [p for p in image_dir.glob("*.png")]
    )
    if not paths:
        print(f"No images found in {image_dir}")
        return 1

    results: list[LayoutResult] = []
    for path in paths:
        res = classify_two_column(
            path,
            downsample_width=args.downsample_width,
            binarize_threshold=args.binarize_threshold,
            mid_band=(args.mid_band_lo, args.mid_band_hi),
            min_valley_ratio=args.min_valley_ratio,
        )
        results.append(res)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8")

    two_ranges = summarize_ranges(results, layout="two_column")
    one_ranges = summarize_ranges(results, layout="single_column")
    print(f"Wrote manifest: {out_path}")
    print(f"Two-column ranges ({len(two_ranges)}): {two_ranges[:10]}{' ...' if len(two_ranges) > 10 else ''}")
    print(f"Single-column ranges ({len(one_ranges)}): {one_ranges[:10]}{' ...' if len(one_ranges) > 10 else ''}")
    if two_ranges:
        best = max(two_ranges, key=lambda r: r[1] - r[0])
        print(f"Suggested dictionary run (largest two-column span): {best}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
