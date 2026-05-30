from __future__ import annotations

import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from dakota_extraction.datasets.training_dataset_builder import TrainingDatasetBuilder
from dakota_extraction.tools.image_converter import ImageConverter

from .qwen3vl_page_processor import Qwen3VLTinkerPageProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dakota dictionary extraction via Tinker Qwen3-VL.")

    parser.add_argument(
        "--input",
        type=str,
        default="Dictionary/grammardictionar00riggrich_jp2",
        help="Input directory containing raw dictionary scans (JP2/JPG/PNG).",
    )
    parser.add_argument(
        "--processed",
        type=str,
        default="data/processed_images",
        help="Intermediate directory for converted/optimized images.",
    )
    parser.add_argument(
        "--extracted",
        type=str,
        default="data/extracted_qwen3vl_tinker",
        help="Directory where structured JSON extractions will be saved.",
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        default="data/reasoning_traces_qwen3vl_tinker",
        help="Directory where raw VLM responses will be saved.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="data/training_datasets_qwen3vl_tinker",
        help="Directory where training datasets will be written.",
    )

    parser.add_argument("--start-page", type=int, default=1, help="Page number to begin processing.")
    parser.add_argument("--end-page", type=int, default=None, help="Page number to stop processing.")

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Tinker base_model name (e.g., Qwen/Qwen3-VL-30B-A3B-Instruct).",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default=None,
        help="HuggingFace tokenizer model name (defaults to Qwen/<model>).",
    )
    parser.add_argument("--base-url", type=str, default=None, help="Optional custom Tinker API base URL.")
    parser.add_argument("--max-tokens", type=int, default=16000, help="Max new tokens to generate per page.")
    parser.add_argument(
        "--resize-long-edge",
        type=int,
        default=None,
        help="If set, resize images so the longest edge is <= this many pixels.",
    )
    parser.add_argument("--patch-size", type=int, default=28, help="Patch size used for image token estimation.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = deterministic).")
    parser.add_argument(
        "--split-columns",
        action="store_true",
        help="For two-column dictionary pages: crop left/right and extract separately, then merge.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip pages whose `page_###.json` already exists in --extracted.",
    )

    parser.add_argument("--skip-conversion", action="store_true", help="Skip JP2->JPEG conversion step.")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip VLM extraction (use existing JSONs).")
    parser.add_argument("--only-datasets", action="store_true", help="Only build datasets from existing JSONs.")
    parser.add_argument("--skip-datasets", action="store_true", help="Skip dataset building step.")
    parser.add_argument(
        "--guardrail-threshold",
        type=float,
        default=0.005,
        help="Fraction of guardrail failures allowed before dataset build errors.",
    )

    return parser


def _detect_image_dir(input_dir: Path, processed_dir: Path) -> Path:
    input_images = (
        list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png"))
    )
    if input_images:
        return input_dir

    processed_images = list(processed_dir.glob("*.jpg")) + list(processed_dir.glob("*.jpeg"))
    if processed_dir.exists() and processed_images:
        return processed_dir
    return input_dir


def main() -> int:
    args = build_parser().parse_args()
    if load_dotenv is not None:
        load_dotenv()

    input_dir = Path(args.input)
    processed_dir = Path(args.processed)
    extracted_dir = Path(args.extracted)
    reasoning_dir = Path(args.reasoning)
    datasets_dir = Path(args.datasets)

    if not args.skip_conversion and not args.only_datasets:
        converter = ImageConverter(input_dir=str(input_dir), output_dir=str(processed_dir), quality=95)
        jp2_files = list(input_dir.glob("*.jp2"))
        jpg_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png"))
        if jp2_files:
            converter.convert_all_jp2_files()
        elif not jpg_files:
            print(f"No images found in {input_dir}")
            return 1

    image_dir = _detect_image_dir(input_dir, processed_dir)

    if not args.skip_extraction and not args.only_datasets:
        processor = Qwen3VLTinkerPageProcessor(
            base_url=args.base_url,
            output_dir=str(extracted_dir),
            reasoning_dir=str(reasoning_dir),
            model=args.model,
            tokenizer_model=args.tokenizer_model,
            resize_long_edge=args.resize_long_edge,
            patch_size=args.patch_size,
            temperature=args.temperature,
            split_columns=bool(args.split_columns),
        )
        processor.batch_extract(
            image_dir=image_dir,
            start_page=args.start_page,
            end_page=args.end_page,
            max_tokens=args.max_tokens,
            skip_existing=bool(args.skip_existing),
        )

    if args.skip_datasets:
        print("Skipping dataset build (--skip-datasets).")
        return 0

    # Build datasets (same builder as Claude pipeline).
    extracted_files = list(extracted_dir.glob("page_*.json"))
    if not extracted_files:
        print(f"No extraction files found in {extracted_dir}")
        return 1

    builder = TrainingDatasetBuilder(
        extraction_dir=str(extracted_dir),
        output_dir=str(datasets_dir),
        guardrail_threshold=float(args.guardrail_threshold),
    )
    try:
        builder.build_all_datasets()
        builder.generate_statistics()
    except ValueError as exc:
        print(f"Dataset build failed: {exc}")
        print("Tip: re-run with `--skip-datasets` or increase `--guardrail-threshold`.")
        return 2

    print("Pipeline complete.")
    print(f"Extracted data: {extracted_dir}")
    print(f"Training datasets: {datasets_dir}")
    print(f"Raw responses: {reasoning_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
