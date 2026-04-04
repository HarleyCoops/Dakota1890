"""Run the maintained Dakota extraction pipeline from source pages to datasets.

This entrypoint keeps the current Dakota path in one place:
1. Convert JP2 pages to JPEG where needed
2. Extract structured Dakota dictionary entries with Claude Sonnet 4.5
3. Build dataset artifacts from the saved extraction JSON
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dakota_extraction.tools.image_converter import ImageConverter
from dakota_extraction.core.page_processor import PageProcessor
from dakota_extraction.datasets.training_dataset_builder import TrainingDatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="Extract Dakota dictionary data")
    parser.add_argument(
        "--input",
        type=str,
        default="Dictionary/grammardictionar00riggrich_jp2",
        help="Input directory with JP2/JPG files",
    )
    parser.add_argument(
        "--processed",
        type=str,
        default="data/processed_images",
        help="Directory for converted images",
    )
    parser.add_argument(
        "--extracted",
        type=str,
        default="data/extracted",
        help="Directory for extraction output",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="data/training_datasets",
        help="Directory for training datasets",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Starting page number",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="Ending page number (None = all)",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=4096,
        help="Reasoning token budget per page",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip JP2 to JPEG conversion",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip OCR extraction (use existing)",
    )
    parser.add_argument(
        "--only-datasets",
        action="store_true",
        help="Only build datasets from existing extractions",
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print(" DAKOTA DICTIONARY EXTRACTION PIPELINE")
    print("="*70)
    print(f"\nInput directory:    {args.input}")
    print(f"Processed images:   {args.processed}")
    print(f"Extracted data:     {args.extracted}")
    print(f"Training datasets:  {args.datasets}")
    print(f"Thinking budget:    {args.thinking_budget} tokens/page")
    print()

    # Step 1: Convert JP2 to JPEG (if needed)
    if not args.skip_conversion and not args.only_datasets:
        print("\n" + "="*70)
        print(" STEP 1: CONVERTING JP2 FILES TO JPEG")
        print("="*70)

        converter = ImageConverter(
            input_dir=args.input,
            output_dir=args.processed,
            quality=95,
        )

        # Check for JP2 files
        jp2_files = list(Path(args.input).glob("*.jp2"))
        jpg_files = list(Path(args.input).glob("*.jpg")) + list(Path(args.input).glob("*.jpeg"))

        if jp2_files:
            print(f"Found {len(jp2_files)} JP2 files to convert...")
            converter.convert_all_jp2_files()
            image_dir = Path(args.processed)
        elif jpg_files:
            print(f"Found {len(jpg_files)} JPEG files (no conversion needed)")
            image_dir = Path(args.input)
        else:
            print(f" No image files found in {args.input}")
            print("\nPlease add dictionary page images to:")
            print(f"  {args.input}/")
            print("\nSupported formats: .jp2, .jpg, .jpeg, .png")
            return
    else:
        # Use existing processed images or input directly
        if Path(args.processed).exists() and list(Path(args.processed).glob("*.jpg")):
            image_dir = Path(args.processed)
        else:
            image_dir = Path(args.input)

    # Step 2: Extract dictionary entries
    if not args.skip_extraction and not args.only_datasets:
        print("\n" + "="*70)
        print(" STEP 2: EXTRACTING DICTIONARY ENTRIES")
        print("="*70)

        processor = PageProcessor(
            output_dir=args.extracted,
            reasoning_dir="data/reasoning_traces",
        )

        # Process pages
        processor.batch_extract(
            image_dir=image_dir,
            start_page=args.start_page,
            end_page=args.end_page,
            thinking_budget=args.thinking_budget,
        )

    # Step 3: Build training datasets
    print("\n" + "="*70)
    print(" STEP 3: BUILDING TRAINING DATASETS")
    print("="*70)

    # Check if we have extractions
    extraction_files = list(Path(args.extracted).glob("page_*.json"))
    if not extraction_files:
        print(f" No extraction files found in {args.extracted}")
        print("\nRun extraction first without --only-datasets flag")
        return

    print(f"Found {len(extraction_files)} extracted pages")

    builder = TrainingDatasetBuilder(
        extraction_dir=args.extracted,
        output_dir=args.datasets,
    )

    builder.build_all_datasets()
    builder.generate_statistics()

    print("\n" + "="*70)
    print(" PIPELINE COMPLETE!")
    print("="*70)
    print(f"\n Extracted data:     {args.extracted}/")
    print(f" Training datasets:  {args.datasets}/")
    print(" Reasoning traces:   data/reasoning_traces/")
    print("\nNext steps:")
    print("  1. Review extracted data in", args.extracted)
    print("  2. Check reasoning traces for quality assurance")
    print("  3. Use training datasets to fine-tune a language model")
    print("  4. See PIPELINE.md and SETUP.md for maintained workflow details")
    print()


if __name__ == "__main__":
    main()
