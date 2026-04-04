#!/usr/bin/env python3
"""
Dakota Dictionary Extraction Pipeline
Following the Stoney Nakoda approach by @harleycoops

This script extracts structured linguistic data from the 1890 Dakota-English
Dictionary by Stephen Return Riggs to create training datasets for language models.

Dictionary: 440 pages of JP2 files
Structure: Two-column format with rich linguistic metadata
Goal: Create fine-tuning dataset for Dakota language model

Usage:
    # Test on first page (recommended first step)
    python extract_dakota_dictionary.py --test

    # Process specific pages
    python extract_dakota_dictionary.py --pages 1-10

    # Process all (will take hours + significant API cost)
    python extract_dakota_dictionary.py --all
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import processors
try:
    from blackfeet_extraction.tools.image_converter import ImageConverter
    from blackfeet_extraction.core.advanced_page_processor import AdvancedPageProcessor
    from blackfeet_extraction.datasets.training_dataset_builder import TrainingDatasetBuilder
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("\nPlease ensure you're in the project root and run:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def check_setup():
    """Verify environment is ready."""
    print("Checking setup...")

    # API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("[ERROR] OPENROUTER_API_KEY not set")
        print("\nAdd to your .env file:")
        print("  OPENROUTER_API_KEY=your_key_here")
        return False

    # Dictionary files
    dict_dir = Path("dictionary/grammardictionar00riggrich_jp2")
    if not dict_dir.exists():
        print(f"[ERROR] Dictionary directory not found: {dict_dir}")
        return False

    jp2_count = len(list(dict_dir.glob("*.jp2")))
    print(f"  [OK] Found {jp2_count} JP2 dictionary pages")

    # Test Pillow JP2 support
    try:
        from PIL import Image
        test_file = next(dict_dir.glob("*.jp2"))
        with Image.open(test_file):
            pass
        print("  [OK] PIL can read JP2 files")
    except Exception as e:
        print(f"[ERROR] PIL cannot read JP2: {e}")
        print("\nInstall OpenJPEG:")
        print("  Windows: https://www.openjpeg.org/")
        print("  Linux: sudo apt-get install libopenjp2-7")
        print("  Mac: brew install openjpeg")
        return False

    print("  [OK] Setup complete\n")
    return True


def test_extraction():
    """Test on first page."""
    print("\n" + "="*70)
    print(" TEST MODE: Extracting First Page")
    print("="*70)

    # Convert first page
    converter = ImageConverter(
        input_dir="dictionary/grammardictionar00riggrich_jp2",
        output_dir="data/processed_images",
        quality=95,
    )

    jp2_files = sorted(Path("dictionary/grammardictionar00riggrich_jp2").glob("*.jp2"))
    print(f"\nConverting {jp2_files[0].name}...")
    first_image = converter.convert_jp2_to_jpeg(jp2_files[0])

    # Extract with advanced processor
    print("\nExtracting dictionary entries...")
    print("Using Dakota-specialized extraction schema...")

    processor = AdvancedPageProcessor(
        output_dir="data/extracted",
        reasoning_dir="data/reasoning_traces",
    )

    extraction = processor.extract_page(
        image_path=first_image,
        page_number=1,
        thinking_budget=6000,  # High for accuracy
        page_context="First page - may contain title page or front matter",
    )

    # Display results
    processor.display_sample_entries(extraction, num=5)

    print("\n" + "="*70)
    print(" TEST COMPLETE")
    print("="*70)
    print("\nReview outputs:")
    print("  - Extraction: data/extracted/page_001.json")
    print("  - Reasoning: data/reasoning_traces/page_001_reasoning.json")
    print("\nIf results look good, run:")
    print(f"  python {sys.argv[0]} --pages 1-10")
    print()


def process_range(start: int, end: int):
    """Process a range of pages."""
    print("\n" + "="*70)
    print(f" PROCESSING PAGES {start}-{end}")
    print("="*70)

    # Convert images
    print("\nStep 1: Converting JP2 to JPEG...")
    converter = ImageConverter(
        input_dir="dictionary/grammardictionar00riggrich_jp2",
        output_dir="data/processed_images",
        quality=95,
    )

    jp2_files = sorted(Path("dictionary/grammardictionar00riggrich_jp2").glob("*.jp2"))
    to_convert = jp2_files[start-1:end]

    print(f"Converting {len(to_convert)} pages...")
    for jp2_file in to_convert:
        converter.convert_jp2_to_jpeg(jp2_file)

    # Extract entries
    print("\nStep 2: Extracting dictionary entries...")
    processor = AdvancedPageProcessor(
        output_dir="data/extracted",
        reasoning_dir="data/reasoning_traces",
    )

    for page_num in range(start, end + 1):
        image_path = Path(f"data/processed_images/grammardictionar00riggrich_{page_num:04d}.jpg")

        if not image_path.exists():
            print(f"[WARNING] Skipping page {page_num} - image not found")
            continue

        try:
            processor.extract_page(
                image_path=image_path,
                page_number=page_num,
                thinking_budget=6000,
            )
        except Exception as e:
            print(f"[ERROR] Error on page {page_num}: {e}")
            continue

    # Build datasets
    print("\nStep 3: Building training datasets...")
    builder = TrainingDatasetBuilder(
        extraction_dir="data/extracted",
        output_dir="data/training_datasets",
    )

    builder.build_all_datasets()
    stats = builder.generate_statistics()

    print("\n" + "="*70)
    print(" EXTRACTION COMPLETE")
    print("="*70)
    print(f"\nPages processed: {start}-{end}")
    print(f"Total entries: {stats.get('total_entries', 0)}")
    print("\nDatasets ready for training:")
    print("  - Translation pairs: data/training_datasets/translation_*.jsonl")
    print("  - Instruction dataset: data/training_datasets/instruction_dataset.jsonl")
    print("  - Vocabulary: data/training_datasets/vocabulary.json")
    print("  - Text corpus: data/training_datasets/blackfeet_corpus.txt")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Dakota dictionary for language model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test              # Test on first page
  %(prog)s --pages 1-10        # Process pages 1-10
  %(prog)s --pages 50-100      # Process pages 50-100
  %(prog)s --all               # Process all 440 pages (expensive!)

Output:
  data/extracted/             - Raw extractions with linguistic metadata
  data/training_datasets/     - Ready-to-use training datasets
  data/reasoning_traces/      - Model reasoning for verification
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true", help="Test on first page")
    group.add_argument("--pages", type=str, help="Page range (e.g., 1-10)")
    group.add_argument("--all", action="store_true", help="Process all 440 pages")

    args = parser.parse_args()

    print("\n" + "="*70)
    print(" DAKOTA DICTIONARY EXTRACTION PIPELINE")
    print(" 1890 Dakota-English Dictionary by Stephen Return Riggs")
    print(" Following @harleycoops Stoney Nakoda approach")
    print("="*70)

    if not check_setup():
        sys.exit(1)

    if args.test:
        test_extraction()

    elif args.pages:
        if "-" not in args.pages:
            print("[ERROR] Pages must be a range (e.g., 1-10)")
            sys.exit(1)

        start, end = map(int, args.pages.split("-"))

        if start < 1 or end > 440 or start > end:
            print("[ERROR] Invalid range. Must be 1-440 and start <= end")
            sys.exit(1)

        estimated_cost = (end - start + 1) * 0.25  # Rough estimate
        print(f"\n[WARNING] This will process {end-start+1} pages")
        print(f"Estimated API cost: ${estimated_cost:.2f}")
        print(f"Estimated time: {(end-start+1)*2} minutes")

        confirm = input("\nContinue? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled")
            return

        process_range(start, end)

    elif args.all:
        print("\n[WARNING] WARNING: Processing all 440 pages will:")
        print("  - Take 12-15 hours")
        print("  - Cost approximately $110 in API fees")
        print("  - Use significant thinking tokens")

        confirm = input("\nAre you absolutely sure? Type 'yes' to continue: ")
        if confirm != "yes":
            print("Cancelled")
            return

        process_range(1, 440)


if __name__ == "__main__":
    main()
