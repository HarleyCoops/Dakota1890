#!/usr/bin/env python3
"""
Quick Start: Blackfeet Dictionary Extraction

This script processes your 440-page Blackfeet grammar/dictionary JP2 files
to create training data for a Blackfeet language model.

Location: dictionary/grammardictionar00riggrich_jp2/*.jp2

Usage:
    # Process first page only (test)
    python extract_blackfeet_dictionary.py --test

    # Process first 10 pages
    python extract_blackfeet_dictionary.py --pages 1-10

    # Process all 440 pages (recommended: run overnight)
    python extract_blackfeet_dictionary.py --all

    # Just build datasets from existing extractions
    python extract_blackfeet_dictionary.py --datasets-only
"""

import os
import sys
from pathlib import Path

# Ensure we have the required imports
try:
    from dotenv import load_dotenv
    from blackfeet_extraction.tools.image_converter import ImageConverter
    from blackfeet_extraction.core.page_processor import PageProcessor
    from blackfeet_extraction.datasets.training_dataset_builder import TrainingDatasetBuilder
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("\nPlease install requirements:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Load environment
load_dotenv()


def check_environment():
    """Check that everything is set up correctly."""
    print("Checking environment...")

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY not found in .env file")
        print("\nPlease add your OpenRouter API key to .env:")
        print("  OPENROUTER_API_KEY=your_key_here")
        return False

    print(f"  [OK] OpenRouter API key: {api_key[:20]}...")

    # Check dictionary files
    dict_dir = Path("dictionary/grammardictionar00riggrich_jp2")
    if not dict_dir.exists():
        print(f"[ERROR] Dictionary directory not found: {dict_dir}")
        return False

    jp2_files = list(dict_dir.glob("*.jp2"))
    print(f"  [OK] Found {len(jp2_files)} JP2 files in {dict_dir}")

    if len(jp2_files) == 0:
        print("[ERROR] No JP2 files found!")
        return False

    # Check Pillow JP2 support
    try:
        from PIL import Image
        # Test opening a JP2 file
        with Image.open(jp2_files[0]):
            pass
        print("  [OK] Pillow can read JP2 files")
    except Exception as e:
        print(f"[ERROR] Pillow cannot read JP2 files: {e}")
        print("\nYou may need to install OpenJPEG library:")
        print("  Windows: download from https://www.openjpeg.org/")
        print("  Linux: sudo apt-get install libopenjp2-7")
        print("  Mac: brew install openjpeg")
        return False

    return True


def test_single_page():
    """Test extraction on a single page."""
    print("\n" + "="*70)
    print(" TEST MODE: Processing First Page Only")
    print("="*70)

    # Convert first JP2
    print("\nStep 1: Converting JP2 to JPEG...")
    converter = ImageConverter(
        input_dir="dictionary/grammardictionar00riggrich_jp2",
        output_dir="data/processed_images",
        quality=95,
    )

    jp2_files = sorted(Path("dictionary/grammardictionar00riggrich_jp2").glob("*.jp2"))
    first_page = converter.convert_jp2_to_jpeg(jp2_files[0])
    print(f"  [OK] Converted: {first_page}")

    # Extract first page
    print("\nStep 2: Extracting dictionary entries...")
    processor = PageProcessor(
        output_dir="data/extracted",
        reasoning_dir="data/reasoning_traces",
    )

    extraction = processor.extract_page(
        image_path=first_page,
        page_number=1,
        thinking_budget=4096,  # High budget for accuracy
    )

    print("\n" + "="*70)
    print(" TEST RESULTS")
    print("="*70)
    print(f"Entries extracted: {len(extraction.get('entries', []))}")
    print(f"Layout: {extraction.get('layout')}")
    print(f"Page notes: {extraction.get('page_notes')}")

    if extraction.get('entries'):
        print("\nFirst 3 entries:")
        for i, entry in enumerate(extraction['entries'][:3], 1):
            print(f"\n  Entry {i}:")
            print(f"    Blackfeet: {entry.get('blackfeet')}")
            print(f"    English: {entry.get('english')}")
            print(f"    POS: {entry.get('pos')}")
            print(f"    Confidence: {entry.get('confidence')}")

    print("\n[OK] Test complete! Review the extraction in data/extracted/page_001.json")
    print("\nIf results look good, run with --pages 1-10 to process more pages")


def process_pages(start: int, end: int):
    """Process a range of pages."""
    print("\n" + "="*70)
    print(f" PROCESSING PAGES {start}-{end}")
    print("="*70)

    # Step 1: Convert JP2 to JPEG
    print("\nStep 1: Converting JP2 files to JPEG...")
    converter = ImageConverter(
        input_dir="dictionary/grammardictionar00riggrich_jp2",
        output_dir="data/processed_images",
        quality=95,
    )

    jp2_files = sorted(Path("dictionary/grammardictionar00riggrich_jp2").glob("*.jp2"))
    pages_to_process = jp2_files[start-1:end]

    print(f"Converting {len(pages_to_process)} pages...")
    converted = []
    for jp2_file in pages_to_process:
        converted_path = converter.convert_jp2_to_jpeg(jp2_file)
        converted.append(converted_path)

    # Step 2: Extract dictionary entries
    print("\nStep 2: Extracting dictionary entries with Qwen3-VL Thinking...")
    processor = PageProcessor(
        output_dir="data/extracted",
        reasoning_dir="data/reasoning_traces",
    )

    processor.batch_extract(
        image_dir=Path("data/processed_images"),
        start_page=start,
        end_page=end,
        thinking_budget=4096,
    )

    # Step 3: Build datasets
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
    print(f"\nProcessed pages: {start}-{end}")
    print(f"Total entries: {stats.get('total_entries', 0)}")
    print("\nOutput locations:")
    print("  - Extracted data: data/extracted/")
    print("  - Training datasets: data/training_datasets/")
    print("  - Reasoning traces: data/reasoning_traces/")


def build_datasets_only():
    """Build datasets from existing extractions."""
    print("\n" + "="*70)
    print(" BUILDING DATASETS FROM EXISTING EXTRACTIONS")
    print("="*70)

    extraction_files = list(Path("data/extracted").glob("page_*.json"))
    if not extraction_files:
        print("[ERROR] No extraction files found in data/extracted/")
        print("\nRun extraction first with --test or --pages flags")
        return

    print(f"Found {len(extraction_files)} extracted pages")

    builder = TrainingDatasetBuilder(
        extraction_dir="data/extracted",
        output_dir="data/training_datasets",
    )

    builder.build_all_datasets()
    builder.generate_statistics()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Blackfeet dictionary to training data"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true", help="Test on first page only")
    group.add_argument("--pages", type=str, help="Page range (e.g., 1-10 or 1-440)")
    group.add_argument("--all", action="store_true", help="Process all 440 pages")
    group.add_argument("--datasets-only", action="store_true", help="Build datasets only")

    args = parser.parse_args()

    print("\n" + "="*70)
    print(" BLACKFEET DICTIONARY EXTRACTION PIPELINE")
    print(" Following the Stoney Nakoda approach by @harleycoops")
    print("="*70)

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Run requested operation
    if args.test:
        test_single_page()

    elif args.datasets_only:
        build_datasets_only()

    elif args.pages:
        # Parse page range
        if "-" in args.pages:
            start, end = map(int, args.pages.split("-"))
        else:
            start = end = int(args.pages)

        if start < 1 or end > 440:
            print("[ERROR] Page range must be between 1 and 440")
            sys.exit(1)

        process_pages(start, end)

    elif args.all:
        confirm = input(
            "\n[WARNING] This will process all 440 pages. This may take several hours "
            "and use significant API tokens.\n\nContinue? [y/N]: "
        )
        if confirm.lower() != "y":
            print("Cancelled.")
            return

        process_pages(1, 440)


if __name__ == "__main__":
    main()
