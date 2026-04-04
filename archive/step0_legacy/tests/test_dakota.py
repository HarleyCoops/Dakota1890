#!/usr/bin/env python3
"""
Dakota Dictionary Extraction Pipeline - Updated for Page 89 Start
Following the Stoney Nakoda approach by @harleycoops

Dictionary Structure:
- Pages 1-88: Grammar rules and linguistic notes (skip for now)
- Pages 89-440: Dictionary entries (our target)

This script extracts structured linguistic data from dictionary pages only.

Usage:
    # Test on page 89 (first dictionary page)
    python extract_dakota_dictionary_v2.py --test

    # Process first 20 dictionary pages
    python extract_dakota_dictionary_v2.py --pages 89-108

    # Process all dictionary pages
    python extract_dakota_dictionary_v2.py --all-dictionary
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
    print(f" Import error: {e}")
    print("\nPlease ensure you're in the project root and run:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


# Constants
DICTIONARY_START_PAGE = 89  # Dictionary entries begin here
DICTIONARY_END_PAGE = 440   # Last page
GRAMMAR_PAGES = 88          # Pages 1-88 are grammar


def check_setup():
    """Verify environment is ready."""
    print("Checking setup...")

    # API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print(" OPENROUTER_API_KEY not set")
        print("\nAdd to your .env file:")
        print("  OPENROUTER_API_KEY=your_key_here")
        return False

    # Dictionary files
    dict_dir = Path("dictionary/grammardictionar00riggrich_jp2")
    if not dict_dir.exists():
        print(f" Dictionary directory not found: {dict_dir}")
        return False

    jp2_count = len(list(dict_dir.glob("*.jp2")))
    print(f"   Found {jp2_count} JP2 pages")
    print(f"   Grammar pages: 1-{GRAMMAR_PAGES}")
    print(f"   Dictionary pages: {DICTIONARY_START_PAGE}-{DICTIONARY_END_PAGE}")

    # Test Pillow JP2 support
    try:
        from PIL import Image
        test_file = next(dict_dir.glob("*.jp2"))
        with Image.open(test_file):
            pass
        print("   PIL can read JP2 files")
    except Exception as e:
        print(f" PIL cannot read JP2: {e}")
        print("\nInstall OpenJPEG:")
        print("  Windows: https://www.openjpeg.org/")
        print("  Linux: sudo apt-get install libopenjp2-7")
        print("  Mac: brew install openjpeg")
        return False

    print("   Setup complete\n")
    return True


def test_extraction():
    """Test on page 89 (first dictionary page)."""
    print("\n" + "="*70)
    print(f" TEST MODE: Page {DICTIONARY_START_PAGE} (First Dictionary Page)")
    print("="*70)
    print(f"\n Pages 1-{GRAMMAR_PAGES}: Grammar rules (skipped)")
    print(f" Pages {DICTIONARY_START_PAGE}-{DICTIONARY_END_PAGE}: Dictionary entries (extracting)\n")

    # Convert page 89
    converter = ImageConverter(
        input_dir="dictionary/grammardictionar00riggrich_jp2",
        output_dir="data/processed_images",
        quality=95,
    )

    # Get page 89 specifically
    page_file = Path(f"dictionary/grammardictionar00riggrich_jp2/grammardictionar00riggrich_{DICTIONARY_START_PAGE:04d}.jp2")

    if not page_file.exists():
        print(f" Page {DICTIONARY_START_PAGE} not found: {page_file}")

        # Show what's available
        jp2_files = sorted(Path("dictionary/grammardictionar00riggrich_jp2").glob("*.jp2"))
        print("\nAvailable pages:")
        print(f"  First: {jp2_files[0].name}")
        print(f"  Last: {jp2_files[-1].name}")
        print(f"  Total: {len(jp2_files)}")
        return

    print(f"Converting {page_file.name}...")
    image = converter.convert_jp2_to_jpeg(page_file)

    # Extract with advanced processor
    print("\nExtracting dictionary entries...")
    print("Using Dakota-specialized extraction schema...")
    print("This will use Qwen3-VL's reasoning to understand entry structure...\n")

    processor = AdvancedPageProcessor(
        output_dir="data/extracted",
        reasoning_dir="data/reasoning_traces",
    )

    extraction = processor.extract_page(
        image_path=image,
        page_number=DICTIONARY_START_PAGE,
        thinking_budget=6000,
        page_context="First dictionary page - entries begin here after grammar section",
    )

    # Display results
    processor.display_sample_entries(extraction, num=5)

    print("\n" + "="*70)
    print(" TEST COMPLETE")
    print("="*70)
    print("\n Review outputs:")
    print(f"  - Extraction: data/extracted/page_{DICTIONARY_START_PAGE:03d}.json")
    print(f"  - Reasoning: data/reasoning_traces/page_{DICTIONARY_START_PAGE:03d}_reasoning.json")
    print("\n If results look good, process more pages:")
    print(f"  python {sys.argv[0]} --pages {DICTIONARY_START_PAGE}-{DICTIONARY_START_PAGE+11}  # 12 pages")
    print(f"  python {sys.argv[0]} --pages {DICTIONARY_START_PAGE}-150  # More pages")
    print()


def process_range(start: int, end: int):
    """Process a range of pages."""
    print("\n" + "="*70)
    print(f" PROCESSING PAGES {start}-{end}")
    print("="*70)

    # Validate range
    if start < DICTIONARY_START_PAGE:
        print(f"\n  WARNING: Pages 1-{GRAMMAR_PAGES} contain grammar rules!")
        print(f"Dictionary entries start at page {DICTIONARY_START_PAGE}.")
        print(f"\n Recommended: --pages {DICTIONARY_START_PAGE}-{end}")

        confirm = input(f"\nProcess pages {start}-{end} anyway? [y/N]: ")
        if confirm.lower() != 'y':
            print(f"\nCancelled. Use --pages {DICTIONARY_START_PAGE}-{end} for dictionary entries.")
            return

    # Convert images
    print("\nStep 1: Converting JP2 to JPEG...")
    converter = ImageConverter(
        input_dir="dictionary/grammardictionar00riggrich_jp2",
        output_dir="data/processed_images",
        quality=95,
    )

    print(f"Converting {end-start+1} pages...")
    for page_num in range(start, end + 1):
        jp2_file = Path(f"dictionary/grammardictionar00riggrich_jp2/grammardictionar00riggrich_{page_num:04d}.jp2")
        if jp2_file.exists():
            converter.convert_jp2_to_jpeg(jp2_file)
        else:
            print(f"    Page {page_num} not found, skipping")

    # Extract entries
    print("\nStep 2: Extracting dictionary entries...")
    processor = AdvancedPageProcessor(
        output_dir="data/extracted",
        reasoning_dir="data/reasoning_traces",
    )

    for page_num in range(start, end + 1):
        image_path = Path(f"data/processed_images/grammardictionar00riggrich_{page_num:04d}.jpg")

        if not image_path.exists():
            print(f"  Skipping page {page_num} - image not found")
            continue

        try:
            print(f"\n{''*70}")
            processor.extract_page(
                image_path=image_path,
                page_number=page_num,
                thinking_budget=6000,
            )
        except Exception as e:
            print(f" Error on page {page_num}: {e}")
            continue

    # Build datasets
    print("\n" + "="*70)
    print(" Step 3: Building training datasets...")
    print("="*70)

    builder = TrainingDatasetBuilder(
        extraction_dir="data/extracted",
        output_dir="data/training_datasets",
    )

    builder.build_all_datasets()
    stats = builder.generate_statistics()

    print("\n" + "="*70)
    print(" EXTRACTION COMPLETE")
    print("="*70)
    print("\n Statistics:")
    print(f"  Pages processed: {start}-{end} ({end-start+1} pages)")
    print(f"  Total entries: {stats.get('total_entries', 0)}")
    print(f"  Avg entries/page: {stats.get('avg_entries_per_page', 0):.1f}")
    print("\n Datasets created:")
    print("  - Translation: data/training_datasets/translation_*.jsonl")
    print("  - Instructions: data/training_datasets/instruction_dataset.jsonl")
    print("  - Vocabulary: data/training_datasets/vocabulary.json")
    print("  - Corpus: data/training_datasets/blackfeet_corpus.txt")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Dakota dictionary entries (pages 89+)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Dictionary Structure:
  Pages 1-{GRAMMAR_PAGES}:    Grammar rules (skip for now)
  Pages {DICTIONARY_START_PAGE}-{DICTIONARY_END_PAGE}: Dictionary entries (extract these)

Examples:
  %(prog)s --test                    # Test on page {DICTIONARY_START_PAGE}
  %(prog)s --pages {DICTIONARY_START_PAGE}-100            # First 12 dictionary pages
  %(prog)s --pages {DICTIONARY_START_PAGE}-200            # More pages
  %(prog)s --all-dictionary          # All dictionary pages ({DICTIONARY_START_PAGE}-{DICTIONARY_END_PAGE})

Costs:
  ~$0.25 per page
  12 pages: ~$3
  All dictionary (352 pages): ~$88
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true",
                      help=f"Test on page {DICTIONARY_START_PAGE} (first dictionary page)")
    group.add_argument("--pages", type=str,
                      help="Page range (e.g., 89-100)")
    group.add_argument("--all-dictionary", action="store_true",
                      help=f"Process all dictionary pages ({DICTIONARY_START_PAGE}-{DICTIONARY_END_PAGE})")

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
            print(" Pages must be a range (e.g., 89-100)")
            sys.exit(1)

        start, end = map(int, args.pages.split("-"))

        if start < 1 or end > DICTIONARY_END_PAGE or start > end:
            print(f" Invalid range. Must be 1-{DICTIONARY_END_PAGE} and start <= end")
            sys.exit(1)

        # Helpful suggestions
        if start < DICTIONARY_START_PAGE:
            print(f"\n Note: Dictionary entries start at page {DICTIONARY_START_PAGE}")
            print(f"   Pages 1-{GRAMMAR_PAGES} contain grammar rules (different structure)")

        num_pages = end - start + 1
        estimated_cost = num_pages * 0.25
        estimated_time = num_pages * 2  # minutes

        print(f"\n Processing {num_pages} pages ({start}-{end})")
        print(f" Estimated cost: ${estimated_cost:.2f}")
        print(f"  Estimated time: {estimated_time} minutes")

        confirm = input("\nContinue? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled")
            return

        process_range(start, end)

    elif args.all_dictionary:
        num_pages = DICTIONARY_END_PAGE - DICTIONARY_START_PAGE + 1
        estimated_cost = num_pages * 0.25
        estimated_hours = (num_pages * 2) / 60

        print(f"\n  Processing ALL dictionary pages ({DICTIONARY_START_PAGE}-{DICTIONARY_END_PAGE})")
        print(f" Total pages: {num_pages}")
        print(f" Estimated cost: ${estimated_cost:.2f}")
        print(f"  Estimated time: {estimated_hours:.1f} hours")
        print("\nThis will:")
        print("  - Use significant API tokens")
        print("  - Take many hours to complete")
        print("  - Generate comprehensive dataset")

        confirm = input("\nType 'yes' to confirm: ")
        if confirm != "yes":
            print("Cancelled")
            return

        process_range(DICTIONARY_START_PAGE, DICTIONARY_END_PAGE)


if __name__ == "__main__":
    main()
