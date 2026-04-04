#!/usr/bin/env python3
"""
Dakota Dictionary Extraction Pipeline - Updated for Page 95 Start
Following the Stoney Nakoda approach by @harleycoops

Dictionary Structure:
- Pages 1-92: Grammar rules and linguistic notes (already extracted for RL training)
- Pages 95-440: Dictionary entries (extract these for SFT training)

This script extracts Dakota words with their English definitions from dictionary pages.
The extracted pairs (headword + definition_primary) feed into synthetic Q&A generation for SFT.

Usage:
    # Test on page 95 (first dictionary page)
    python extract_dakota_dictionary_v2.py --test

    # Process first 20 dictionary pages
    python extract_dakota_dictionary_v2.py --pages 95-114

    # Process all dictionary pages
    python extract_dakota_dictionary_v2.py --all-dictionary
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path so we can import dakota_extraction
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment
load_dotenv()

# Import processors
try:
    from dakota_extraction.tools.image_converter import ImageConverter
    from dakota_extraction.core.advanced_page_processor import AdvancedPageProcessor
    from dakota_extraction.datasets.training_dataset_builder import TrainingDatasetBuilder
except ImportError as e:
    print(f"ERROR: Import error: {e}")
    print("\nPlease ensure you're in the project root and run:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


# Constants
DICTIONARY_START_PAGE = 95  # Dictionary entries begin here (after grammar section ends at 92)
DICTIONARY_END_PAGE = 440   # Last page
GRAMMAR_PAGES = 92          # Pages 1-92 are grammar (ends at page 92)


def check_setup():
    """Verify environment is ready."""
    print("Checking setup...")

    # API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        print("\nAdd to your .env file:")
        print("  ANTHROPIC_API_KEY=your_key_here")
        return False

    # Dictionary files
    dict_dir = Path("Dictionary/grammardictionar00riggrich_jp2")
    if not dict_dir.exists():
        print(f"ERROR: Dictionary directory not found: {dict_dir}")
        return False

    jp2_count = len(list(dict_dir.glob("*.jp2")))
    print(f"  Found {jp2_count} JP2 pages")
    print(f"  Grammar pages: 1-{GRAMMAR_PAGES}")
    print(f"  Dictionary pages: {DICTIONARY_START_PAGE}-{DICTIONARY_END_PAGE}")

    # Test Pillow JP2 support
    try:
        from PIL import Image
        test_file = next(dict_dir.glob("*.jp2"))
        with Image.open(test_file):
            pass
        print("  PIL can read JP2 files")
    except Exception as e:
        print(f"ERROR: PIL cannot read JP2: {e}")
        print("\nInstall OpenJPEG:")
        print("  Windows: https://www.openjpeg.org/")
        print("  Linux: sudo apt-get install libopenjp2-7")
        print("  Mac: brew install openjpeg")
        return False

    print("  Setup complete\n")
    return True


def test_extraction():
    """Test on page 95 (first dictionary page)."""
    print("\n" + "="*70)
    print(f" TEST MODE: Page {DICTIONARY_START_PAGE} (First Dictionary Page)")
    print("="*70)
    print(f"\nPages 1-{GRAMMAR_PAGES}: Grammar rules (already extracted)")
    print(f"Pages {DICTIONARY_START_PAGE}-{DICTIONARY_END_PAGE}: Dictionary entries (extracting for SFT)\n")

    # Convert page 95
    converter = ImageConverter(
        input_dir="Dictionary/grammardictionar00riggrich_jp2",
        output_dir="data/processed_images",
        quality=95,
    )

    # Get page 95 specifically
    page_file = Path(f"Dictionary/grammardictionar00riggrich_jp2/grammardictionar00riggrich_{DICTIONARY_START_PAGE:04d}.jp2")

    if not page_file.exists():
        print(f"ERROR: Page {DICTIONARY_START_PAGE} not found: {page_file}")

        # Show what's available
        jp2_files = sorted(Path("Dictionary/grammardictionar00riggrich_jp2").glob("*.jp2"))
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
    print("This will use Claude Sonnet 4.5 to extract dictionary entries...\n")

    processor = AdvancedPageProcessor(
        output_dir="data/extracted",
        reasoning_dir="data/reasoning_traces",
    )

    extraction = processor.extract_page(
        image_path=image,
        page_number=DICTIONARY_START_PAGE,
        max_tokens=16000,
        page_context="First dictionary page - entries begin here after grammar section",
    )

    # Display results
    processor.display_sample_entries(extraction, num=5)

    print("\n" + "="*70)
    print(" TEST COMPLETE")
    print("="*70)
    print("\nReview outputs:")
    print(f"  - Extraction: data/extracted/page_{DICTIONARY_START_PAGE:03d}.json")
    print(f"  - Reasoning: data/reasoning_traces/page_{DICTIONARY_START_PAGE:03d}_reasoning.json")
    print("\nIf results look good, process more pages:")
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
        print(f"\nWARNING: Pages 1-{GRAMMAR_PAGES} contain grammar rules!")
        print(f"Dictionary entries start at page {DICTIONARY_START_PAGE}.")
        print(f"\nRecommended: --pages {DICTIONARY_START_PAGE}-{end}")

        confirm = input(f"\nProcess pages {start}-{end} anyway? [y/N]: ")
        if confirm.lower() != 'y':
            print(f"\nCancelled. Use --pages {DICTIONARY_START_PAGE}-{end} for dictionary entries.")
            return

    # Check what's already extracted
    output_dir = Path("data/extracted")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    already_extracted = []
    needs_extraction = []
    
    for page_num in range(start, end + 1):
        output_file = output_dir / f"page_{page_num:03d}.json"
        if output_file.exists():
            # Verify it's a valid JSON with entries
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('entries') and len(data.get('entries', [])) > 0:
                        already_extracted.append(page_num)
                        continue
            except (json.JSONDecodeError, Exception):
                # File exists but is invalid, reprocess it
                pass
        
        needs_extraction.append(page_num)
    
    print("\n" + "="*70)
    print(" EXTRACTION STATUS CHECK")
    print("="*70)
    print(f"Total pages in range: {end-start+1}")
    print(f"Already extracted: {len(already_extracted)}")
    print(f"Needs extraction: {len(needs_extraction)}")
    
    if already_extracted:
        print(f"\nSkipping already-extracted pages: {min(already_extracted)}-{max(already_extracted)}")
        print(f"  (Sample: {', '.join(map(str, sorted(already_extracted)[:10]))}{'...' if len(already_extracted) > 10 else ''})")
    
    if not needs_extraction:
        print("\n[SUCCESS] All pages in this range have already been extracted!")
        print(f"   Extracted files are in: {output_dir}")
        return
    
    # Convert images (only for pages that need extraction)
    print("\n" + "="*70)
    print(" Step 1: Converting JP2 to JPEG (only missing images)...")
    print("="*70)
    converter = ImageConverter(
        input_dir="Dictionary/grammardictionar00riggrich_jp2",
        output_dir="data/processed_images",
        quality=95,
    )

    images_converted = 0
    images_skipped = 0
    
    for page_num in needs_extraction:
        jp2_file = Path(f"Dictionary/grammardictionar00riggrich_jp2/grammardictionar00riggrich_{page_num:04d}.jp2")
        image_path = Path(f"data/processed_images/grammardictionar00riggrich_{page_num:04d}.jpg")
        
        if image_path.exists():
            images_skipped += 1
            continue
            
        if jp2_file.exists():
            converter.convert_jp2_to_jpeg(jp2_file)
            images_converted += 1
        else:
            print(f"  WARNING: Page {page_num} JP2 not found, skipping")

    print(f"\nImage conversion: {images_converted} converted, {images_skipped} already existed")

    # Extract entries (only for pages that need extraction)
    print("\n" + "="*70)
    print(f" Step 2: Extracting dictionary entries ({len(needs_extraction)} pages)...")
    print("="*70)
    processor = AdvancedPageProcessor(
        output_dir="data/extracted",
        reasoning_dir="data/reasoning_traces",
    )

    extracted_count = 0
    error_count = 0

    for page_num in needs_extraction:
        image_path = Path(f"data/processed_images/grammardictionar00riggrich_{page_num:04d}.jpg")

        if not image_path.exists():
            print(f"WARNING: Skipping page {page_num} - image not found")
            error_count += 1
            continue

        try:
            print(f"\n{'-'*70}")
            processor.extract_page(
                image_path=image_path,
                page_number=page_num,
                max_tokens=16000,
            )
            extracted_count += 1
        except Exception as e:
            print(f"ERROR: Error on page {page_num}: {e}")
            error_count += 1
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
    print("\nStatistics:")
    print(f"  Page range requested: {start}-{end} ({end-start+1} pages)")
    print(f"  Already extracted (skipped): {len(already_extracted)} pages")
    print(f"  Newly extracted: {extracted_count} pages")
    print(f"  Errors: {error_count} pages")
    print(f"  Total entries: {stats.get('total_entries', 0)}")
    print(f"  Avg entries/page: {stats.get('avg_entries_per_page', 0):.1f}")
    print("\nExtracted dictionary entries saved to:")
    print("  - data/extracted/page_*.json")
    print("\nNext steps:")
    print("  1. Run generate_synthetic_dakota.py to create Q&A pairs")
    print("  2. Run convert_extracted_to_chat.py to format for OpenAI fine-tuning")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Dakota dictionary entries (pages 95+) for SFT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Dictionary Structure:
  Pages 1-{GRAMMAR_PAGES}:    Grammar rules (already extracted for RL training)
  Pages {DICTIONARY_START_PAGE}-{DICTIONARY_END_PAGE}: Dictionary entries (extract these for SFT)

Examples:
  %(prog)s --test                    # Test on page {DICTIONARY_START_PAGE}
  %(prog)s --pages {DICTIONARY_START_PAGE}-114            # First 20 dictionary pages
  %(prog)s --pages {DICTIONARY_START_PAGE}-200            # More pages
  %(prog)s --all-dictionary          # All dictionary pages ({DICTIONARY_START_PAGE}-{DICTIONARY_END_PAGE})

Costs:
  ~$0.25 per page
  20 pages: ~$5
  All dictionary (346 pages): ~$86.50
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true",
                      help=f"Test on page {DICTIONARY_START_PAGE} (first dictionary page)")
    group.add_argument("--pages", type=str,
                      help="Page range (e.g., 95-110)")
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
            print("ERROR: Pages must be a range (e.g., 95-114)")
            sys.exit(1)

        start, end = map(int, args.pages.split("-"))

        if start < 1 or end > DICTIONARY_END_PAGE or start > end:
            print(f"ERROR: Invalid range. Must be 1-{DICTIONARY_END_PAGE} and start <= end")
            sys.exit(1)

        # Helpful suggestions
        if start < DICTIONARY_START_PAGE:
            print(f"\nNote: Dictionary entries start at page {DICTIONARY_START_PAGE}")
            print(f"   Pages 1-{GRAMMAR_PAGES} contain grammar rules (different structure)")

        num_pages = end - start + 1
        estimated_cost = num_pages * 0.25
        estimated_time = num_pages * 2  # minutes

        print(f"\nProcessing {num_pages} pages ({start}-{end})")
        print(f"Estimated cost: ${estimated_cost:.2f}")
        print(f"Estimated time: {estimated_time} minutes")

        confirm = input("\nContinue? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled")
            return

        process_range(start, end)

    elif args.all_dictionary:
        num_pages = DICTIONARY_END_PAGE - DICTIONARY_START_PAGE + 1
        estimated_cost = num_pages * 0.25
        estimated_hours = (num_pages * 2) / 60

        print(f"\nWARNING: Processing ALL dictionary pages ({DICTIONARY_START_PAGE}-{DICTIONARY_END_PAGE})")
        print(f"Total pages: {num_pages}")
        print(f"Estimated cost: ${estimated_cost:.2f}")
        print(f"Estimated time: {estimated_hours:.1f} hours")
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
