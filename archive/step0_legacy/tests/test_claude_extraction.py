#!/usr/bin/env python3
"""
Test Dakota Dictionary Extraction with Claude Sonnet 4.5

This uses Anthropic's API directly - more reliable than OpenRouter.
Processes 20 consecutive dictionary pages (109-128).
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from blackfeet_extraction.tools.image_converter import ImageConverter
from blackfeet_extraction.core.claude_page_processor import ClaudePageProcessor
from blackfeet_extraction.datasets.training_dataset_builder import TrainingDatasetBuilder

# Load environment
load_dotenv()

# Check for anthropic package
if importlib.util.find_spec("anthropic") is None:
    print("ERROR: anthropic package not installed")
    print("\nInstall with:")
    print("  pip install anthropic")
    sys.exit(1)


def process_pages(start_page: int = 89, end_page: int = 108):
    """Process multiple pages with Claude."""

    num_pages = end_page - start_page + 1

    print("\n" + "="*70)
    print(" DAKOTA DICTIONARY EXTRACTION - CLAUDE SONNET 4.5")
    print(f" Processing {num_pages} pages ({start_page}-{end_page})")
    print("="*70)

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\nERROR: ANTHROPIC_API_KEY not set")
        print("\nAdd to your .env file:")
        print("  ANTHROPIC_API_KEY=your_key_here")
        print("\nGet your key from: https://console.anthropic.com/")
        return False

    print("\nOK API key found")

    # Estimate cost
    estimated_cost = num_pages * 0.05
    estimated_time = num_pages * 45  # seconds per page
    print(f"\nEstimated cost: ${estimated_cost:.2f}")
    print(f"Estimated time: {estimated_time/60:.1f} minutes")

    confirm = input(f"\nProcess {num_pages} pages? [y/N]: ")
    if confirm.lower() != 'y':
        print("Cancelled")
        return False

    # Step 1: Convert all pages
    print("\n" + "-"*70)
    print(f"Step 1: Converting Pages {start_page}-{end_page} (JP2 to JPEG)")
    print("-"*70)

    converter = ImageConverter(
        input_dir="dictionary/grammardictionar00riggrich_jp2",
        output_dir="data/processed_images",
        quality=95,
    )

    converted_images = []
    for page_num in range(start_page, end_page + 1):
        page_file = Path(f"dictionary/grammardictionar00riggrich_jp2/grammardictionar00riggrich_{page_num:04d}.jp2")

        if not page_file.exists():
            print(f"WARNING: Page {page_num} not found, skipping")
            continue

        # Check if already converted
        output_file = Path(f"data/processed_images/grammardictionar00riggrich_{page_num:04d}.jpg")
        if output_file.exists():
            print(f"Page {page_num}: Already converted, using existing")
            converted_images.append((page_num, output_file))
        else:
            image = converter.convert_jp2_to_jpeg(page_file)
            converted_images.append((page_num, image))

    print(f"\nOK {len(converted_images)} images ready for extraction")

    # Step 2: Extract with Claude
    print("\n" + "-"*70)
    print("Step 2: Extracting Dictionary Entries with Claude")
    print("-"*70)
    print("\nUsing:")
    print("  - Model: claude-sonnet-4-5-20250929")
    print("  - Max tokens: 16000 per page")
    print("  - Specialized Dakota dictionary prompt")
    print()

    processor = ClaudePageProcessor(
        output_dir="data/extracted",
        reasoning_dir="data/reasoning_traces",
    )

    successful = 0
    failed = 0
    total_entries = 0

    for page_num, image_path in converted_images:
        try:
            print(f"\n[{successful + failed + 1}/{len(converted_images)}] Processing page {page_num}...")

            start_time = time.time()

            extraction = processor.extract_page(
                image_path=image_path,
                page_number=page_num,
                max_tokens=16000,
                page_context=f"Dictionary page {page_num}",
            )

            elapsed = time.time() - start_time
            num_entries = len(extraction.get('entries', []))
            total_entries += num_entries

            print(f"OK Page {page_num}: {num_entries} entries in {elapsed:.1f}s")
            successful += 1

        except Exception as e:
            print(f"ERROR Page {page_num}: {e}")
            failed += 1
            continue

    # Step 3: Build datasets
    print("\n" + "-"*70)
    print("Step 3: Building Training Datasets")
    print("-"*70)

    try:
        builder = TrainingDatasetBuilder(
            extraction_dir="data/extracted",
            output_dir="data/training_datasets",
        )

        builder.build_all_datasets()
        print("\nOK Datasets created")

    except Exception as e:
        print(f"WARNING: Dataset building skipped: {e}")

    # Summary
    print("\n" + "="*70)
    print(" EXTRACTION COMPLETE")
    print("="*70)
    print(f"\nPages processed: {successful}/{len(converted_images)}")
    print(f"Failed: {failed}")
    print(f"Total entries: {total_entries}")
    print(f"Average entries/page: {total_entries/successful if successful > 0 else 0:.1f}")

    print("\nOutput files:")
    print("  - Extractions: data/extracted/page_*.json")
    print("  - Responses: data/reasoning_traces/page_*_claude_response.txt")
    print("  - Datasets: data/training_datasets/")

    print("\nNext steps:")
    print("  1. Review data/extracted/ for quality")
    print("  2. Check sample entries in page_089.json")
    print("  3. Process more pages if quality is good")
    print("  4. Use datasets in data/training_datasets/ for model training")

    return successful > 0


def main():
    # Skip if in offline mode for CI
    if os.getenv("OFFLINE") == "1":
        print("OFFLINE mode: Skipping Claude extraction test.")
        sys.exit(0)

    success = process_pages(start_page=109, end_page=128)

    if success:
        print("\n" + "="*70)
        print(" SUCCESS!")
        print("="*70)
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print(" FAILED - Check errors above")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()
