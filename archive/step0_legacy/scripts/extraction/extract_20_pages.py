#!/usr/bin/env python3
"""
Extract 20 Dakota Dictionary Pages with Claude Sonnet 4.5 (Non-Interactive)

Processes pages 109-128 automatically without confirmation prompts.
"""

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


def main():
    """Process pages 89-108 automatically."""

    start_page = 109
    end_page = 128
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
        return False

    print("\nOK API key found")

    # Show estimates
    estimated_cost = num_pages * 0.05
    estimated_time = num_pages * 45  # seconds per page
    print(f"\nEstimated cost: ${estimated_cost:.2f}")
    print(f"Estimated time: {estimated_time/60:.1f} minutes")
    print("\nStarting in 3 seconds...")
    time.sleep(3)

    # Step 1: Convert all pages
    print("\n" + "-"*70)
    print(f"Step 1: Converting Pages {start_page}-{end_page}")
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
            print(f"Page {page_num}: Already converted")
            converted_images.append((page_num, output_file))
        else:
            image = converter.convert_jp2_to_jpeg(page_file)
            converted_images.append((page_num, image))

    print(f"\nOK {len(converted_images)} images ready")

    # Step 2: Extract with Claude
    print("\n" + "-"*70)
    print("Step 2: Extracting Dictionary Entries")
    print("-"*70)

    processor = ClaudePageProcessor(
        output_dir="data/extracted",
        reasoning_dir="data/reasoning_traces",
    )

    successful = 0
    failed = 0
    total_entries = 0
    start_time = time.time()

    for page_num, image_path in converted_images:
        try:
            page_start = time.time()
            print(f"\n[{successful + failed + 1}/{len(converted_images)}] Page {page_num}...", end=" ", flush=True)

            extraction = processor.extract_page(
                image_path=image_path,
                page_number=page_num,
                max_tokens=16000,
                page_context=f"Dictionary page {page_num}",
            )

            elapsed = time.time() - page_start
            num_entries = len(extraction.get('entries', []))
            total_entries += num_entries

            print(f"{num_entries} entries ({elapsed:.1f}s)")
            successful += 1

        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1
            continue

    total_time = time.time() - start_time

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
        print("OK Datasets created")

    except Exception as e:
        print(f"WARNING: {e}")

    # Summary
    print("\n" + "="*70)
    print(" EXTRACTION COMPLETE")
    print("="*70)
    print(f"\nPages processed: {successful}/{len(converted_images)}")
    print(f"Failed: {failed}")
    print(f"Total entries: {total_entries}")
    print(f"Average: {total_entries/successful if successful > 0 else 0:.1f} entries/page")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average: {total_time/successful if successful > 0 else 0:.1f} seconds/page")

    print("\nOutput:")
    print("  - data/extracted/page_*.json")
    print("  - data/reasoning_traces/page_*_claude_response.txt")
    print("  - data/training_datasets/")

    return successful > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
