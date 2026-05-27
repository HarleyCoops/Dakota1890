#!/usr/bin/env python3
"""
Dakota Grammar Extraction Pipeline
Extracts grammar rules from pages 1-88 for RL training
"""

import base64
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dakota_extraction.tools.image_converter import ImageConverter
from dakota_extraction.core.grammar_extraction_prompt import build_grammar_extraction_prompt

# Load environment
load_dotenv()

DEFAULT_EXTRACTION_MODEL = os.getenv("DAKOTA_EXTRACTION_MODEL") or os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-6"


def build_grammar_extraction_prompt_with_context(page_number: int):
    """Build specialized prompt for grammar rule extraction."""
    # Use the more detailed prompt from grammar_extraction_prompt.py
    base_prompt = build_grammar_extraction_prompt(
        page_context=f"""CRITICAL: This is page {page_number} from the 1890 Dakota Grammar section. 

Extract ALL grammatical information including:
- ANY morphological patterns (affixes, word formation)
- ANY phonological rules (sound changes, vowel harmony)
- ANY syntactic patterns (word order, sentence structure)
- ANY interlinear examples (Dakota text with word-by-word glosses)
- ANY linguistic terminology that describes grammatical structures

Even if this page appears to be:
- Introductory material
- Ethnographic content
- Historical notes
- Cultural descriptions

STILL extract ANY grammatical patterns, linguistic rules, or morphological examples found within the text.

Look for:
- Dakota words with grammatical explanations
- Examples of word formation
- Patterns of affixation
- Sound change rules
- Sentence structure examples
- Any linguistic terminology that describes how Dakota works grammatically

If you find Dakota words being analyzed linguistically, extract the grammatical patterns even if they're embedded in non-grammar content."""
    )
    return base_prompt


def extract_grammar_page_with_claude(image_path: Path, page_number: int, model: str = DEFAULT_EXTRACTION_MODEL) -> dict:
    """Extract grammar rules from a single page using Claude."""

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Read and encode image
    print(f"  Reading {image_path.name}...")
    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    # Build prompt
    prompt = build_grammar_extraction_prompt_with_context(page_number)

    print(f"  Sending to Claude model {model}...")

    # Extract with Claude
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encoded
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }]
    )

    # Parse response
    response_text = response.content[0].text

    # Extract JSON from response
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    try:
        extraction = json.loads(response_text)
        extraction["page_number"] = page_number
        
        # Handle schema differences - normalize field names
        # The detailed prompt might use "interlinear_examples" instead of "interlinear_texts"
        if "interlinear_examples" in extraction and "interlinear_texts" not in extraction:
            extraction["interlinear_texts"] = extraction.pop("interlinear_examples")
        
        # Ensure grammar_rules exists (might be empty but should exist)
        if "grammar_rules" not in extraction:
            extraction["grammar_rules"] = []
        
        return extraction
    except json.JSONDecodeError as e:
        print(f"  ERROR: Failed to parse JSON: {e}")
        print(f"  Response preview: {response_text[:500]}")
        return {
            "page_number": page_number,
            "grammar_rules": [],
            "interlinear_texts": [],
            "linguistic_terms": [],
            "error": str(e),
            "raw_response": response_text[:2000]  # Store first 2000 chars for debugging
        }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Dakota grammar rules from pages 1-88"
    )
    parser.add_argument(
        "--pages",
        type=str,
        default="10-88",  # Skip front matter pages 1-9
        help="Page range to extract (e.g., 10-88, 10-20). Note: Pages 1-9 are front matter"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test on page 10 only"
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompts"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EXTRACTION_MODEL,
        help=f"Anthropic model to use for grammar extraction (default: {DEFAULT_EXTRACTION_MODEL})",
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print(" DAKOTA GRAMMAR EXTRACTION")
    print(" Pages 1-88: Grammar Rules and Linguistic Analysis")
    print("="*70)

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\nERROR: ANTHROPIC_API_KEY not set")
        print("Add to .env file: ANTHROPIC_API_KEY=your_key_here")
        return 1

    # Setup directories
    jp2_dir = Path("Dictionary/grammardictionar00riggrich_jp2")
    processed_dir = Path("data/processed_images")
    output_dir = Path("data/grammar_extracted")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine page range
    if args.test:
        start_page, end_page = 10, 10
        print("\nTEST MODE: Processing page 10 only")
    else:
        if "-" in args.pages:
            start_page, end_page = map(int, args.pages.split("-"))
        else:
            start_page = end_page = int(args.pages)

    num_pages = end_page - start_page + 1
    print(f"\nProcessing pages {start_page}-{end_page} ({num_pages} pages)")
    print(f"Extraction model: {args.model}")
    print(f"Estimated cost: ${num_pages * 0.25:.2f}")
    print(f"Estimated time: {num_pages * 2} minutes")

    if not args.test and not args.yes and num_pages > 10:
        confirm = input("\nContinue? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled")
            return 1

    # Convert JP2 to JPEG if needed
    print("\nStep 1: Converting images...")
    converter = ImageConverter(
        input_dir=str(jp2_dir),
        output_dir=str(processed_dir),
        quality=95
    )

    for page_num in range(start_page, end_page + 1):
        jp2_file = jp2_dir / f"grammardictionar00riggrich_{page_num:04d}.jp2"
        if jp2_file.exists():
            converter.convert_jp2_to_jpeg(jp2_file)

    # Extract grammar rules
    print("\nStep 2: Extracting grammar rules...")

    all_extractions = []
    failed_pages = []

    for page_num in range(start_page, end_page + 1):
        print(f"\nPage {page_num}/{end_page}")

        image_path = processed_dir / f"grammardictionar00riggrich_{page_num:04d}.jpg"

        if not image_path.exists():
            print("  WARNING: Image not found, skipping")
            continue

        try:
            extraction = extract_grammar_page_with_claude(image_path, page_num, model=args.model)

            # Save individual page
            output_file = output_dir / f"grammar_page_{page_num:03d}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extraction, f, indent=2, ensure_ascii=False)

            print(f"  Extracted {len(extraction.get('grammar_rules', []))} rules")
            print(f"  Found {len(extraction.get('interlinear_texts', []))} interlinear texts")
            print(f"  Saved to {output_file}")

            all_extractions.append(extraction)

        except Exception as e:
            print(f"  ERROR: {e}")
            failed_pages.append(page_num)
            continue

    # Save combined output
    combined_file = output_dir / f"grammar_combined_{start_page}-{end_page}.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump({
            "page_range": f"{start_page}-{end_page}",
            "total_pages": len(all_extractions),
            "pages": all_extractions
        }, f, indent=2, ensure_ascii=False)

    # Statistics
    total_rules = sum(len(e.get('grammar_rules', [])) for e in all_extractions)
    total_examples = sum(
        sum(len(r.get('examples', [])) for r in e.get('grammar_rules', []))
        for e in all_extractions
    )
    total_interlinear = sum(len(e.get('interlinear_texts', [])) for e in all_extractions)

    print("\n" + "="*70)
    print(" EXTRACTION COMPLETE")
    print("="*70)
    print(f"\nPages processed: {len(all_extractions)}")
    print(f"Pages failed: {len(failed_pages)}")
    print(f"Grammar rules: {total_rules}")
    print(f"Example sentences: {total_examples}")
    print(f"Interlinear texts: {total_interlinear}")
    print(f"\nOutput directory: {output_dir}/")
    print(f"Combined file: {combined_file}")
    if failed_pages:
        print(f"Failed pages: {failed_pages}")
    print("\nNext step:")
    print("  python organize_grammar_for_rl.py --input data/grammar_extracted/")
    print()
    return 1 if failed_pages else 0


if __name__ == "__main__":
    raise SystemExit(main())
