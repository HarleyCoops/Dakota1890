#!/usr/bin/env python3
"""
Test Dakota Grammar Extraction with Claude Sonnet 4.5

Tests if VLM correctly captures Dakota special characters (ć, š, ŋ, ḣ, ṡ)
from the processed grammar images without needing Tesseract training.
"""

import base64
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from blackfeet_extraction.core.dakota_extraction_prompt import build_dakota_extraction_prompt

# Load environment
load_dotenv()

# Check for anthropic package
try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed")
    print("\nInstall with:")
    print("  pip install anthropic")
    sys.exit(1)


def test_dakota_extraction(image_path: Path):
    """Test extraction on a Dakota grammar page."""

    print("\n" + "="*80)
    print(" DAKOTA CHARACTER EXTRACTION TEST - CLAUDE SONNET 4.5")
    print("="*80)
    print(f"\nImage: {image_path.name}")
    print()

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in environment")
        print("\nAdd to your .env file:")
        print("  ANTHROPIC_API_KEY=your_key_here")
        print("\nGet your key from: https://console.anthropic.com/")
        return

    print("OK API key found")

    # Initialize Claude client
    client = anthropic.Anthropic(api_key=api_key)

    # Read and encode image
    print(f"\nReading image: {image_path}")
    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    # Build Dakota-specific extraction prompt
    page_context = "This page contains Dakota interlinear translations. Focus on preserving special characters: ć, š, ŋ, ḣ, ṡ, ó, á"
    prompt = build_dakota_extraction_prompt(page_context)

    print("\nAnalyzing with Claude Sonnet 4.5...")
    print("  Model: claude-sonnet-4-5-20250929")
    print("  Max tokens: 16000")
    print("  Focus: Dakota character preservation")
    print()

    # Call Claude API
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": encoded,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        # Extract text response
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        print("OK Response received")
        print(f"  Input tokens: {response.usage.input_tokens}")
        print(f"  Output tokens: {response.usage.output_tokens}")

        # Parse and display results
        print("\n" + "="*80)
        print(" EXTRACTION RESULTS")
        print("="*80)

        # Try to parse as JSON
        try:
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()

            extracted_data = json.loads(json_text)
            print("\nOK Successfully parsed JSON output\n")

            # Check for special characters
            print("SPECIAL CHARACTERS DETECTED:")
            print("-" * 80)

            all_special_chars = set()
            if 'interlinear_entries' in extracted_data:
                for entry in extracted_data['interlinear_entries']:
                    if 'special_characters_found' in entry:
                        all_special_chars.update(entry['special_characters_found'])

            # Save to file first (Unicode safe)
            output_dir = Path("data/dakota_test")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / "dakota_extraction_test.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=2, ensure_ascii=False)

            response_path = output_dir / "dakota_extraction_test_response.txt"
            with open(response_path, 'w', encoding='utf-8') as f:
                f.write(response_text)

            if all_special_chars:
                print(f"Found {len(all_special_chars)} special character types (saved to file)")
                print()
            else:
                print("No special characters listed (check extraction)")
                print()

            # Display sample entries
            if 'interlinear_entries' in extracted_data and extracted_data['interlinear_entries']:
                print("SAMPLE BLACKFOOT ENTRIES:")
                print("-" * 80)

                for i, entry in enumerate(extracted_data['interlinear_entries'][:3], 1):
                    print(f"\nEntry {i}:")
                    # Use ASCII replacement for console
                    blackfoot = entry.get('blackfoot_text', 'N/A')
                    print("  Blackfoot text extracted (see JSON for full Unicode)")
                    print(f"  Length: {len(blackfoot)} chars")

                    print(f"  English:   {entry.get('english_translation', 'N/A')}")
                    print(f"  Confidence: {entry.get('confidence', 'N/A')}")

                    # Highlight special characters found
                    special = entry.get('special_characters_found', [])
                    if special:
                        print(f"  Special char count: {len(special)}")

            # Display vocabulary
            if 'vocabulary_items' in extracted_data and extracted_data['vocabulary_items']:
                print("\n\nVOCABULARY EXTRACTED:")
                print("-" * 80)

                print(f"Extracted {len(extracted_data['vocabulary_items'])} vocabulary items")
                print("(See JSON file for full Unicode text)")

            print(f"\n\nOK Full extraction saved to: {output_path}")
            print(f"OK Full response saved to: {response_path}")

        except json.JSONDecodeError as e:
            print("\nWARNING: Could not parse as JSON")
            print(f"Error: {e}")
            print("\nRaw response (first 1000 chars):")
            print("-" * 80)
            print(response_text[:1000])

        print("\n" + "="*80)
        print(" CHARACTER VALIDATION")
        print("="*80)
        print("\nExpected Dakota characters: c-acute, s-caron, eng, o-acute, dotted consonants")
        print("See extracted JSON file for full Unicode preservation")
        print()
        print("  SUCCESS - VLM extraction completed!")
        print("  Check data/dakota_test/dakota_extraction_test.json for full results")
        print()

    except Exception as e:
        print(f"\nERROR during API call: {e}")
        raise


def main():
    # Test on first Dakota grammar page
    test_image = Path("data/processed_images/grammardictionar00riggrich_0089.jpg")

    if not test_image.exists():
        print(f"ERROR: Test image not found: {test_image}")
        print("\nAvailable images in data/processed_images/:")
        images_dir = Path("data/processed_images")
        if images_dir.exists():
            for img in sorted(images_dir.glob("*.jpg")):
                print(f"  {img.name}")
        sys.exit(1)

    test_dakota_extraction(test_image)


if __name__ == "__main__":
    main()
