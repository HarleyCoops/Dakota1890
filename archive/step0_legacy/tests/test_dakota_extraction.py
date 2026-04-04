"""
Test Dakota Character Extraction

This script tests whether the VLM correctly captures Dakota special characters
like ć, š, ŋ, ḣ, ṡ from historical grammar/dictionary images.
"""

import os
import sys
from pathlib import Path
import json
import base64

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed")
    print("Install with: pip install anthropic")
    sys.exit(1)

from blackfeet_extraction.core.dakota_extraction_prompt import build_dakota_extraction_prompt


def test_dakota_extraction(image_path: Path, max_tokens: int = 16000):
    """
    Test extraction on a Dakota grammar page using Claude Sonnet 4.5.

    Args:
        image_path: Path to image file
        max_tokens: Maximum tokens for response
    """
    print("\n" + "="*80)
    print(" DAKOTA CHARACTER EXTRACTION TEST")
    print("="*80)
    print(f"\nImage: {image_path.name}")
    print(f"Max tokens: {max_tokens}")
    print()

    # Initialize client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in environment")
        print("Set it in your .env file or export it:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Build extraction prompt
    page_context = "This is page from Dakota Grammar and Dictionary. Focus on preserving special characters: ć, š, ŋ, ḣ, ṡ, ó, á"
    prompt = build_dakota_extraction_prompt(page_context)

    print(" Analyzing image with Claude Sonnet 4.5...")
    print("   (Optimized for Dakota character preservation)")
    print()

    # Read and encode image
    with open(image_path, 'rb') as f:
        image_data = base64.standard_b64encode(f.read()).decode('utf-8')

    # Extract with Claude
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }]
    )

    result = {
        'text': response.content[0].text,
        'response_tokens': response.usage.output_tokens,
        'input_tokens': response.usage.input_tokens
    }

    # Display results
    print("\n" + "="*80)
    print(" EXTRACTION RESULTS")
    print("="*80)

    print(f"\n Input tokens: {result['input_tokens']}")
    print(f" Response tokens: {result['response_tokens']}")
    print()

    # Try to parse as JSON
    try:
        extracted_data = json.loads(result['text'])
        print(" Successfully parsed JSON output\n")

        # Check for special characters
        print(" SPECIAL CHARACTERS DETECTED:")
        print("-" * 80)

        if 'interlinear_entries' in extracted_data:
            all_special_chars = set()
            for entry in extracted_data['interlinear_entries']:
                if 'special_characters_found' in entry:
                    all_special_chars.update(entry['special_characters_found'])

            if all_special_chars:
                print(f"Found: {', '.join(sorted(all_special_chars))}")
                print()
            else:
                print("No special characters listed (check extraction)")
                print()

        # Display sample entries
        if 'interlinear_entries' in extracted_data and extracted_data['interlinear_entries']:
            print(" SAMPLE EXTRACTED ENTRIES:")
            print("-" * 80)

            for i, entry in enumerate(extracted_data['interlinear_entries'][:3], 1):
                print(f"\nEntry {i}:")
                print(f"  Blackfoot: {entry.get('blackfoot_text', 'N/A')}")
                print(f"  Glosses:   {' '.join(entry.get('word_glosses', []))}")
                print(f"  English:   {entry.get('english_translation', 'N/A')}")
                print(f"  Confidence: {entry.get('confidence', 'N/A')}")

                # Highlight special characters
                special = entry.get('special_characters_found', [])
                if special:
                    print(f"  Special chars in this entry: {', '.join(special)}")

        # Display vocabulary items
        if 'vocabulary_items' in extracted_data and extracted_data['vocabulary_items']:
            print("\n\n VOCABULARY EXTRACTED:")
            print("-" * 80)

            for item in extracted_data['vocabulary_items'][:5]:
                word = item.get('blackfoot_word', 'N/A')
                gloss = item.get('gloss', 'N/A')
                chars = item.get('special_chars', [])

                char_display = f" [{', '.join(chars)}]" if chars else ""
                print(f"  {word:<20} = {gloss}{char_display}")

        # Save full results
        output_path = Path("data/test_extraction_blackfoot.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)

        print(f"\n\n Full extraction saved to: {output_path}")

    except json.JSONDecodeError as e:
        print("  Warning: Could not parse as JSON")
        print(f"Error: {e}")
        print("\nRaw response:")
        print("-" * 80)
        print(result['text'])

    print("\n" + "="*80)
    print(" TEST COMPLETE")
    print("="*80)
    print()

    # Character validation summary
    print(" CHARACTER VALIDATION:")
    print("-" * 80)
    print("Check the extracted text above for these Dakota characters:")
    print("   ć (c-acute)     - in Wićášta, mićú")
    print("   š (s-caron)     - in Wićášta, wašte")
    print("   ŋ (eng)         - in éiŋhiŋtku, toŋaŋa")
    print("   ó (o-acute)     - in Wióni")
    print("   á, é, í, ú      - various pitch accents")
    print("   ḣ, ṡ            - dotted consonants (if present)")
    print()
    print("If these characters appear correctly above, the VLM extraction is working! ")
    print()


if __name__ == "__main__":
    # Skip if in offline mode for CI
    if os.getenv("OFFLINE") == "1":
        print("OFFLINE mode: Skipping Dakota extraction test.")
        sys.exit(0)

    # Test on first Dakota dictionary page
    test_image = Path("data/processed_images/grammardictionar00riggrich_0089.jpg")

    if not test_image.exists():
        print(f" Test image not found: {test_image}")
        print("\nAvailable images:")
        images_dir = Path("data/processed_images")
        if images_dir.exists():
            for img in images_dir.glob("*.jpg"):
                print(f"  {img}")
        else:
            print(f"  Directory not found: {images_dir}")
        sys.exit(1)

    test_dakota_extraction(test_image, max_tokens=16000)
