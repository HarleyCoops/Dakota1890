"""
Test inference with Claude Sonnet 4.5 via Anthropic API.
This script tests both text and image analysis capabilities.
"""

import os
from pathlib import Path
import base64

from dotenv import load_dotenv

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed")
    print("Install with: pip install anthropic")
    exit(1)

# Load environment variables
load_dotenv()

def test_text_inference():
    """Test basic text reasoning."""
    print("=" * 60)
    print("TEST 1: Text Reasoning")
    print("=" * 60)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(" ANTHROPIC_API_KEY not found in .env file!")
        return False

    print(f" API Key loaded: {api_key[:20]}...")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        print(" Client initialized")

        print("\nSending query: 'Explain what a vision-language model is in one sentence.'")
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": "Explain what a vision-language model is in one sentence."
            }]
        )

        print("\n" + "-" * 60)
        print("RESPONSE:")
        print("-" * 60)
        print(response.content[0].text)
        print("-" * 60)
        print(f"Input tokens: {response.usage.input_tokens}")
        print(f"Output tokens: {response.usage.output_tokens}")

        print("\n Text inference test PASSED!\n")
        return True

    except Exception as e:
        print(f"\n Text inference test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_image_inference():
    """Test image analysis with the Dakota dictionary."""
    print("=" * 60)
    print("TEST 2: Image Analysis (Dakota Dictionary)")
    print("=" * 60)

    image_path = Path("Public/Dictionary.jpeg")

    if not image_path.exists():
        print(f" Image not found at {image_path}")
        print("Please ensure the dictionary image exists in the Public/ folder.")
        return False

    print(f" Image found: {image_path}")

    api_key = os.getenv("ANTHROPIC_API_KEY")

    try:
        client = anthropic.Anthropic(api_key=api_key)

        print("\nAnalyzing dictionary image...")
        print("Query: 'What language is this dictionary for? List 3 words you can see.'")

        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.standard_b64encode(f.read()).decode('utf-8')

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=800,
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
                        "text": "What language is this dictionary for? List 3 words you can see with their translations if visible."
                    }
                ]
            }]
        )

        print("\n" + "-" * 60)
        print("ANALYSIS:")
        print("-" * 60)
        print(response.content[0].text)
        print("-" * 60)
        print(f"Input tokens: {response.usage.input_tokens}")
        print(f"Output tokens: {response.usage.output_tokens}")

        print("\n Image inference test PASSED!\n")
        return True

    except Exception as e:
        print(f"\n Image inference test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    # Skip if in offline mode for CI
    if os.getenv("OFFLINE") == "1":
        print("OFFLINE mode: Skipping inference tests.")
        return

    print("\n")
    print("*" * 60)
    print("  Claude Sonnet 4.5 Inference Test")
    print("*" * 60)
    print("\n")

    # Test 1: Text inference
    text_passed = test_text_inference()

    # Test 2: Image inference
    image_passed = test_image_inference()

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Text Inference:  {' PASSED' if text_passed else ' FAILED'}")
    print(f"Image Inference: {' PASSED' if image_passed else ' FAILED'}")
    print("=" * 60)

    if text_passed and image_passed:
        print("\n All tests passed! You're ready to use the model.")
        print("\nNext steps:")
        print("  1. Try modifying the prompts in this script")
        print("  2. Experiment with different max_tokens values")
        print("  3. Analyze your own images")
        print("  4. Run extraction scripts: python test_dakota_claude.py")
    else:
        print("\n  Some tests failed. Check the error messages above.")
        print("\nCommon issues:")
        print("  - Invalid API key (check your .env file)")
        print("  - Network connectivity")
        print("  - Missing dependencies (run: pip install -r requirements.txt)")
        print("  - Rate limits (wait a moment and try again)")


if __name__ == "__main__":
    main()
