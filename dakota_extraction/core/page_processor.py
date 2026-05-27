"""
Dakota Dictionary Page Processor

This module processes individual dictionary pages using Claude
to extract structured linguistic data with detailed responses.

Inspired by the Stoney Nakoda language preservation project by @harleycoops.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed")
    print("Install with: pip install anthropic")
    raise

import base64

DEFAULT_ANTHROPIC_MODEL = os.getenv("DAKOTA_EXTRACTION_MODEL") or os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-6"


class PageProcessor:
    """Process individual dictionary pages to extract structured linguistic data."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: str = "data/extracted",
        reasoning_dir: str = "data/reasoning_traces",
        model: str = DEFAULT_ANTHROPIC_MODEL,
    ):
        """
        Initialize the page processor.

        Args:
            api_key: Anthropic API key (defaults to env var ANTHROPIC_API_KEY)
            output_dir: Directory for structured extraction output
            reasoning_dir: Directory for reasoning traces
            model: Claude model to use
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.output_dir = Path(output_dir)
        self.reasoning_dir = Path(reasoning_dir)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reasoning_dir.mkdir(parents=True, exist_ok=True)

    def extract_page(
        self,
        image_path: Path,
        page_number: int,
        max_tokens: int = 8192,
    ) -> Dict[str, Any]:
        """
        Extract structured data from a single dictionary page.

        Args:
            image_path: Path to dictionary page image
            page_number: Page number for tracking
            max_tokens: Maximum output tokens for Claude

        Returns:
            Structured extraction with entries, metadata, and response
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"\n{'='*60}")
        print(f"Processing Page {page_number}: {image_path.name}")
        print(f"{'='*60}")

        # Craft extraction prompt
        prompt = self._build_extraction_prompt()

        # Encode image
        image_data = self._encode_image(image_path)

        print(f"Analyzing page with Claude model {self.model}...")
        
        # Call Claude API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_data["media_type"],
                                "data": image_data["data"],
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

        # Parse response
        extraction = self._parse_response(response_text, page_number)

        # Add metadata
        extraction["metadata"] = {
            "page_number": page_number,
            "image_path": str(image_path),
            "processed_at": datetime.now().isoformat(),
            "model": self.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        # Save response
        self._save_response(page_number, response_text, extraction)

        # Save extraction
        self._save_extraction(page_number, extraction)

        print(f" Extracted {len(extraction.get('entries', []))} entries")
        print(f" Input tokens: {response.usage.input_tokens}")
        print(f" Output tokens: {response.usage.output_tokens}")

        return extraction

    def _build_extraction_prompt(self) -> str:
        """Build the extraction prompt for the model."""
        return """Analyze this historical Dakota language dictionary page from 1890.

**Your Task:**
Extract ALL dictionary entries in structured format. For each entry, identify:

1. **Dakota Word** (headword in Dakota language)
2. **English Translation** (meaning in English)
3. **Part of Speech** (noun, verb, adjective, etc., if indicated)
4. **Grammatical Info** (any notes about grammar, conjugation, etc.)
5. **Example Sentences** (if present)
6. **Etymology or Notes** (any additional information)

**Important Guidelines:**
- Preserve original spelling exactly as printed
- Handle diacritics and special characters carefully
- Identify the structure: is this single-column or multi-column?
- Track your confidence for each extraction
- Note any damaged, unclear, or ambiguous text
- Preserve the sequential order of entries

**Output Format:**
Return a JSON object with this structure:

```json
{
  "layout": "single-column" or "multi-column",
  "entries": [
    {
      "entry_id": "unique_id",
      "dakota": "the Dakota word",
      "english": "the English translation",
      "pos": "part of speech or null",
      "grammatical_notes": "any grammar info or null",
      "examples": ["example sentences"],
      "etymology": "etymological notes or null",
      "confidence": 0.0-1.0,
      "notes": "any extraction concerns or null"
    }
  ],
  "page_notes": "overall observations about this page"
}
```

**Think carefully about:**
- Typography conventions (how are headwords vs definitions formatted?)
- Abbreviations used (n. for noun, v. for verb, etc.)
- How to distinguish between different entries
- Whether there are subsections or related entries

Provide ONLY the JSON output, no other text."""

    def _parse_response(self, response_text: str, page_number: int) -> Dict[str, Any]:
        """
        Parse the model's response into structured data.

        Args:
            response_text: Raw response from model
            page_number: Page number for error tracking

        Returns:
            Parsed extraction dictionary
        """
        try:
            # Try to extract JSON from response
            # Look for JSON code block or raw JSON
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

            extraction = json.loads(json_text)

            # Add entry IDs if missing
            for i, entry in enumerate(extraction.get("entries", [])):
                if "entry_id" not in entry:
                    entry["entry_id"] = f"page_{page_number:03d}_entry_{i+1:03d}"

            return extraction

        except json.JSONDecodeError as e:
            print(f"  Warning: Could not parse JSON response: {e}")
            print(f"Raw response:\n{response_text[:500]}...")

            # Return fallback structure
            return {
                "layout": "unknown",
                "entries": [],
                "page_notes": f"JSON parsing failed: {e}",
                "raw_response": response_text,
            }

    def _save_extraction(self, page_number: int, extraction: Dict[str, Any]) -> None:
        """Save extraction to disk."""
        output_path = self.output_dir / f"page_{page_number:03d}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extraction, f, indent=2, ensure_ascii=False)
        print(f" Saved extraction to: {output_path}")

    def _encode_image(self, image_path: Path) -> Dict[str, str]:
        """Encode image to base64 for Claude API."""
        image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("utf-8")

        # Determine media type from extension
        ext = image_path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        media_type = media_types.get(ext, "image/jpeg")

        return {
            "data": encoded,
            "media_type": media_type,
        }

    def _save_response(
        self,
        page_number: int,
        response_text: str,
        extraction: Dict[str, Any],
    ) -> None:
        """Save Claude's full response for verification."""
        response_path = self.reasoning_dir / f"page_{page_number:03d}_claude_response.txt"
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(f"Page {page_number} - Claude model {self.model} Response\n")
            f.write(f"{'='*60}\n\n")
            f.write(response_text)
        print(f" Saved response to: {response_path}")

    def batch_extract(
        self,
        image_dir: Path,
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_tokens: int = 8192,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple pages in batch.

        Args:
            image_dir: Directory containing page images
            start_page: Starting page number
            end_page: Ending page number (None = all)
            max_tokens: Maximum output tokens per page

        Returns:
            List of extractions
        """
        image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg")) + sorted(image_dir.glob("*.png"))

        if end_page:
            image_paths = image_paths[start_page - 1 : end_page]
        else:
            image_paths = image_paths[start_page - 1 :]

        print(f"\n{'='*60}")
        print(f"BATCH EXTRACTION: {len(image_paths)} pages")
        print(f"{'='*60}\n")

        extractions = []
        for i, image_path in enumerate(image_paths, start=start_page):
            try:
                extraction = self.extract_page(image_path, i, max_tokens)
                extractions.append(extraction)
            except Exception as e:
                print(f" Error processing page {i}: {e}")
                # Continue with next page
                continue

        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE: {len(extractions)} pages processed")
        print(f"{'='*60}\n")

        return extractions


def main():
    """Example usage."""
    # Initialize processor
    processor = PageProcessor(
        output_dir="data/extracted",
        reasoning_dir="data/reasoning_traces",
    )

    # Example 1: Single page extraction
    example_image = Path("Public/Dictionary.jpeg")
    if example_image.exists():
        extraction = processor.extract_page(
            image_path=example_image,
            page_number=1,
            max_tokens=8192,  # Max tokens for detailed output
        )

        print("\n" + "="*60)
        print("EXTRACTION SUMMARY")
        print("="*60)
        print(f"Entries extracted: {len(extraction.get('entries', []))}")
        print(f"Layout detected: {extraction.get('layout')}")
        print(f"Page notes: {extraction.get('page_notes')}")

        if extraction.get('entries'):
            print("\nFirst entry:")
            first = extraction['entries'][0]
            print(f"  Dakota: {first.get('dakota')}")
            print(f"  English: {first.get('english')}")
            print(f"  Confidence: {first.get('confidence')}")

    # Example 2: Batch processing
    # Uncomment when you have multiple pages
    # dictionary_dir = Path("data/dictionary_pages")
    # if dictionary_dir.exists():
    #     processor.batch_extract(
    #         image_dir=dictionary_dir,
    #         start_page=1,
    #         end_page=10,
    #         max_tokens=8192,
    #     )


if __name__ == "__main__":
    main()
