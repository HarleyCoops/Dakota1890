"""Claude page processor for Dakota dictionary extraction."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import base64

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed")
    print("Install with: pip install anthropic")
    raise

from dakota_extraction.schemas.dictionary_schema import (
    DictionaryEntry,
    validate_entry,
    expand_pos,
)
from dakota_extraction.core.extraction_prompt import build_extraction_prompt

DEFAULT_ANTHROPIC_MODEL = os.getenv("DAKOTA_EXTRACTION_MODEL") or os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-6"


class ClaudePageProcessor:
    """Process dictionary pages with Claude."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: str = "data/extracted",
        reasoning_dir: str = "data/reasoning_traces",
        model: str = DEFAULT_ANTHROPIC_MODEL,
    ):
        """
        Initialize Claude processor.

        Args:
            api_key: Anthropic API key (defaults to env ANTHROPIC_API_KEY)
            output_dir: Directory for extraction output
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

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reasoning_dir.mkdir(parents=True, exist_ok=True)

    def extract_page(
        self,
        image_path: Path,
        page_number: int,
        max_tokens: int = 16000,
        page_context: str = "",
    ) -> Dict[str, Any]:
        """
        Extract structured dictionary entries from a page.

        Args:
            image_path: Path to page image
            page_number: Page number
            max_tokens: Maximum output tokens
            page_context: Optional context about this page

        Returns:
            Extraction with DictionaryEntry objects
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"\n{'='*70}")
        print(f"Processing Page {page_number}: {image_path.name}")
        print(f"{'='*70}")

        # Read and encode image
        image_data = self._encode_image(image_path)

        # Build specialized prompt
        prompt = build_extraction_prompt(page_context)

        print(f"Analyzing with Claude model {self.model}...")
        print("Using Dakota dictionary specialized prompt...")

        # Call Claude API
        try:
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
            extraction = self._parse_response(
                response_text,
                page_number,
                image_path.name,
            )

            # Add metadata
            extraction["metadata"] = {
                "page_number": page_number,
                "image_path": str(image_path),
                "processed_at": datetime.now().isoformat(),
                "model": self.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

            # Validate entries
            self._validate_extraction(extraction)

            # Save extraction
            self._save_extraction(page_number, extraction)

            # Save response (no separate reasoning trace, it's in the JSON)
            self._save_response(page_number, response_text, extraction)

            print(f"OK Extracted {len(extraction.get('entries', []))} entries")
            print(f"OK Input tokens: {response.usage.input_tokens}")
            print(f"OK Output tokens: {response.usage.output_tokens}")

            return extraction

        except Exception as e:
            print(f"ERROR during API call: {e}")
            raise

    def _encode_image(self, image_path: Path) -> Dict[str, str]:
        """Encode image to base64."""
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

    def _parse_response(
        self,
        response_text: str,
        page_number: int,
        source_image: str,
    ) -> Dict[str, Any]:
        """Parse Claude response into structured format."""
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

            raw_data = json.loads(json_text)

            # Convert to DictionaryEntry objects
            entries = []
            for i, entry_data in enumerate(raw_data.get("entries", [])):
                # Generate entry ID
                entry_data["entry_id"] = f"page_{page_number:03d}_entry_{i+1:03d}"
                entry_data["page_number"] = page_number
                entry_data["source_image"] = source_image

                # Expand POS if abbreviated
                if entry_data.get("part_of_speech"):
                    entry_data["pos_full"] = expand_pos(
                        entry_data["part_of_speech"]
                    )

                # Create DictionaryEntry object
                try:
                    entry = DictionaryEntry(**entry_data)
                    entries.append(entry.to_dict())
                except Exception as e:
                    print(f"  WARNING: Could not create entry {i+1}: {e}")
                    # Keep raw data
                    entries.append(entry_data)

            return {
                "page_metadata": raw_data.get("page_metadata", {}),
                "entries": entries,
                "raw_response": response_text,
            }

        except json.JSONDecodeError as e:
            print(f"WARNING: JSON parsing failed: {e}")
            print(f"Raw response:\n{response_text[:500]}...")
            return {
                "page_metadata": {"error": str(e)},
                "entries": [],
                "raw_response": response_text,
            }

    def _validate_extraction(self, extraction: Dict[str, Any]) -> None:
        """Validate extracted entries and print warnings."""
        print(f"\n{'-'*70}")
        print("VALIDATION")
        print(f"{'-'*70}")

        total = len(extraction.get("entries", []))
        valid_count = 0
        warnings = []

        for entry_dict in extraction.get("entries", []):
            # Convert back to DictionaryEntry for validation
            try:
                entry = DictionaryEntry(**entry_dict)
                is_valid, issues = validate_entry(entry)

                if is_valid:
                    valid_count += 1
                else:
                    for issue in issues:
                        warnings.append(f"{entry.headword}: {issue}")

            except Exception as e:
                warnings.append(f"Entry validation error: {e}")

        print(f"Total entries: {total}")
        print(f"Valid entries: {valid_count}")
        print(f"Warnings: {len(warnings)}")

        if warnings:
            print("\nValidation Issues:")
            for warning in warnings[:10]:  # Show first 10
                print(f"  WARNING: {warning}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more")

    def _save_extraction(self, page_number: int, extraction: Dict[str, Any]) -> None:
        """Save extraction to disk."""
        output_path = self.output_dir / f"page_{page_number:03d}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extraction, f, indent=2, ensure_ascii=False)
        print(f"OK Saved to: {output_path}")

    def _save_response(
        self,
        page_number: int,
        response_text: str,
        extraction: Dict[str, Any],
    ) -> None:
        """Save Claude's full response."""
        response_path = self.reasoning_dir / f"page_{page_number:03d}_claude_response.txt"
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(f"Page {page_number} - Claude model {self.model} Response\n")
            f.write(f"{'='*70}\n\n")
            f.write(response_text)
        print(f"OK Saved response to: {response_path}")

    def display_sample_entries(self, extraction: Dict[str, Any], num: int = 3) -> None:
        """Display sample entries for review."""
        entries = extraction.get("entries", [])
        if not entries:
            print("\nNo entries to display")
            return

        print(f"\n{'='*70}")
        print(f"SAMPLE ENTRIES (showing {min(num, len(entries))} of {len(entries)})")
        print(f"{'='*70}")

        for entry_dict in entries[:num]:
            print(f"\n{entry_dict.get('headword', 'N/A')}")
            print(f"  POS: {entry_dict.get('part_of_speech', 'N/A')}")
            if entry_dict.get("derived_from"):
                print(f"  From: {entry_dict['derived_from']}")
            print(f"  Definition: {entry_dict.get('definition_primary', 'N/A')}")
            if entry_dict.get("inflected_forms"):
                print(f"  Forms: {', '.join(entry_dict['inflected_forms'])}")
            print(f"  Confidence: {entry_dict.get('confidence', 0):.2f}")
            print(f"  Column: {entry_dict.get('column', '?')}")


def main():
    """Test with Claude."""
    from dakota_extraction.tools.image_converter import ImageConverter

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        print("\nAdd to your .env file:")
        print("  ANTHROPIC_API_KEY=your_key_here")
        return

    # Convert first dictionary page
    print("Step 1: Converting page 89...")
    converter = ImageConverter(
        input_dir="Dictionary/grammardictionar00riggrich_jp2",
        output_dir="data/processed_images",
    )

    page_89 = Path("Dictionary/grammardictionar00riggrich_jp2/grammardictionar00riggrich_0089.jp2")
    if page_89.exists():
        first_image = converter.convert_jp2_to_jpeg(page_89)
    else:
        print("ERROR: Page 89 not found")
        return

    # Extract with Claude
    print("\nStep 2: Extracting with Claude...")
    processor = ClaudePageProcessor(
        output_dir="data/extracted",
        reasoning_dir="data/reasoning_traces",
    )

    extraction = processor.extract_page(
        image_path=first_image,
        page_number=89,
        max_tokens=16000,
        page_context="First dictionary page - entries begin here after grammar section",
    )

    # Display results
    processor.display_sample_entries(extraction, num=5)


if __name__ == "__main__":
    main()
