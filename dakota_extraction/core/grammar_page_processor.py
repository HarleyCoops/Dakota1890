"""
Grammar Page Processor for Claude API

Processes Dakota grammar pages (1-88) using Claude
with the specialized grammar extraction prompt.
"""

import anthropic
import base64
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import re

from dakota_extraction.core.grammar_extraction_prompt import (
    build_grammar_extraction_prompt,
    build_focused_rule_extraction_prompt
)
from dakota_extraction.schemas.grammar_schema import (
    GrammarPage,
    GrammarRule,
    InterlinearExample,
    MorphologicalTransformation,
    GrammarRuleType,
    TaskDifficulty
)

DEFAULT_ANTHROPIC_MODEL = os.getenv("DAKOTA_EXTRACTION_MODEL") or os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-6"


class GrammarPageProcessor:
    """
    Process Dakota grammar pages using Claude

    Optimized for extracting testable grammar rules for RL training
    """

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_ANTHROPIC_MODEL):
        """
        Initialize the grammar page processor

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be provided or set in environment")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

    def process_page(
        self,
        image_path: Path,
        page_number: int,
        page_context: str = "",
        focus_rule_type: Optional[str] = None,
        max_tokens: int = 16000
    ) -> GrammarPage:
        """
        Process a single grammar page

        Args:
            image_path: Path to page image (JPEG)
            page_number: Page number
            page_context: Optional context (e.g., "Chapter III: Verb Forms")
            focus_rule_type: Optional focus on specific rule type
            max_tokens: Max tokens for response

        Returns:
            GrammarPage object with extracted rules
        """
        # Read and encode image
        image_bytes = image_path.read_bytes()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        # Build prompt
        if focus_rule_type:
            prompt = build_focused_rule_extraction_prompt(focus_rule_type)
        else:
            prompt = build_grammar_extraction_prompt(page_context)

        # Call Claude API
        print(f"Processing page {page_number} with Claude model {self.model}...")
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
                                "media_type": "image/jpeg",
                                "data": encoded_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        # Extract JSON from response
        response_text = response.content[0].text
        extracted_data = self._parse_response(response_text)

        # Convert to GrammarPage object
        grammar_page = self._build_grammar_page(
            extracted_data,
            page_number,
            str(image_path)
        )

        return grammar_page

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from Claude's response

        Handles responses with or without markdown code blocks
        """
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try without code blocks
            json_str = response_text

        try:
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Response text: {response_text[:500]}...")

            # Fallback: return minimal structure
            return {
                "page_metadata": {
                    "page_number": None,
                    "extraction_error": str(e)
                },
                "grammar_rules": [],
                "interlinear_examples": []
            }

    def _build_grammar_page(
        self,
        data: Dict[str, Any],
        page_number: int,
        source_image: str
    ) -> GrammarPage:
        """
        Convert extracted JSON to GrammarPage object
        """
        metadata = data.get("page_metadata", {})

        # Parse grammar rules
        grammar_rules = []
        for rule_data in data.get("grammar_rules", []):
            try:
                rule = self._parse_grammar_rule(rule_data, page_number)
                grammar_rules.append(rule)
            except Exception as e:
                print(f"Error parsing grammar rule: {e}")
                print(f"Rule data: {rule_data}")

        # Parse interlinear examples
        interlinear_examples = []
        for example_data in data.get("interlinear_examples", []):
            try:
                example = self._parse_interlinear_example(example_data)
                interlinear_examples.append(example)
            except Exception as e:
                print(f"Error parsing interlinear example: {e}")

        return GrammarPage(
            page_number=page_number,
            chapter=metadata.get("chapter"),
            section_title=metadata.get("section_title"),
            grammar_rules=grammar_rules,
            interlinear_examples=interlinear_examples,
            linguistic_notes=data.get("linguistic_notes"),
            source_image=source_image,
            quality_issues=metadata.get("quality_issues"),
            extraction_confidence=data.get("extraction_confidence", 1.0)
        )

    def _parse_grammar_rule(self, rule_data: Dict[str, Any], page_number: int) -> GrammarRule:
        """Parse grammar rule from JSON data"""

        # Parse rule type
        rule_type_str = rule_data.get("rule_type", "morphology")
        try:
            rule_type = GrammarRuleType(rule_type_str)
        except ValueError:
            rule_type = GrammarRuleType.MORPHOLOGY

        # Parse difficulty
        difficulty_str = rule_data.get("difficulty", "intermediate")
        try:
            difficulty = TaskDifficulty(difficulty_str)
        except ValueError:
            difficulty = TaskDifficulty.INTERMEDIATE

        # Parse transformations
        transformations = []
        for trans_data in rule_data.get("transformations", []):
            trans = MorphologicalTransformation(
                base_form=trans_data.get("base_form", ""),
                transformed_form=trans_data.get("transformed_form", ""),
                affixes=trans_data.get("affixes", []),
                gloss_base=trans_data.get("gloss_base", ""),
                gloss_transformed=trans_data.get("gloss_transformed", ""),
                special_chars=trans_data.get("special_chars", []),
                phonological_changes=trans_data.get("phonological_changes")
            )
            transformations.append(trans)

        # Generate rule_id if not present
        rule_id = rule_data.get("rule_id")
        if not rule_id:
            rule_name_slug = rule_data.get("rule_name", "unknown").lower().replace(" ", "_")[:30]
            rule_id = f"page_{page_number:03d}_{rule_name_slug}"

        return GrammarRule(
            rule_id=rule_id,
            rule_type=rule_type,
            rule_name=rule_data.get("rule_name", ""),
            description=rule_data.get("description", ""),
            pattern=rule_data.get("pattern", ""),
            constraints=rule_data.get("constraints", []),
            exceptions=rule_data.get("exceptions", []),
            transformations=transformations,
            page_number=page_number,
            difficulty=difficulty,
            testable=rule_data.get("testable", True),
            verification_criteria=rule_data.get("verification_criteria", []),
            reward_function=rule_data.get("reward_function")
        )

    def _parse_interlinear_example(self, example_data: Dict[str, Any]) -> InterlinearExample:
        """Parse interlinear example from JSON data"""
        return InterlinearExample(
            dakota_text=example_data.get("dakota_text", ""),
            word_glosses=example_data.get("word_glosses", []),
            english_translation=example_data.get("english_translation", ""),
            dakota_words=example_data.get("dakota_words"),
            morpheme_breakdown=example_data.get("morpheme_breakdown"),
            special_characters_found=example_data.get("special_characters_found", [])
        )

    def process_page_range(
        self,
        image_dir: Path,
        start_page: int,
        end_page: int,
        output_dir: Path,
        page_context_map: Optional[Dict[int, str]] = None
    ) -> List[GrammarPage]:
        """
        Process a range of grammar pages

        Args:
            image_dir: Directory containing page images
            start_page: First page to process
            end_page: Last page to process (inclusive)
            output_dir: Directory to save JSON outputs
            page_context_map: Optional mapping of page_number → context string

        Returns:
            List of GrammarPage objects
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        grammar_pages = []

        for page_num in range(start_page, end_page + 1):
            # Find image file
            image_path = self._find_page_image(image_dir, page_num)
            if not image_path:
                print(f"Warning: Image for page {page_num} not found")
                continue

            # Get context if available
            context = ""
            if page_context_map:
                context = page_context_map.get(page_num, "")

            # Process page
            try:
                grammar_page = self.process_page(image_path, page_num, context)
                grammar_pages.append(grammar_page)

                # Save to JSON
                output_path = output_dir / f"grammar_page_{page_num:03d}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(grammar_page.to_json())

                print(
                    f"OK Page {page_num}: {len(grammar_page.grammar_rules)} rules, "
                    f"{len(grammar_page.interlinear_examples)} examples"
                )

            except Exception as e:
                print(f"ERROR processing page {page_num}: {e}")

        return grammar_pages

    def _find_page_image(self, image_dir: Path, page_num: int) -> Optional[Path]:
        """Find image file for a given page number"""
        # Try common naming patterns
        patterns = [
            f"grammardictionar00riggrich_{page_num:04d}.jpg",
            f"page_{page_num:03d}.jpg",
            f"page_{page_num:04d}.jpg",
            f"{page_num:03d}.jpg",
            f"{page_num:04d}.jpg"
        ]

        for pattern in patterns:
            path = image_dir / pattern
            if path.exists():
                return path

        return None


if __name__ == "__main__":
    # Test grammar page processor
    processor = GrammarPageProcessor()

    # Test on a single page
    test_image = Path("data/processed_images/grammardictionar00riggrich_0061.jpg")

    if test_image.exists():
        print("Testing grammar extraction on page 61...")
        grammar_page = processor.process_page(
            test_image,
            page_number=61,
            page_context="Chapter IX: Interlinear Translations - Parable of the Prodigal Son"
        )

        print("\n" + "="*80)
        print("EXTRACTION RESULTS")
        print("="*80)
        print(f"Page: {grammar_page.page_number}")
        print(f"Chapter: {grammar_page.chapter}")
        print(f"Grammar Rules: {len(grammar_page.grammar_rules)}")
        print(f"Interlinear Examples: {len(grammar_page.interlinear_examples)}")
        print(f"Confidence: {grammar_page.extraction_confidence}")

        # Show first rule
        if grammar_page.grammar_rules:
            print("\n" + "="*80)
            print("FIRST GRAMMAR RULE")
            print("="*80)
            print(grammar_page.grammar_rules[0].to_json())

        # Show first interlinear example
        if grammar_page.interlinear_examples:
            print("\n" + "="*80)
            print("FIRST INTERLINEAR EXAMPLE")
            print("="*80)
            example = grammar_page.interlinear_examples[0]
            print(f"Dakota: {example.dakota_text}")
            print(f"Glosses: {example.word_glosses}")
            print(f"English: {example.english_translation}")

        # Generate RL tasks
        print("\n" + "="*80)
        print("GENERATED RL TASKS (first 3)")
        print("="*80)
        tasks = grammar_page.generate_all_rl_tasks()[:3]
        for i, task in enumerate(tasks, 1):
            print(f"\nTask {i}:")
            print(json.dumps(task, indent=2, ensure_ascii=False))
    else:
        print(f"Test image not found: {test_image}")
        print("Please run image conversion first or adjust the path")
